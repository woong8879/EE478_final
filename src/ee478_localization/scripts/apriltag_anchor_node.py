#!/usr/bin/env python3
"""ee478_localization/apriltag_anchor_node.py

AprilTag world-anchor for VIO drift correction.

Detects AprilTags (cv2.aruco, no extra deps) in the IR image, looks up each
tag's KNOWN world position, and back-computes the drone's TRUE pose -- then
publishes /landmark_anchor_pose, which vio_bridge_node blends SMOOTHLY into
/mavros/vision_pose/pose (anchor_blend_rate m/s, no teleport).

Geometry (same idea as landmark_anchor_publisher, but tags are exact):
    tag_obs_in_map = drone_pose_map + R(yaw) @ tag_in_body
    drone_true_map = tag_known_map  - R(yaw) @ tag_in_body

Frames
------
  world (config) : x FORWARD, y RIGHT, z UP   (FRU, as the tags were given)
  map / VIO      : x forward,  y LEFT,  z up   (FLU, ROS / mavros)
  -> world->map is a y-flip (x, -y, z). Tag positions are converted to the map
     frame ON LOAD, so all anchor math is in the map frame == the vision_pose
     frame the bridge expects.

  optical (cam)  : x RIGHT, y DOWN, z FORWARD  (solvePnP tvec is here)
  -> optical->body(FLU): x_b = z_o, y_b = -x_o, z_b = -y_o, then + cam mount.

Assumptions
-----------
- Drone takes off at the world origin facing +x_world (the course direction),
  so the map frame and world frame share an origin and differ only by the
  y-flip. Mostly level (yaw-only body->map rotation), z~0.7 m gentle flight.
- Position (xyz) correction only; yaw correction would need the tags' world
  orientations too (out of scope here).
"""
import json
import math
import threading

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from mavros_msgs.msg import State
from std_msgs.msg import String
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform)


_ARUCO_DICTS = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag25h9":  cv2.aruco.DICT_APRILTAG_25h9,
    "tag16h5":  cv2.aruco.DICT_APRILTAG_16h5,
}


def _yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class AprilTagAnchor(object):
    def __init__(self):
        rospy.init_node("apriltag_anchor")
        self.lock = threading.Lock()

        self.image_topic = rospy.get_param(
            "~image_topic", "/camera/infra1/image_rect_raw")
        self.info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/infra1/camera_info")
        self.drone_pose_topic = rospy.get_param(
            "~drone_pose_topic", "/mavros/local_position/pose")
        self.anchor_topic = rospy.get_param(
            "~anchor_topic", "/landmark_anchor_pose")
        self.world_frame = rospy.get_param("~world_frame", "map")

        self.tag_size = float(rospy.get_param("~tag_size_m", 0.20))
        self.min_period_s = float(rospy.get_param("~min_period_s", 0.3))
        self.max_range_m = float(rospy.get_param("~max_range_m", 8.0))
        # Multi-tag fusion: when several tags are visible we median-reject any
        # estimate further than this from the median, then average the rest.
        self.fuse_tol = float(rospy.get_param("~fuse_outlier_tol_m", 0.5))
        # optical -> base_link via the ACTUAL camera-mount TF (base_link ->
        # camera_link from px4_real, then camera_link -> *_optical from the
        # RealSense driver). This carries the real mount tilt (this drone's cam
        # looks ~12.8 deg UP, pitch -0.2231) + offset; a hand-rolled axis swap
        # ignores the tilt and biases the anchor (the bug we just hit).
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # Tag positions are given DIRECTLY in the map frame (FLU, +y = left).
        # NO axis flip (an earlier version wrongly assumed +y=right input).
        tags_world = rospy.get_param("~tags", {})
        self.tags_map = {}
        for tid, p in tags_world.items():
            self.tags_map[int(tid)] = (float(p[0]), float(p[1]), float(p[2]))
        if not self.tags_map:
            rospy.logwarn("[apriltag_anchor] no ~tags loaded -- nothing to anchor on")

        fam = rospy.get_param("~tag_family", "tag36h11")
        dict_id = _ARUCO_DICTS.get(fam, cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        try:
            params = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
            self._new_api = True
        except AttributeError:                       # OpenCV < 4.7 fallback
            self._params = cv2.aruco.DetectorParameters_create()
            self._new_api = False

        # Tag object points (marker plane, centre origin), aruco corner order:
        # top-left, top-right, bottom-right, bottom-left.
        s = self.tag_size * 0.5
        self.obj_pts = np.array([[-s, s, 0.0], [s, s, 0.0],
                                 [s, -s, 0.0], [-s, -s, 0.0]],
                                dtype=np.float32)

        self.bridge = CvBridge()
        self.K = None
        self.D = None
        self.drone_pose = None
        self.armed = False
        self.last_pub_t = 0.0
        self.last_proc_t = 0.0
        self.pub_count = 0
        # Only EMIT anchor corrections while ARMED (in flight). On the ground /
        # pre-arm the drone sits at the known world origin, so an anchor there
        # only RISKS corrupting the EKF (e.g. a hand-held test tag, which the
        # node assumes is at its fixed course position -> bogus pose -> false
        # "airborne" climb -> premature course start). Detection still logs.
        self.require_armed = bool(rospy.get_param("~require_armed", True))
        # emit_anchor=false -> DETECTION-ONLY mode: publish /apriltag/detections
        # (ids + ranges, for the delivery-camera trigger) but never an anchor
        # correction. Lets the node run always while the EKF-touching anchor
        # path stays opt-in (apriltag_anchor:=true).
        self.emit_anchor = bool(rospy.get_param("~emit_anchor", True))

        self.pub = rospy.Publisher(self.anchor_topic, PoseStamped, queue_size=5)
        # Always-on detection stream: JSON '[{"id":271,"range":0.43}, ...]'
        # (range = straight-line camera->tag distance, m).
        self.pub_det = rospy.Publisher("/apriltag/detections", String,
                                       queue_size=5)
        rospy.Subscriber(self.info_topic, CameraInfo, self.on_info, queue_size=1)
        rospy.Subscriber(self.drone_pose_topic, PoseStamped,
                         self.on_pose, queue_size=5)
        rospy.Subscriber("/mavros/state", State, self.on_state, queue_size=5)
        rospy.Subscriber(self.image_topic, Image, self.on_image, queue_size=1)

        rospy.loginfo(
            "[apriltag_anchor] %s + %s -> %s | %d tags, size %.3f m, family %s",
            self.image_topic, self.drone_pose_topic, self.anchor_topic,
            len(self.tags_map), self.tag_size, fam)

    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            # image is RECTIFIED (image_rect_raw) -> no distortion
            self.D = np.zeros(5, dtype=np.float64)

    def on_pose(self, msg):
        with self.lock:
            self.drone_pose = msg

    def on_state(self, msg):
        self.armed = bool(msg.armed)

    def _detect(self, gray):
        if self._new_api:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self._params)
        return corners, ids

    @staticmethod
    def _median(v):
        s = sorted(v)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    def _fuse(self, ests):
        """ests = [(x, y, z, tag_id), ...] drone-pose estimates, one per tag.
        Median-reject outliers (wrong id / bad PnP), average the inliers."""
        if len(ests) == 1:
            e = ests[0]
            return (e[0], e[1], e[2]), [e[3]]
        mx = self._median([e[0] for e in ests])
        my = self._median([e[1] for e in ests])
        mz = self._median([e[2] for e in ests])
        tol = self.fuse_tol
        inl = [e for e in ests
               if abs(e[0] - mx) <= tol and abs(e[1] - my) <= tol
               and abs(e[2] - mz) <= tol]
        if not inl:
            inl = ests
        n = float(len(inl))
        return ((sum(e[0] for e in inl) / n,
                 sum(e[1] for e in inl) / n,
                 sum(e[2] for e in inl) / n),
                [e[3] for e in inl])

    def on_image(self, msg):
        if self.K is None:
            rospy.logwarn_throttle(
                5.0, "[apriltag_anchor] waiting for camera_info on %s",
                self.info_topic)
            return
        t_now = rospy.Time.now().to_sec()
        if t_now - self.last_proc_t < self.min_period_s:
            return
        self.last_proc_t = t_now

        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[apriltag_anchor] cv_bridge: %s", e)
            return

        corners, ids = self._detect(gray)
        if ids is None or len(ids) == 0:
            return
        # Detection log (works on the bench WITHOUT mavros) -- this confirms the
        # camera + tag pipeline before flight.
        seen = [int(t) for t in ids.flatten()]
        rospy.loginfo_throttle(1.0, "[apriltag_anchor] SAW tags %s", seen)

        # --- per-tag PnP (range) — needed for BOTH the detections stream and
        # the anchor. sights = [(tid, (xo,yo,zo) optical, rng), ...]
        sights = []
        for i, tid in enumerate(ids.flatten()):
            tid = int(tid)
            ok, _rvec, tvec = cv2.solvePnP(
                self.obj_pts, corners[i][0], self.K, self.D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            xo, yo, zo = float(tvec[0]), float(tvec[1]), float(tvec[2])
            rng = math.sqrt(xo * xo + yo * yo + zo * zo)
            if rng > self.max_range_m or zo <= 0.0:
                continue
            sights.append((tid, (xo, yo, zo), rng))

        # --- ALWAYS-ON detections stream, independent of the armed / pose /
        # emit_anchor gates. Consumers: delivery-camera trigger (id + range)
        # and the gate-quiz checker (body = tag position in base_link FLU,
        # included when the camera-mount TF is up; null on the bare bench).
        if sights:
            dets = []
            for tid, (xo, yo, zo), rng in sights:
                body = None
                pt = PointStamped()
                pt.header.frame_id = msg.header.frame_id
                pt.header.stamp = rospy.Time(0)
                pt.point.x, pt.point.y, pt.point.z = xo, yo, zo
                try:
                    pb = self.tf_buf.transform(pt, self.base_frame,
                                               rospy.Duration(0.02))
                    body = [round(pb.point.x, 3), round(pb.point.y, 3),
                            round(pb.point.z, 3)]
                except Exception:
                    pass
                dets.append({"id": tid, "range": round(rng, 3), "body": body})
            self.pub_det.publish(String(data=json.dumps(dets)))

        # ---------- anchor path (EKF-touching) ----------
        if not self.emit_anchor:
            return
        # SAFETY: do not emit corrections unless ARMED. Pre-arm the drone is at
        # the known origin; an anchor here only risks corrupting the EKF (e.g. a
        # hand-held test tag -> bogus pose -> false climb -> course starts before
        # arming). Detection above still publishes so the bench test works.
        if self.require_armed and not self.armed:
            rospy.loginfo_throttle(
                3.0, "[apriltag_anchor] not armed -> anchor HELD (detection only)")
            return

        with self.lock:
            pose = self.drone_pose
        if pose is None:
            rospy.logwarn_throttle(
                2.0, "[apriltag_anchor] tags seen but NO drone pose -- "
                "detection OK; anchoring needs /mavros/local_position/pose")
            return

        yaw = _yaw_from_quat(pose.pose.orientation)
        cy, sy = math.cos(yaw), math.sin(yaw)
        dx, dy, dz = (pose.pose.position.x, pose.pose.position.y,
                      pose.pose.position.z)

        ests = []       # one (x, y, z, tag_id) drone-pose estimate per tag
        for tid, (xo, yo, zo), rng in sights:
            tag_map = self.tags_map.get(tid)
            if tag_map is None:
                continue
            # optical -> base_link (FLU) through the REAL camera-mount TF
            # (carries the ~12.8 deg up tilt + offset). rospy.Time(0) = latest
            # static transform.
            pt = PointStamped()
            pt.header.frame_id = msg.header.frame_id
            pt.header.stamp = rospy.Time(0)
            pt.point.x, pt.point.y, pt.point.z = xo, yo, zo
            try:
                pb = self.tf_buf.transform(pt, self.base_frame,
                                           rospy.Duration(0.05))
            except Exception as e:
                rospy.logwarn_throttle(
                    5.0, "[apriltag_anchor] TF %s->%s not ready: %s",
                    msg.header.frame_id, self.base_frame, e)
                continue
            xb, yb, zb = pb.point.x, pb.point.y, pb.point.z
            # body -> map (drone yaw rotation)
            rwx = cy * xb - sy * yb
            rwy = sy * xb + cy * yb
            rwz = zb
            ests.append((tag_map[0] - rwx, tag_map[1] - rwy,
                         tag_map[2] - rwz, tid))

        if not ests:
            return
        # FUSE every visible tag (median-reject outliers, average inliers) ->
        # more robust + less noisy than trusting a single tag.
        drone_true, used = self._fuse(ests)

        out = PoseStamped()
        out.header.stamp = (msg.header.stamp if msg.header.stamp.to_sec() > 0
                            else rospy.Time.now())
        out.header.frame_id = self.world_frame
        out.pose.position.x = drone_true[0]
        out.pose.position.y = drone_true[1]
        out.pose.position.z = drone_true[2]
        out.pose.orientation = pose.pose.orientation   # xyz correction only
        self.pub.publish(out)

        with self.lock:
            self.last_pub_t = t_now
            self.pub_count += 1
        rospy.loginfo_throttle(
            1.0, "[apriltag_anchor] %d tag(s) %s -> drone_true="
            "(%.2f,%.2f,%.2f)  est=(%.2f,%.2f,%.2f)  emitted=%d",
            len(used), used, drone_true[0], drone_true[1], drone_true[2],
            dx, dy, dz, self.pub_count)


if __name__ == "__main__":
    try:
        AprilTagAnchor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
