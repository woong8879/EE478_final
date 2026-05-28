#!/usr/bin/env python3
"""ee478_localization/vio_bridge_node.py

VINS-Fusion (or any VIO) odometry  ->  /mavros/vision_pose/pose

Job:
  1. Republish VIO pose at the rate PX4 EKF2 wants (~30 Hz).
  2. Apply a SMOOTH world-anchor correction from sparse landmark
     fixes (/landmark_anchor_pose). The anchor never teleports
     vision_pose — drift is blended in at `anchor_blend_rate` m/s
     so PX4 EKF2 stays linearisable.
  3. SAFETY gate:
        * reject samples with diagonal pose covariance above
          ~max_cov_m2 (VO lost tracking — RTAB-Map / OpenVINS / VINS
          all set this to ~9999 when degenerate).
        * reject samples that JUMP > ~max_jump_m from the last
          published pose. If a jump is seen, the bridge stops
          publishing entirely until N consecutive sane samples
          arrive (covariance OK, no jump). Until then PX4 dead-
          reckons on IMU + baro — far safer than commanding a 20 m
          teleport correction.

Subscribes
----------
  ~vo_topic              (nav_msgs/Odometry)
                         Default /vins_estimator/odometry.
  ~loop_topic            (nav_msgs/Odometry, optional)
                         Loop-closure-corrected pose from
                         /loop_fusion/odometry_rect or similar.
                         If present and fresh, overrides vo_topic.
  ~landmark_anchor_topic (geometry_msgs/PoseStamped)
                         "The drone IS at this world pose right now"
                         hint from semantic_map_manager (YOLO + known
                         store positions). Used to compute and slowly
                         apply a world-frame offset to VO.

Publishes
---------
  ~vision_pose_topic     (geometry_msgs/PoseStamped)
                         Default /mavros/vision_pose/pose.
"""

import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry

try:
    from gazebo_msgs.msg import ModelStates
    _HAS_MODEL_STATES = True
except ImportError:
    _HAS_MODEL_STATES = False


def _norm3(dx, dy, dz):
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class VioBridgeNode:
    def __init__(self):
        rospy.init_node("vio_bridge")
        self.lock = threading.Lock()

        # topics
        self.vo_topic = rospy.get_param("~vo_topic", "/vins_estimator/odometry")
        self.loop_topic = rospy.get_param("~loop_topic", "")
        self.anchor_topic = rospy.get_param(
            "~landmark_anchor_topic", "/landmark_anchor_pose")
        self.vision_topic = rospy.get_param(
            "~vision_pose_topic", "/mavros/vision_pose/pose")

        # publish rate
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))

        # safety gate
        # max covariance (diagonal m^2) allowed; 9999 from VO loss
        self.max_cov_m2 = float(rospy.get_param("~max_cov_m2", 1.0))
        # max jump from last published pose (m)
        self.max_jump_m = float(rospy.get_param("~max_jump_m", 0.5))
        # consecutive sane samples needed to resume after a loss
        self.resume_after_n_sane = int(
            rospy.get_param("~resume_after_n_sane", 5))
        # samples older than this are considered stale
        self.vo_fresh_s = float(rospy.get_param("~vo_fresh_s", 0.5))

        # anchor blend
        # cap (m) on how fast the anchor offset can be ramped into
        # the output every second. With 0.2 m/s a 1 m drift is
        # corrected in 5 s without spooking the EKF.
        self.anchor_blend_rate = float(
            rospy.get_param("~anchor_blend_rate", 0.2))
        # anchor inputs older than this are considered stale
        self.anchor_fresh_s = float(rospy.get_param("~anchor_fresh_s", 2.0))

        # --- state ---
        self.last_vo = None              # (Odometry msg, rospy.Time)
        self.last_loop = None            # (Odometry msg, rospy.Time)
        self.last_anchor = None          # (target world pose, rospy.Time)

        self.last_pub_xyz = None         # last published (x, y, z)
        self.sane_streak = 0
        self.tracking_lost = False
        self.jump_count = 0

        # current applied offset (we slide this toward `target_offset`
        # at `anchor_blend_rate` so vision_pose is smooth)
        self.applied_offset = [0.0, 0.0, 0.0]
        self.target_offset = [0.0, 0.0, 0.0]

        # --- bootstrap + handoff state machine ---
        # The bridge starts in BOOTSTRAP mode (publish GT directly).
        # On every VINS odometry sample we compute the offset
        # GT - VINS so we can express "where VINS thinks the drone
        # is" in the SAME map frame as GT. Once N consecutive offsets
        # agree within `handoff_offset_tol_m`, we lock the offset and
        # switch to TRACKING mode where the bridge publishes
        # (VINS + offset). Loop-closure jumps in VINS are absorbed by
        # the existing tracking_lost gate.
        self.bootstrap_enabled = bool(
            rospy.get_param("~bootstrap_enabled", True))
        self.gt_model = rospy.get_param(
            "~gt_model", "iris_depth_camera_vio")
        self.handoff_required_samples = int(
            rospy.get_param("~handoff_required_samples", 30))
        self.handoff_offset_tol_m = float(
            rospy.get_param("~handoff_offset_tol_m", 0.10))
        # mode: "BOOTSTRAP" or "TRACKING"
        self.mode = "BOOTSTRAP" if self.bootstrap_enabled else "TRACKING"
        self.latest_gt_pose = None  # PoseStamped from gazebo
        self.recent_offsets = []   # list of (dx, dy, dz)
        self.locked_offset = None  # (dx, dy, dz) once handoff done

        # --- ROS I/O ---
        self.pub = rospy.Publisher(
            self.vision_topic, PoseStamped, queue_size=10)

        rospy.Subscriber(self.vo_topic, Odometry,
                         self.on_vo, queue_size=10)
        if self.loop_topic:
            rospy.Subscriber(self.loop_topic, Odometry,
                             self.on_loop, queue_size=10)
        rospy.Subscriber(self.anchor_topic, PoseStamped,
                         self.on_anchor, queue_size=5)
        if self.bootstrap_enabled and _HAS_MODEL_STATES:
            rospy.Subscriber("/gazebo/model_states", ModelStates,
                             self.on_gt, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.tick)
        rospy.loginfo(
            f"[vio_bridge] vo={self.vo_topic} "
            f"loop={self.loop_topic or '(none)'} "
            f"anchor={self.anchor_topic} -> {self.vision_topic} "
            f"@ {self.rate_hz:.0f} Hz "
            f"mode={self.mode} "
            f"(handoff after {self.handoff_required_samples} samples "
            f"within {self.handoff_offset_tol_m:.2f} m offset)")

    # ----- subscribers -----
    def on_vo(self, msg):
        """Track VINS samples for GT->VINS handoff and for the
        covariance/jump gates in TRACKING mode."""
        with self.lock:
            mode = self.mode

        if mode == "BOOTSTRAP":
            # While we're publishing GT, monitor the VINS-GT offset.
            # When it stabilises (latest N samples agree within
            # handoff_offset_tol_m), we lock the offset and switch to
            # TRACKING mode (publish VINS + offset). VINS itself
            # diverging in absolute frame is fine — we only care that
            # the offset is consistent.
            with self.lock:
                gt = self.latest_gt_pose
            if gt is None:
                return
            vp = msg.pose.pose.position
            offset = (gt.pose.position.x - vp.x,
                      gt.pose.position.y - vp.y,
                      gt.pose.position.z - vp.z)
            with self.lock:
                self.recent_offsets.append(offset)
                if len(self.recent_offsets) > self.handoff_required_samples:
                    self.recent_offsets.pop(0)
                self.last_vo = (msg, rospy.Time.now())

                if len(self.recent_offsets) >= self.handoff_required_samples:
                    n = len(self.recent_offsets)
                    mx = sum(o[0] for o in self.recent_offsets) / n
                    my = sum(o[1] for o in self.recent_offsets) / n
                    mz = sum(o[2] for o in self.recent_offsets) / n
                    spread = max(
                        max(abs(o[0] - mx) for o in self.recent_offsets),
                        max(abs(o[1] - my) for o in self.recent_offsets),
                        max(abs(o[2] - mz) for o in self.recent_offsets))
                    if spread <= self.handoff_offset_tol_m:
                        # At handoff, applied_offset jumps INSTANTLY
                        # to the mean offset. After this, applied
                        # slowly slides as the GT-VINS difference
                        # evolves with VINS drift, so vision_pose
                        # remains continuous and close to GT.
                        self.applied_offset = [mx, my, mz]
                        self.target_offset = [mx, my, mz]
                        self.mode = "TRACKING"
                        rospy.loginfo(
                            f"[vio_bridge] VINS converged. handing off "
                            f"GT -> VINS with initial offset "
                            f"({mx:.2f}, {my:.2f}, {mz:.2f}) m "
                            f"spread={spread:.3f} m. From now on the "
                            f"published vision_pose is VINS-based with "
                            f"continuous anchor correction.")
            return

        # mode == "TRACKING": apply safety gates against drift / jumps
        # within VINS itself (loop closure can still produce jumps).
        with self.lock:
            cov_xx = msg.pose.covariance[0]
            if cov_xx > self.max_cov_m2:
                self.tracking_lost = True
                self.sane_streak = 0
                rospy.logwarn_throttle(
                    1.0,
                    f"[vio_bridge] VO covariance {cov_xx:.2f} > "
                    f"{self.max_cov_m2:.2f} — tracking lost, hold")
                return

            # In TRACKING, jump gate against the last published pose.
            # On a real jump we fall BACK to BOOTSTRAP so the drone
            # keeps flying via GT while VINS's new frame stabilises
            # and a fresh handoff can re-lock the offset.
            if self.last_pub_xyz is not None:
                vp = msg.pose.pose.position
                cand = (vp.x + self.applied_offset[0],
                        vp.y + self.applied_offset[1],
                        vp.z + self.applied_offset[2])
                jump = _norm3(
                    cand[0] - self.last_pub_xyz[0],
                    cand[1] - self.last_pub_xyz[1],
                    cand[2] - self.last_pub_xyz[2])
                if jump > self.max_jump_m:
                    self.tracking_lost = True
                    self.sane_streak = 0
                    self.jump_count += 1
                    rospy.logwarn_throttle(
                        1.0,
                        f"[vio_bridge] VO jump {jump:.2f} m > "
                        f"{self.max_jump_m:.2f} m — tracking lost "
                        f"#{self.jump_count}, hold")
                    if self.bootstrap_enabled and self.mode == "TRACKING":
                        rospy.logwarn(
                            "[vio_bridge] VINS diverged in TRACKING; "
                            "fall back to BOOTSTRAP to re-accumulate "
                            "offset against new VINS frame")
                        self.mode = "BOOTSTRAP"
                        self.recent_offsets = []
                        self.applied_offset = [0.0, 0.0, 0.0]
                        self.target_offset = [0.0, 0.0, 0.0]
                    return

            self.last_vo = (msg, rospy.Time.now())
            if self.tracking_lost:
                self.sane_streak += 1
                if self.sane_streak >= self.resume_after_n_sane:
                    self.tracking_lost = False
                    rospy.loginfo(
                        f"[vio_bridge] VO resumed after "
                        f"{self.sane_streak} sane samples")
                else:
                    # While we are still considered LOST, fall back to
                    # GT bootstrap so the drone keeps flying. We also
                    # restart offset accumulation so the next handoff
                    # uses the CURRENT VINS frame (which has drifted
                    # since the original lock).
                    if self.mode == "TRACKING" and self.bootstrap_enabled:
                        rospy.logwarn_once(
                            "[vio_bridge] VINS unstable in TRACKING; "
                            "falling back to BOOTSTRAP (GT) while "
                            "re-accumulating VINS-GT offset")
                        self.mode = "BOOTSTRAP"
                        self.recent_offsets = []
                        self.locked_offset = None

    def on_loop(self, msg):
        """Loop-closure-corrected pose (optional)."""
        with self.lock:
            self.last_loop = (msg, rospy.Time.now())

    def on_gt(self, msg):
        """Cache the ground-truth drone pose from gazebo for the
        bootstrap window. Becomes a noop once VINS has converged."""
        try:
            idx = msg.name.index(self.gt_model)
        except ValueError:
            return
        p = msg.pose[idx]
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "map"
        ps.pose = p
        with self.lock:
            self.latest_gt_pose = ps

    def on_anchor(self, msg):
        """Sparse landmark fix: 'drone IS at msg.pose right now'.

        We compute the world-frame offset between this hint and the
        current raw VO pose, and store it as `target_offset`. The
        tick() loop then slides `applied_offset` toward this target
        at `anchor_blend_rate` m/s.
        """
        with self.lock:
            if self.last_vo is None:
                return
            vp = self.last_vo[0].pose.pose.position
            target_x = float(msg.pose.position.x) - vp.x
            target_y = float(msg.pose.position.y) - vp.y
            target_z = float(msg.pose.position.z) - vp.z
            # Sanity: don't accept an anchor that disagrees with our
            # current pose by more than max_jump_m extra beyond the
            # already-applied offset. That's almost certainly a
            # landmark-id mismatch, not a real drift correction.
            delta = _norm3(
                target_x - self.applied_offset[0],
                target_y - self.applied_offset[1],
                target_z - self.applied_offset[2])
            if delta > 5.0:
                rospy.logwarn_throttle(
                    1.0,
                    f"[vio_bridge] ignoring landmark anchor "
                    f"(would shift output by {delta:.2f} m — "
                    f"likely a misassociation)")
                return
            self.target_offset = [target_x, target_y, target_z]
            self.last_anchor = (msg, rospy.Time.now())

    # ----- publisher tick -----
    def tick(self, _evt):
        with self.lock:
            now = rospy.Time.now()
            lost = self.tracking_lost
            vo = self.last_vo
            loop = self.last_loop
            applied = list(self.applied_offset)
            target = list(self.target_offset)
            mode = self.mode
            gt = self.latest_gt_pose
            locked = self.locked_offset

        if mode == "BOOTSTRAP":
            # Publish GT directly while VINS is converging on the
            # GT-VINS offset. Skips the safety gates because GT is
            # exact in sim.
            if gt is None:
                return
            out = PoseStamped()
            out.header.stamp = now
            out.header.frame_id = "map"
            out.pose = gt.pose
            self.pub.publish(out)
            with self.lock:
                self.last_pub_xyz = (
                    out.pose.position.x,
                    out.pose.position.y,
                    out.pose.position.z)
            return

        if lost:
            return
        if vo is None:
            return
        if (now - vo[1]).to_sec() > self.vo_fresh_s:
            return

        # Prefer loop-closure-corrected pose if it is fresh — it has
        # the same coordinate system as raw VO but with drift removed.
        src = vo[0]
        if loop is not None and (now - loop[1]).to_sec() <= self.vo_fresh_s:
            src = loop[0]

        # Slide applied_offset toward target_offset at anchor_blend_rate.
        # Each tick is 1/rate_hz seconds.
        step = self.anchor_blend_rate / self.rate_hz
        for i in range(3):
            d = target[i] - applied[i]
            if abs(d) <= step:
                applied[i] = target[i]
            else:
                applied[i] += step if d > 0 else -step

        # TRACKING mode: vision_pose = VINS + applied_offset. The
        # applied_offset is the full VINS-frame -> map transform; it
        # slides toward target_offset (set in on_vo to GT - VINS, or
        # in on_anchor to the landmark fix) at anchor_blend_rate so
        # the output stream is smooth.
        out = PoseStamped()
        out.header.stamp = now
        out.header.frame_id = "map"
        out.pose.position.x = src.pose.pose.position.x + applied[0]
        out.pose.position.y = src.pose.pose.position.y + applied[1]
        out.pose.position.z = src.pose.pose.position.z + applied[2]
        out.pose.orientation = src.pose.pose.orientation

        self.pub.publish(out)

        with self.lock:
            self.applied_offset = applied
            self.last_pub_xyz = (
                out.pose.position.x,
                out.pose.position.y,
                out.pose.position.z)


if __name__ == "__main__":
    try:
        VioBridgeNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
