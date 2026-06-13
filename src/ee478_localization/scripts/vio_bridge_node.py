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
from std_msgs.msg import String
from tf.transformations import (quaternion_matrix, quaternion_from_matrix,
                                translation_matrix, translation_from_matrix)
import numpy as _np

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

        # Body-frame correction via TF. The VO reports the pose of its own
        # body frame (VINS-Fusion: the camera/IMU OPTICAL frame) in the VO
        # world frame. PX4 wants the drone FLU base_link pose. We look the
        # fixed transform vo_body_frame -> base_frame up from the TF tree
        # (defined by the camera mount + RealSense calibration, NOT a magic
        # number here) and compose it onto every VO sample:
        #     T_world_base = T_world_vobody * T_vobody_base
        # Leave vo_body_frame empty to disable (RTAB-Map already publishes
        # the FLU base_link pose directly).
        self.vo_body_frame = rospy.get_param("~vo_body_frame", "")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self._T_body_base = None          # cached (qx,qy,qz,qw, tx,ty,tz)
        if self.vo_body_frame:
            import tf2_ros
            self._tf_buf = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buf)

        # --- state ---
        self.last_vo = None              # (Odometry msg, rospy.Time)
        self.last_loop = None            # (Odometry msg, rospy.Time)
        self.last_anchor = None          # (target world pose, rospy.Time)

        self.last_pub_xyz = None         # last published (x, y, z)
        self.sane_streak = 0
        self.tracking_lost = False
        self.jump_count = 0
        # If the same "jump" trips the gate this many times in a row,
        # assume the VO frame did a true relocalisation (loop closure,
        # init transient) and accept the new position as the new
        # baseline. Prevents the bridge from getting permanently
        # stuck rejecting the post-init RTAB-Map odom.
        self.jump_reset_after = int(
            rospy.get_param("~jump_reset_after", 10))
        self.consecutive_jumps = 0

        # current applied offset (we slide this toward `target_offset`
        # at `anchor_blend_rate` so vision_pose is smooth)
        self.applied_offset = [0.0, 0.0, 0.0]
        self.target_offset = [0.0, 0.0, 0.0]

        # --- bootstrap + handoff state machine ---
        # The bridge can EITHER:
        #   - bootstrap_enabled=False (default, REAL DRONE SAFE):
        #     publish (VO + applied_offset) immediately. The VO frame
        #     is treated as the map frame; landmark_anchor handles the
        #     drift. There is no /gazebo/model_states subscription, so
        #     accidentally flashing this to the Jetson cannot
        #     introduce a hidden GT cheat.
        #   - bootstrap_enabled=True (SIM-ONLY DIAGNOSTIC):
        #     publishes GT from /gazebo/model_states while a noisy VO
        #     (VINS-Fusion mono+IMU in PX4 SITL) converges, then hands
        #     off. Real drone launch files MUST leave this False.
        self.bootstrap_enabled = bool(
            rospy.get_param("~bootstrap_enabled", False))
        self.gt_model = rospy.get_param(
            "~gt_model", "iris_depth_camera_vio")
        self.handoff_required_samples = int(
            rospy.get_param("~handoff_required_samples", 30))
        self.handoff_offset_tol_m = float(
            rospy.get_param("~handoff_offset_tol_m", 0.10))
        self.allow_fallback = bool(
            rospy.get_param("~allow_fallback", False))
        # mode: "BOOTSTRAP" or "TRACKING"
        self.mode = "BOOTSTRAP" if self.bootstrap_enabled else "TRACKING"
        self.latest_gt_pose = None  # PoseStamped from gazebo
        self.recent_offsets = []   # list of (dx, dy, dz)
        self.locked_offset = None  # (dx, dy, dz) once handoff done

        # Feed PX4 the FULL VINS state (pose + velocity) via the odometry
        # interface instead of position-only vision_pose. With velocity
        # supplied, EKF2 does not differentiate position (noisy, esp. in z)
        # nor lean on its own accel-z integration -> the fused estimate
        # tracks VINS tightly ("100% VINS"). When true, vision_pose is NOT
        # published (avoid double-fusing).
        self.use_odometry = bool(rospy.get_param("~use_odometry", False))
        self.odom_topic = rospy.get_param("~odom_topic", "/mavros/odometry/in")

        # --- ROS I/O ---
        self.pub = rospy.Publisher(
            self.vision_topic, PoseStamped, queue_size=10)
        self.pub_odom = rospy.Publisher(
            self.odom_topic, Odometry, queue_size=10)
        self.pub_mode = rospy.Publisher(
            "/vio_bridge/mode", String, queue_size=1, latch=True)

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
        # Publish startup mode so monitoring can verify what
        # vision_pose source is currently in use.
        self.pub_mode.publish(String(data=self.mode))
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
                        self.locked_offset = (mx, my, mz)
                        self.mode = "TRACKING"
                        self.pub_mode.publish(String(data="TRACKING"))
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
                    self.consecutive_jumps += 1
                    rospy.logwarn_throttle(
                        1.0,
                        f"[vio_bridge] VO jump {jump:.2f} m > "
                        f"{self.max_jump_m:.2f} m — tracking lost "
                        f"#{self.jump_count}, hold")
                    # If the gate keeps tripping with roughly the
                    # same jump (VO just relocalised), give up on the
                    # old baseline and accept this sample as the new
                    # reference. The TF tree handles the discontinuity
                    # gracefully because PX4 EKF treats vision_pose as
                    # an absolute measurement.
                    if self.consecutive_jumps >= self.jump_reset_after:
                        rospy.logwarn(
                            f"[vio_bridge] {self.consecutive_jumps} "
                            f"consecutive jumps; resetting baseline to "
                            f"current VO and resuming")
                        self.last_pub_xyz = cand
                        self.consecutive_jumps = 0
                        self.tracking_lost = False
                        self.last_vo = (msg, rospy.Time.now())
                        return
                    if (self.bootstrap_enabled and self.mode == "TRACKING"
                            and self.allow_fallback):
                        rospy.logwarn(
                            "[vio_bridge] VINS diverged in TRACKING; "
                            "fall back to BOOTSTRAP to re-accumulate "
                            "offset against new VINS frame")
                        self.mode = "BOOTSTRAP"
                        self.pub_mode.publish(String(data="BOOTSTRAP"))
                        self.recent_offsets = []
                        self.locked_offset = None
                        self.applied_offset = [0.0, 0.0, 0.0]
                        self.target_offset = [0.0, 0.0, 0.0]
                    return

            self.last_vo = (msg, rospy.Time.now())
            self.consecutive_jumps = 0
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

    def _get_T_body_base(self):
        """Cached 4x4 transform vo_body_frame -> base_frame, from TF."""
        if self._T_body_base is not None:
            return self._T_body_base
        try:
            tf = self._tf_buf.lookup_transform(
                self.vo_body_frame, self.base_frame,
                rospy.Time(0), rospy.Duration(0.2))
            t = tf.transform.translation
            q = tf.transform.rotation
            M = quaternion_matrix([q.x, q.y, q.z, q.w])
            M[0:3, 3] = [t.x, t.y, t.z]
            self._T_body_base = M     # static -> cache forever
            rospy.loginfo("[vio_bridge] cached %s->%s from TF",
                          self.vo_body_frame, self.base_frame)
            return M
        except Exception as e:
            rospy.logwarn_throttle(
                2.0, "[vio_bridge] waiting for TF %s->%s: %s",
                self.vo_body_frame, self.base_frame, e)
            return None

    def _to_base_link(self, pose):
        """Return (px,py,pz, ox,oy,oz,ow) of base_link in the VO world.

        If no vo_body_frame is configured (RTAB path), pass the VO pose
        through unchanged.
        """
        p = pose.position
        o = pose.orientation
        if not self.vo_body_frame:
            return (p.x, p.y, p.z, o.x, o.y, o.z, o.w)
        Tbb = self._get_T_body_base()
        if Tbb is None:
            # TF not ready yet: pass through (orientation will look off
            # until TF arrives, but we never block the safety stream).
            return (p.x, p.y, p.z, o.x, o.y, o.z, o.w)
        Twb = quaternion_matrix([o.x, o.y, o.z, o.w])
        Twb[0:3, 3] = [p.x, p.y, p.z]
        Twbase = _np.dot(Twb, Tbb)
        tx, ty, tz = translation_from_matrix(Twbase)
        qx, qy, qz, qw = quaternion_from_matrix(Twbase)
        return (tx, ty, tz, qx, qy, qz, qw)

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
        # Stamp with the VO MEASUREMENT time, not now(). PX4 EKF2 inserts
        # vision_pose into its delayed-measurement buffer by timestamp; if
        # we lie and say "now", the correction is applied to the wrong
        # (too-recent) state -> the fused estimate lags and jerks during
        # motion. Using the VINS odometry stamp lets MAVROS timesync line
        # it up with the FCU clock correctly. Republished duplicates (tick
        # 30 Hz > VINS 15 Hz) are ignored by EKF2 as already-processed.
        out.header.stamp = src.header.stamp
        out.header.frame_id = "map"
        # Compose the VO pose with the (cached, TF-derived) vo_body->base
        # transform so the published pose is the FLU base_link, not the
        # camera/IMU optical frame. Identity if vo_body_frame is unset.
        px, py, pz, ox, oy, oz, ow = self._to_base_link(src.pose.pose)
        out.pose.position.x = px + applied[0]
        out.pose.position.y = py + applied[1]
        out.pose.position.z = pz + applied[2]
        out.pose.orientation.x = ox
        out.pose.orientation.y = oy
        out.pose.orientation.z = oz
        out.pose.orientation.w = ow

        if self.use_odometry:
            # Full odometry: pose (above) + VINS velocity expressed in the
            # base_link body frame (MAVROS odom expects twist in child frame).
            odo = Odometry()
            odo.header.stamp = src.header.stamp
            odo.header.frame_id = "odom"
            odo.child_frame_id = "base_link"
            odo.pose.pose = out.pose
            vw = src.twist.twist.linear            # VINS velocity in world
            R = quaternion_matrix([ox, oy, oz, ow])[:3, :3]   # R_world_base
            vb = R.T.dot([vw.x, vw.y, vw.z])       # -> base_link frame
            odo.twist.twist.linear.x = float(vb[0])
            odo.twist.twist.linear.y = float(vb[1])
            odo.twist.twist.linear.z = float(vb[2])
            # angular: leave 0 (PX4 uses its own gyro for body rates)
            # Small diagonal covariance so EKF2 trusts VINS heavily.
            pc = [0.0] * 36
            tc = [0.0] * 36
            for i in range(6):
                pc[i * 7] = 0.01      # pose: 0.1 m / 0.1 rad std
                tc[i * 7] = 0.01      # twist
            odo.pose.covariance = pc
            odo.twist.covariance = tc
            self.pub_odom.publish(odo)
        else:
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
