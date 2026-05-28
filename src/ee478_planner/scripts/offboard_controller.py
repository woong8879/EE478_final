#!/usr/bin/env python3
"""ee478_drone_control/offboard_controller.py

Generic offboard position controller for PX4 via MAVROS.

Provides a high-level Python API and a ROS interface:

  Topics (subscribed):
    ~goal       (geometry_msgs/PoseStamped)  -- desired pose in local ENU frame
    ~cmd_raw    (mavros_msgs/PositionTarget) -- full trajectory feedforward

  Topics (published):
    /mavros/setpoint_position/local  (geometry_msgs/PoseStamped)
    /mavros/setpoint_raw/local       (mavros_msgs/PositionTarget)
  Services consumed:
    /mavros/cmd/arming
    /mavros/set_mode

Behavior:
  * Streams setpoints at 20 Hz (PX4 requires >= 2 Hz before OFFBOARD will be acc
epted).
  * On startup: takes off to ~takeoff_alt (default 1.5 m) above current xy, sets
 OFFBOARD, arms.
  * Whenever a new ~goal arrives, becomes the new target setpoint.
  * Pure position controller -- PX4's onboard cascade handles the rest.

Run with QGroundControl open for manual override / arming if PX4 refuses auto-ar
m.
"""

import math as _math
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode


class OffboardController:
    def __init__(self):
        rospy.init_node("offboard_controller")
        self.lock = threading.Lock()

        # ---- params ----
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.takeoff_alt = float(rospy.get_param("~takeoff_alt", 1.5))
        self.auto_arm = bool(rospy.get_param("~auto_arm", True))
        self.frame_id = rospy.get_param("~frame_id", "map")
        # Cap for the velocity feedforward injected on top of position
        # setpoints. Keeps long position steps from sending PX4 into
        # huge climbs/overshoots.
        self.max_step_vel = float(rospy.get_param("~max_step_vel", 1.0))
        # Vertical velocity feedforward cap (m/s). Separately specified
        # because the SITL altitude loop diverges if we let xy and z
        # share a single horizon-aware velocity cap (z gets squeezed to
        # near-zero on long horizontal steps and the drone drifts up).
        self.max_step_vel_z = float(
            rospy.get_param("~max_step_vel_z", 0.5))
        # Hold takeoff setpoint for this long after arming so PX4 EKF z
        # converges before any horizontal navigation goals are honoured.
        self.settle_s = float(rospy.get_param("~settle_s", 120.0))
        self.armed_at = None

        # ---- state ----
        self.state = State()
        self.current_pose = None
        self.setpoint = PoseStamped()
        self.setpoint.header.frame_id = self.frame_id
        self.setpoint.pose.position.x = 0.0
        self.setpoint.pose.position.y = 0.0
        self.setpoint.pose.position.z = self.takeoff_alt
        self.setpoint.pose.orientation.w = 1.0
        self.has_origin = False
        self.use_raw = False
        self.raw_setpoint = PositionTarget()

        # ---- pubs / subs / services ----
        self.pub_sp = rospy.Publisher("/mavros/setpoint_position/local",
                                      PoseStamped, queue_size=10)
        self.pub_sp_raw = rospy.Publisher("/mavros/setpoint_raw/local",
                                          PositionTarget, queue_size=10)
        rospy.Subscriber("/mavros/state", State, self.on_state, queue_size=5)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)
        rospy.Subscriber("~goal", PoseStamped, self.on_goal, queue_size=5)
        rospy.Subscriber("~cmd_raw", PositionTarget, self.on_cmd_raw, queue_size=5)

        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        self.srv_arm = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.srv_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        rospy.loginfo("[offboard] ready; auto_arm=%s takeoff_alt=%.2f",
                      self.auto_arm, self.takeoff_alt)

    # ---------------- callbacks ----------------
    def on_state(self, msg):
        self.state = msg

    def on_pose(self, msg):
        self.current_pose = msg
        if not self.has_origin:
            # latch initial XY so takeoff is straight up
            with self.lock:
                self.setpoint.pose.position.x = msg.pose.position.x
                self.setpoint.pose.position.y = msg.pose.position.y
                self.setpoint.pose.position.z = msg.pose.position.z + self.takeoff_alt
                self.setpoint.pose.orientation = msg.pose.orientation
            self.has_origin = True
            rospy.loginfo("[offboard] origin latched at (%.2f,%.2f,%.2f); "
                          "takeoff target z=%.2f",
                          msg.pose.position.x, msg.pose.position.y,
                          msg.pose.position.z, self.setpoint.pose.position.z)

    def on_goal(self, msg):
        if self.armed_at is None:
            rospy.loginfo_throttle(2.0, "[offboard] dropping goal — not armed yet")
            return
        held = (rospy.Time.now() - self.armed_at).to_sec()
        if held < self.settle_s:
            rospy.loginfo_throttle(
                2.0, "[offboard] holding takeoff (%.0f/%.0f s) — goal "
                "(%.2f,%.2f,%.2f) deferred",
                held, self.settle_s,
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
            return
        with self.lock:
            self.use_raw = False
            self.setpoint.header.frame_id = msg.header.frame_id or self.frame_id
            self.setpoint.pose = msg.pose
        rospy.loginfo_throttle(1.0, "[offboard] new goal: (%.2f,%.2f,%.2f)",
                               msg.pose.position.x, msg.pose.position.y,
                               msg.pose.position.z)

    def on_cmd_raw(self, msg):
        if self.armed_at is None:
            return
        held = (rospy.Time.now() - self.armed_at).to_sec()
        if held < self.settle_s:
            return
        with self.lock:
            self.use_raw = True
            self.raw_setpoint = msg
            self.raw_setpoint.header.frame_id = msg.header.frame_id or self.frame_id
            self.raw_setpoint.header.stamp = rospy.Time.now()
            self.pub_sp_raw.publish(self.raw_setpoint)
        rospy.loginfo_throttle(1.0, "[offboard] new raw cmd: (%.2f,%.2f,%.2f)",
                               msg.position.x, msg.position.y,
                               msg.position.z)

    # ---------------- main loop ----------------
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        # warm up: stream setpoints before OFFBOARD switch
        rospy.loginfo("[offboard] streaming initial setpoints...")
        for _ in range(int(self.rate_hz * 2.0)):
            if rospy.is_shutdown():
                return
            self._publish()
            rate.sleep()

        last_req = rospy.Time(0)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if self.auto_arm and (now - last_req) > rospy.Duration(2.0):
                if self.state.mode != "OFFBOARD":
                    try:
                        res = self.srv_mode(custom_mode="OFFBOARD")
                        if res.mode_sent:
                            rospy.loginfo("[offboard] requested OFFBOARD")
                    except Exception as e:
                        rospy.logwarn(f"set_mode: {e}")
                elif not self.state.armed:
                    try:
                        res = self.srv_arm(True)
                        if res.success:
                            rospy.loginfo("[offboard] vehicle armed")
                            self.armed_at = rospy.Time.now()
                    except Exception as e:
                        rospy.logwarn(f"arming: {e}")
                elif self.armed_at is None and self.state.armed:
                    # External arm (e.g., already armed when we connected)
                    self.armed_at = rospy.Time.now()
                last_req = now
            self._publish()
            rate.sleep()

    def _publish(self):
        """Stream the active setpoint to PX4.

        When the active goal is a plain PoseStamped (`use_raw=False`),
        publish a PositionTarget on /mavros/setpoint_raw/local with a
        VELOCITY FEEDFORWARD pointing from the current pose toward the
        goal, capped at `max_step_vel`. This is what PX4 SITL actually
        needs to track a multi-metre position step — without the
        velocity hint the position controller is so sluggish that long
        steps either stall the drone or send it climbing to weird
        altitudes. We keep publishing /mavros/setpoint_position/local
        as well so any other consumer that still wants the raw goal
        sees it.
        """
        with self.lock:
            if self.use_raw:
                return
            sp = self.setpoint
            sp.header.stamp = rospy.Time.now()
            self.pub_sp.publish(sp)

            # Build a setpoint_raw with position + capped velocity.
            cp = self.current_pose
            if cp is None:
                return
            tgt = PositionTarget()
            tgt.header.stamp = rospy.Time.now()
            tgt.header.frame_id = self.frame_id
            tgt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
            # type_mask: use POSITION + VELOCITY (feedforward), set yaw,
            # ignore acceleration and yaw_rate.
            tgt.type_mask = (PositionTarget.IGNORE_AFX
                             | PositionTarget.IGNORE_AFY
                             | PositionTarget.IGNORE_AFZ
                             | PositionTarget.IGNORE_YAW_RATE)
            tgt.position.x = sp.pose.position.x
            tgt.position.y = sp.pose.position.y
            tgt.position.z = sp.pose.position.z

            # Decouple horizontal vs vertical velocity feedforward.
            # XY: cap by max_step_vel along the xy direction so a long
            # horizontal step doesn't ask PX4 for 10 m/s.
            # Z: ALWAYS commanded toward the target altitude at up to
            # max_step_vel_z. Without this, PX4 SITL's altitude loop
            # drifts (it tries to maintain z by attitude, and the
            # horizontal pitching pushes the drone up). With a steady
            # vz hint, PX4 actively converges to z_target.
            dx = sp.pose.position.x - cp.pose.position.x
            dy = sp.pose.position.y - cp.pose.position.y
            dz = sp.pose.position.z - cp.pose.position.z
            dxy = (dx * dx + dy * dy) ** 0.5
            v_max = self.max_step_vel
            if dxy > 1e-3:
                k_xy = min(1.0, v_max / dxy)
                tgt.velocity.x = dx * k_xy
                tgt.velocity.y = dy * k_xy
            v_max_z = self.max_step_vel_z
            if abs(dz) > 1e-3:
                tgt.velocity.z = max(-v_max_z,
                                     min(v_max_z, dz * 2.0))
            q = sp.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            tgt.yaw = _math.atan2(siny, cosy)
            self.pub_sp_raw.publish(tgt)


if __name__ == "__main__":
    try:
        OffboardController().spin()
    except rospy.ROSInterruptException:
        pass
