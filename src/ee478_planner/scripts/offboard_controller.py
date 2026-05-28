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
        with self.lock:
            if not self.use_raw:
                self.setpoint.header.stamp = rospy.Time.now()
                self.pub_sp.publish(self.setpoint)


if __name__ == "__main__":
    try:
        OffboardController().spin()
    except rospy.ROSInterruptException:
        pass
