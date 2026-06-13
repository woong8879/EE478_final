#!/usr/bin/env python3
"""waypoint_loop_node.py — fly a sequence of waypoints (optionally looping).

A minimal arrival-based goal sequencer for the EGO-Planner path. It feeds
ONE /next_goal at a time to ego_bridge (which forwards it to EGO and emits
/goal_reached when the drone gets within its arrival_radius). On each
/goal_reached we advance to the next waypoint; with ~loop:=true the list
wraps so the drone keeps circling the course.

This is a TEST harness (e.g. "fly one lap of the map"), not a graded
subtask — the mission uses mission_fsm_node instead.

Params:
  ~waypoints   "x1,y1;x2,y2;..."  waypoints in the local map/ENU frame
               (z comes from ~z). Relative to the SVO origin = where the
               stack started (~= takeoff spot), so keep them modest.
  ~z           hover altitude for every waypoint (m)
  ~loop        true -> wrap to waypoint 0 after the last (default true)
  ~frame_id    goal frame (default "map")
  ~start_delay_s   wait after launch before the first goal so takeoff +
                   offboard settle finish (default 14 s; settle_s is 10)
  ~resend_period_s republish the active /next_goal this often as a
                   keep-alive in case ego_bridge missed it (default 3 s)

Topics:
  pub  /next_goal      geometry_msgs/PoseStamped
  sub  /goal_reached   std_msgs/Int32   (from ego_bridge)
"""

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32


def _parse_waypoints(s):
    wps = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in chunk.replace(",", " ").split() if p]
        if len(parts) < 2:
            continue
        wps.append((float(parts[0]), float(parts[1])))
    return wps


class WaypointLoop:
    def __init__(self):
        rospy.init_node("waypoint_loop")
        self.wps = _parse_waypoints(rospy.get_param("~waypoints", "1.5,0; 1.5,1.5; 0,1.5; 0,0"))
        self.z = float(rospy.get_param("~z", 0.5))
        self.loop = bool(rospy.get_param("~loop", True))
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.start_delay = float(rospy.get_param("~start_delay_s", 14.0))
        self.resend_period = float(rospy.get_param("~resend_period_s", 3.0))
        # RELATIVE mode: waypoints are offsets from the drone's position at
        # mission start (i.e. AFTER takeoff + hover settle), not the raw SVO
        # origin. This makes the takeoff VIO spike irrelevant to the course --
        # the stabilized hover position becomes (0,0).
        self.relative = bool(rospy.get_param("~relative", True))
        self.pose_topic = rospy.get_param("~pose_topic",
                                          "/mavros/local_position/pose")
        self.cur = None
        self.cur_z = 0.0
        self.origin = (0.0, 0.0)
        # wait until the drone has actually climbed to ~hover height (takeoff
        # done + settling) before latching the origin / starting the course --
        # robust to slow takeoff + when the operator flips OFFBOARD.
        self.takeoff_done_z = self.z - float(rospy.get_param("~alt_margin", 0.15))
        self.settle_after_alt_s = float(rospy.get_param("~settle_after_alt_s", 4.0))

        if not self.wps:
            rospy.logerr("[wp_loop] no waypoints parsed; aborting")
            raise SystemExit(1)

        self.idx = 0
        self.lap = 0
        self.done = False
        self._last_send = 0.0

        self.pub = rospy.Publisher("/next_goal", PoseStamped, queue_size=2, latch=True)
        rospy.Subscriber("/goal_reached", Int32, self.on_reached, queue_size=5)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.on_pose, queue_size=5)
        rospy.loginfo("[wp_loop] %d waypoints, loop=%s, z=%.2f, start in %.0fs",
                      len(self.wps), self.loop, self.z, self.start_delay)

    def on_pose(self, msg):
        self.cur = (msg.pose.position.x, msg.pose.position.y)
        self.cur_z = msg.pose.position.z

    def _goal_msg(self):
        x, y = self.wps[self.idx]
        m = PoseStamped()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.pose.position.x = self.origin[0] + x
        m.pose.position.y = self.origin[1] + y
        m.pose.position.z = self.z
        m.pose.orientation.w = 1.0
        return m

    def _send_current(self):
        self.pub.publish(self._goal_msg())
        self._last_send = rospy.Time.now().to_sec()

    def on_reached(self, _msg):
        if self.done:
            return
        x, y = self.wps[self.idx]
        rospy.loginfo("[wp_loop] reached wp %d (%.2f,%.2f)", self.idx, x, y)
        self.idx += 1
        if self.idx >= len(self.wps):
            if self.loop:
                self.idx = 0
                self.lap += 1
                rospy.loginfo("[wp_loop] === lap %d complete, looping ===", self.lap)
            else:
                self.done = True
                rospy.loginfo("[wp_loop] === course complete (no loop) ===")
                return
        self._send_current()

    def spin(self):
        # Wait for takeoff + offboard settle before the first goal.
        t0 = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < self.start_delay:
            rospy.sleep(0.2)
        # Then wait until the drone has actually reached hover altitude (slow
        # takeoff) and settled, so the origin latches at the STABLE hover pose.
        rospy.loginfo("[wp_loop] waiting for climb to ~%.2f m...", self.takeoff_done_z)
        ta = rospy.Time.now()
        while not rospy.is_shutdown() and self.cur_z < self.takeoff_done_z \
                and (rospy.Time.now() - ta).to_sec() < 90.0:
            rospy.sleep(0.2)
        if self.cur_z < self.takeoff_done_z:
            rospy.logwarn("[wp_loop] altitude %.2f not reached in 90s; starting anyway",
                          self.takeoff_done_z)
        rospy.sleep(self.settle_after_alt_s)
        # Latch the (now stabilized) hover position as the course origin so the
        # takeoff VIO spike doesn't shift the waypoints.
        if self.relative:
            t1 = rospy.Time.now()
            while not rospy.is_shutdown() and self.cur is None \
                    and (rospy.Time.now() - t1).to_sec() < 5.0:
                rospy.sleep(0.1)
            if self.cur is not None:
                self.origin = self.cur
                rospy.loginfo("[wp_loop] origin latched at hover pos (%.2f, %.2f)",
                              self.origin[0], self.origin[1])
            else:
                rospy.logwarn("[wp_loop] no pose on %s; origin=(0,0)", self.pose_topic)
        rospy.loginfo("[wp_loop] starting course at wp 0")
        self._send_current()
        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown():
            if not self.done:
                now = rospy.Time.now().to_sec()
                if now - self._last_send > self.resend_period:
                    self._send_current()   # keep-alive resend of active goal
            rate.sleep()


if __name__ == "__main__":
    try:
        WaypointLoop().spin()
    except rospy.ROSInterruptException:
        pass
