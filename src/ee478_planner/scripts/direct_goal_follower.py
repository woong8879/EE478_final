#!/usr/bin/env python3
"""ee478_planner/direct_goal_follower.py

Trivial /next_goal -> /offboard/goal relay (no obstacle avoidance).

This is the "direct" planner path used by sub-tasks that do not need
EGO-Planner (s6 signature spin, s8 land) and by flight_test.launch's
planner:=direct mode. It simply forwards whatever pose is published on
/next_goal to the offboard_controller's goal input, continuously, so
that PX4 keeps a fresh position setpoint.

  Topics (subscribed):
    /next_goal                       (geometry_msgs/PoseStamped)
    /mavros/local_position/pose      (geometry_msgs/PoseStamped)

  Topics (published):
    /offboard/goal                   (geometry_msgs/PoseStamped)
    /goal_reached                    (std_msgs/Int32) -- monotonically
        increasing counter, bumped once each time the drone first comes
        within ~arrival_radius_m of the active goal.

Why republish continuously instead of forwarding once:
  offboard_controller drops goals that arrive before it is armed +
  settled (settle_s). A single forward during that window would be
  lost. Streaming the latest goal at rate_hz guarantees the goal is
  honoured as soon as the controller is ready.
"""

import threading

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32


class DirectGoalFollower:
    def __init__(self):
        rospy.init_node("direct_goal_follower")
        self.lock = threading.Lock()

        # ---- params ----
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.arrival_radius_m = float(
            rospy.get_param("~arrival_radius_m", 0.5))
        self.goal_in_topic = rospy.get_param("~goal_in", "/next_goal")
        self.goal_out_topic = rospy.get_param("~goal_out", "/offboard/goal")

        # ---- state ----
        self.goal = None          # latest PoseStamped on /next_goal
        self.current_pose = None  # latest /mavros/local_position/pose
        self.reached_count = 0
        self.reached_latched = False  # already counted arrival for this goal

        # ---- pubs / subs ----
        self.pub_goal = rospy.Publisher(self.goal_out_topic, PoseStamped,
                                        queue_size=5)
        self.pub_reached = rospy.Publisher("/goal_reached", Int32,
                                           queue_size=5)
        rospy.Subscriber(self.goal_in_topic, PoseStamped, self.on_goal,
                         queue_size=5)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)

        rospy.loginfo("[direct] ready; %s -> %s, arrival_radius=%.2f m",
                      self.goal_in_topic, self.goal_out_topic,
                      self.arrival_radius_m)

    # ---------------- callbacks ----------------
    def on_goal(self, msg):
        with self.lock:
            # New goal target resets arrival latch.
            self.goal = msg
            self.reached_latched = False
        rospy.loginfo_throttle(
            1.0, "[direct] new goal: (%.2f,%.2f,%.2f)",
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def on_pose(self, msg):
        self.current_pose = msg

    # ---------------- main loop ----------------
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self.lock:
                goal = self.goal
            if goal is not None:
                goal.header.stamp = rospy.Time.now()
                self.pub_goal.publish(goal)
                self._check_arrival(goal)
            rate.sleep()

    def _check_arrival(self, goal):
        cp = self.current_pose
        if cp is None or self.reached_latched:
            return
        dx = goal.pose.position.x - cp.pose.position.x
        dy = goal.pose.position.y - cp.pose.position.y
        dz = goal.pose.position.z - cp.pose.position.z
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist <= self.arrival_radius_m:
            self.reached_latched = True
            self.reached_count += 1
            self.pub_reached.publish(Int32(data=self.reached_count))
            rospy.loginfo("[direct] goal reached (#%d, dist=%.2f m)",
                          self.reached_count, dist)


if __name__ == "__main__":
    try:
        DirectGoalFollower().spin()
    except rospy.ROSInterruptException:
        pass
