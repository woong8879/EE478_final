#!/usr/bin/env python3
"""rviz_goal_relay_node.py

Let you set EGO goals by clicking RViz's "2D Nav Goal" tool. RViz publishes
that to /move_base_simple/goal at z=0; we force z=goal_z (flight height) and
forward it to /next_goal, which the ego_bridge feeds into EGO.
"""
import rospy
from geometry_msgs.msg import PoseStamped


class GoalRelay(object):
    def __init__(self):
        self.goal_z = float(rospy.get_param("~goal_z", 0.7))
        self.out = rospy.Publisher("/next_goal", PoseStamped, queue_size=5)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.cb,
                         queue_size=5)
        rospy.loginfo("[goal_relay] RViz 2D Nav Goal -> /next_goal (z=%.2f)",
                      self.goal_z)

    def cb(self, msg):
        msg.pose.position.z = self.goal_z
        self.out.publish(msg)
        rospy.loginfo("[goal_relay] goal -> (%.2f, %.2f, %.2f)",
                      msg.pose.position.x, msg.pose.position.y, self.goal_z)


if __name__ == "__main__":
    rospy.init_node("rviz_goal_relay")
    GoalRelay()
    rospy.spin()
