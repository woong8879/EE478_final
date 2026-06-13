#!/usr/bin/env python3
"""svo_to_mavros_node.py

FC-less EGO testing: the flight controller (and MAVROS) is gone, so EGO has no
/mavros/local_position/pose|odom. SVO still produces a VIO body pose from the
RealSense IR (no FC needed), so we just republish it under the MAVROS names EGO
already expects. With this, ego_planner.launch runs UNCHANGED:
  /svo/pose_imu  ->  /mavros/local_position/pose  (PoseStamped, for camera_pose)
                 ->  /mavros/local_position/odom  (Odometry,    for EGO odom)
Both in the SVO world frame relabelled "map", child "base_link".
"""
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry


class SvoToMavros(object):
    def __init__(self):
        self.in_topic = rospy.get_param("~in_topic", "/svo/pose_imu")
        self.world = rospy.get_param("~world_frame", "map")
        self.child = rospy.get_param("~body_frame", "base_link")
        self.pub_pose = rospy.Publisher("/mavros/local_position/pose",
                                        PoseStamped, queue_size=10)
        self.pub_odom = rospy.Publisher("/mavros/local_position/odom",
                                        Odometry, queue_size=10)
        rospy.Subscriber(self.in_topic, PoseWithCovarianceStamped, self.cb,
                         queue_size=20)
        rospy.loginfo("[svo_to_mavros] %s -> /mavros/local_position/{pose,odom}",
                      self.in_topic)

    def cb(self, msg):
        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = self.world
        ps.pose = msg.pose.pose
        self.pub_pose.publish(ps)

        od = Odometry()
        od.header.stamp = msg.header.stamp
        od.header.frame_id = self.world
        od.child_frame_id = self.child
        od.pose.pose = msg.pose.pose
        self.pub_odom.publish(od)


if __name__ == "__main__":
    rospy.init_node("svo_to_mavros")
    SvoToMavros()
    rospy.spin()
