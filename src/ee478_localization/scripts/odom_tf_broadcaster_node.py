#!/usr/bin/env python3
"""ee478_localization/odom_tf_broadcaster_node.py

Bridge /mavros/local_position/pose -> tf2 (map -> base_link).

MAVROS's local_position plugin does not broadcast TF by default in our
sim config, so the TF tree is disconnected: the static base_link ->
camera_link transform exists but the dynamic root (map -> base_link)
does not. EGO's world_cloud_publisher cannot then resolve
map <- camera_link and silently drops every cloud.
"""

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped


class OdomTfBroadcaster:
    def __init__(self):
        rospy.init_node("odom_tf_broadcaster")
        self.parent = rospy.get_param("~parent_frame", "map")
        self.child = rospy.get_param("~child_frame", "base_link")
        self.in_topic = rospy.get_param(
            "~in_topic", "/mavros/local_position/pose")

        self.br = tf2_ros.TransformBroadcaster()
        rospy.Subscriber(self.in_topic, PoseStamped, self.on_pose,
                         queue_size=10)
        rospy.loginfo(
            f"[odom_tf] {self.in_topic} -> tf({self.parent} -> "
            f"{self.child})")

    def on_pose(self, msg):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.parent
        t.child_frame_id = self.child
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.br.sendTransform(t)


if __name__ == "__main__":
    try:
        OdomTfBroadcaster()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
