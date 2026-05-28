#!/usr/bin/env python3
"""ee478_localization/camera_pose_publisher_node.py

Publish the depth camera's WORLD pose as PoseStamped at the rate the
drone's body pose arrives. Combined with the gazebo depth-image
topic, this lets EGO-Planner's grid_map use pose_type=1 (depth +
pose) instead of pose_type=2 (cloud + odom).

Why this matters: with pose_type=2 we need to transform the 400k-point
gazebo depth cloud from camera_link to map BEFORE handing it to EGO,
and the only tool we have in Python (tf2_sensor_msgs.do_transform_cloud)
is so slow (>500 ms per cloud) that we drop ~75% of frames. The
resulting stale grid_map + 0.4 m drone displacement between cloud
samples puts phantom obstacle cells right where the drone currently
is, and EGO fails with "drone is in obstacle".

With pose_type=1 EGO does the projection itself in C++ at full
camera rate, using:
  camera_pose (this node)  +  depth_image (gazebo)  +  intrinsics
"""

import threading

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped


class CameraPosePublisher:
    def __init__(self):
        rospy.init_node("camera_pose_publisher")

        self.body_pose_topic = rospy.get_param(
            "~body_pose_topic", "/mavros/local_position/pose")
        self.out_topic = rospy.get_param(
            "~out_topic", "/drone_0_pcl_render_node/camera_pose")
        self.world_frame  = rospy.get_param("~world_frame",  "map")
        self.body_frame   = rospy.get_param("~body_frame",   "base_link")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_link")

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Static body->camera TF can be cached on first lookup.
        self.body_to_cam = None
        self.lock = threading.Lock()

        self.pub = rospy.Publisher(self.out_topic, PoseStamped,
                                   queue_size=10)
        rospy.Subscriber(self.body_pose_topic, PoseStamped,
                         self.on_body_pose, queue_size=10)

        rospy.loginfo(
            f"[cam_pose] {self.body_pose_topic} + tf({self.body_frame} "
            f"-> {self.camera_frame}) -> {self.out_topic}")

    def _ensure_body_to_cam(self):
        with self.lock:
            if self.body_to_cam is not None:
                return self.body_to_cam
        try:
            tf = self.tf_buffer.lookup_transform(
                self.body_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(0.5))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        with self.lock:
            self.body_to_cam = tf
        return tf

    def on_body_pose(self, msg):
        bc = self._ensure_body_to_cam()
        if bc is None:
            return
        # Compose world<-body (from msg) with body<-camera (from tf)
        # to get world<-camera. We use tf2_geometry_msgs's
        # do_transform_pose by building a PoseStamped in the camera
        # frame at the origin and transforming it through both.
        from tf2_geometry_msgs import do_transform_pose
        import tf_conversions
        import numpy as np

        # Build the camera origin pose in body frame from the static
        # body->camera TF.
        cam_in_body = PoseStamped()
        cam_in_body.header.stamp = msg.header.stamp
        cam_in_body.header.frame_id = self.body_frame
        cam_in_body.pose.position.x = bc.transform.translation.x
        cam_in_body.pose.position.y = bc.transform.translation.y
        cam_in_body.pose.position.z = bc.transform.translation.z
        cam_in_body.pose.orientation = bc.transform.rotation

        # Compose world<-body * body<-camera explicitly.
        from geometry_msgs.msg import TransformStamped
        world_from_body = TransformStamped()
        world_from_body.header.stamp = msg.header.stamp
        world_from_body.header.frame_id = self.world_frame
        world_from_body.child_frame_id = self.body_frame
        world_from_body.transform.translation.x = msg.pose.position.x
        world_from_body.transform.translation.y = msg.pose.position.y
        world_from_body.transform.translation.z = msg.pose.position.z
        world_from_body.transform.rotation = msg.pose.orientation

        cam_in_world = do_transform_pose(cam_in_body, world_from_body)
        cam_in_world.header.stamp = msg.header.stamp
        cam_in_world.header.frame_id = self.world_frame
        self.pub.publish(cam_in_world)


if __name__ == "__main__":
    try:
        CameraPosePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
