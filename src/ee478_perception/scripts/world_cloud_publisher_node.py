#!/usr/bin/env python3
"""ee478_perception/world_cloud_publisher_node.py

Republish the gazebo depth camera point cloud in MAP frame.

EGO-Planner's grid_map module expects the input cloud to already be in
the world frame; it doesn't do any TF lookup of its own. The raw
gazebo cloud comes out in the camera optical frame ("camera_link"),
so without this transformer EGO's occupancy grid stays empty (every
point lands outside the bounded map volume after being interpreted
verbatim as world coordinates).

This node uses tf2 to transform the incoming cloud to ~target_frame
(default "map") at the cloud's own header.stamp, then republishes it
on ~out_topic. The base_link -> camera_link static TF plus the
odom -> base_link tree from MAVROS provides the chain.
"""

import struct
import threading

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 -- registers PointStamped
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from geometry_msgs.msg import TransformStamped


class WorldCloudPublisher:
    def __init__(self):
        rospy.init_node("world_cloud_publisher")

        self.in_topic = rospy.get_param(
            "~in_topic", "/iris_depth_camera_vio/camera/depth/points")
        self.out_topic = rospy.get_param(
            "~out_topic", "/drone_0_iris_depth_camera/camera/depth/points")
        self.target_frame = rospy.get_param("~target_frame", "map")
        self.lookup_timeout = rospy.Duration(
            float(rospy.get_param("~lookup_timeout_s", 0.05)))
        # Downsample the cloud BEFORE the tf transform so the Python
        # do_transform_cloud doesn't drown in 400k point clouds
        # (848x480 depth). We keep every Nth point — voxel grid
        # would be cleaner but adds a pcl_ros dep.
        self.stride = int(rospy.get_param("~stride", 4))
        # Drop transformed points outside [z_min, z_max] in world frame.
        # z_min=0.1 keeps the LOWER halves of poles (model bases at
        # ground z=0) while still dropping pure-ground artefacts.
        self.z_min = float(rospy.get_param("~z_min", 0.10))
        self.z_max = float(rospy.get_param("~z_max", 3.0))
        # Drop points within `self_radius` of the drone xy to avoid the
        # drone's own body / prop wash artefacts being ingested.
        self.self_radius = float(rospy.get_param("~self_radius_m", 1.5))
        self.drone_xy = None  # (x, y) updated by separate pose sub

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher(self.out_topic, PointCloud2,
                                   queue_size=2)
        rospy.Subscriber(self.in_topic, PointCloud2, self.on_cloud,
                         queue_size=2)
        from geometry_msgs.msg import PoseStamped as _PS
        rospy.Subscriber("/mavros/local_position/pose", _PS,
                         self._on_drone_pose, queue_size=5)

        self.warn_count = 0
        rospy.loginfo(
            f"[world_cloud] {self.in_topic} ({self.target_frame}) "
            f"-> {self.out_topic}")

    def _on_drone_pose(self, msg):
        self.drone_xy = (msg.pose.position.x, msg.pose.position.y)

    def on_cloud(self, msg):
        try:
            # Use the LATEST available transform rather than the cloud's
            # exact stamp. The mavros pose stream and the depth cloud
            # publish on different threads and the TF broadcaster lags
            # the cloud slightly; with exact-stamp lookups we drop ~88
            # of every 100 frames as "future stamp not yet seen".
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id,
                rospy.Time(0), self.lookup_timeout)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            if self.warn_count < 5:
                rospy.logwarn_throttle(
                    1.0,
                    f"[world_cloud] tf {self.target_frame} <- "
                    f"{msg.header.frame_id}: {e}")
                self.warn_count += 1
            return

        # 1) Downsample BEFORE transforming. The raw depth cloud at
        # 848x480 is ~400k points; transforming all of them in Python
        # takes ~700 ms per frame and we drop 87% of input. By
        # striding to every Nth point we keep ~50k points which
        # transforms in <100 ms, giving us 8-10 Hz output.
        kept = []
        for pt in pc2.read_points(msg, skip_nans=True,
                                   field_names=("x", "y", "z")):
            kept.append(pt)
        kept = kept[::self.stride]
        if not kept:
            return
        small = pc2.create_cloud_xyz32(msg.header, kept)
        out = do_transform_cloud(small, tf)

        # 2) Post-filter: drop points too low (ground), too high
        # (ceiling), or too close to the drone (self-detection).
        pts = np.array(list(pc2.read_points(
            out, skip_nans=True, field_names=("x", "y", "z"))),
            dtype=np.float32)
        if pts.size == 0:
            return
        mask = (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        if self.drone_xy is not None:
            dx = pts[:, 0] - self.drone_xy[0]
            dy = pts[:, 1] - self.drone_xy[1]
            mask &= (dx * dx + dy * dy) > (self.self_radius ** 2)
        pts = pts[mask]
        if pts.size == 0:
            return

        filtered = pc2.create_cloud_xyz32(out.header,
                                          pts.tolist())
        filtered.header.stamp = msg.header.stamp
        filtered.header.frame_id = self.target_frame
        self.pub.publish(filtered)


if __name__ == "__main__":
    try:
        WorldCloudPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
