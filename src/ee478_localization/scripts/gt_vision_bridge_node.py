#!/usr/bin/env python3
"""ee478_localization/gt_vision_bridge_node.py

SIM ONLY: republish the Gazebo ground-truth pose of the drone as
/mavros/vision_pose/pose at 30 Hz. PX4 EKF2 fuses it as the
external-vision source. Behaves like a perfect VIO: zero drift,
zero noise.

This is used to validate the planner / agent / perception stack
in simulation without fighting PX4 SITL's mono-VIO initialisation
quirks. The REAL drone build replaces this node with VINS-Fusion
(or RTAB-Map RGBD); every consumer downstream is unchanged because
the contract is the same `/mavros/vision_pose/pose` topic.

Subscribes:
  /gazebo/model_states  (gazebo_msgs/ModelStates)

Publishes:
  /mavros/vision_pose/pose  (geometry_msgs/PoseStamped) @ ~rate_hz
"""

import threading

import rospy
from geometry_msgs.msg import PoseStamped


class GtVisionBridgeNode:
    def __init__(self):
        rospy.init_node("gt_vision_bridge")
        self.lock = threading.Lock()

        self.target_model = rospy.get_param("~target_model",
                                            "iris_depth_camera")
        self.vision_topic = rospy.get_param("~vision_pose_topic",
                                            "/mavros/vision_pose/pose")
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))

        self.last_pose = None
        self.pub = rospy.Publisher(self.vision_topic, PoseStamped,
                                   queue_size=10)

        try:
            from gazebo_msgs.msg import ModelStates
        except ImportError:
            rospy.logerr("[gt_bridge] gazebo_msgs not available")
            return
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self.on_states, queue_size=2)

        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.tick)
        rospy.loginfo(
            f"[gt_bridge] {self.target_model} -> {self.vision_topic} "
            f"@ {self.rate_hz} Hz (SIM ONLY)")

    def on_states(self, msg):
        try:
            idx = msg.name.index(self.target_model)
        except ValueError:
            return
        with self.lock:
            self.last_pose = msg.pose[idx]

    def tick(self, _evt):
        with self.lock:
            pose = self.last_pose
        if pose is None:
            return
        out = PoseStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = "map"
        out.pose = pose
        self.pub.publish(out)


if __name__ == "__main__":
    try:
        GtVisionBridgeNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
