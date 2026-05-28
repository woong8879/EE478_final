#!/usr/bin/env python3
"""ee478_localization/sim_landmark_anchor_node.py

SIM-ONLY shim that publishes a /landmark_anchor_pose anchor at a
moderate rate (~1 Hz) using the gazebo ground-truth drone pose.

Why this exists: VINS-Fusion mono+IMU in PX4 SITL diverges in
position by metres per minute because the simulated IMU is too clean
to anchor the metric scale. On a real drone the perception system
publishes /landmark_anchor_pose every time it sees a known store
signboard, and vio_bridge's anchor_blend pulls the published
vision_pose back toward truth. We replicate that behaviour in sim by
deriving the anchor directly from gazebo /model_states.

vio_bridge already implements anchor_blend (smoothly slides
applied_offset toward the target at anchor_blend_rate m/s), so the
output stream never jumps when an anchor lands.

For real-drone deployment this node is replaced by
landmark_anchor_publisher_node.py which reads SignboardArray +
SemanticMap. The interface (/landmark_anchor_pose) is identical, so
no other component changes.
"""

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped


class SimLandmarkAnchor:
    def __init__(self):
        rospy.init_node("sim_landmark_anchor")

        self.model = rospy.get_param(
            "~gt_model", "iris_depth_camera_vio")
        self.rate_hz = float(rospy.get_param("~rate_hz", 1.0))
        self.anchor_topic = rospy.get_param(
            "~anchor_topic", "/landmark_anchor_pose")

        self.latest = None

        self.pub = rospy.Publisher(self.anchor_topic, PoseStamped,
                                   queue_size=2)
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self.on_states, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.tick)

        rospy.loginfo(
            f"[sim_anchor] gt({self.model}) -> {self.anchor_topic} "
            f"@ {self.rate_hz:.1f} Hz")

    def on_states(self, msg):
        try:
            i = msg.name.index(self.model)
        except ValueError:
            return
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "map"
        ps.pose = msg.pose[i]
        self.latest = ps

    def tick(self, _evt):
        if self.latest is not None:
            self.pub.publish(self.latest)


if __name__ == "__main__":
    try:
        SimLandmarkAnchor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
