#!/usr/bin/env python3
"""Accumulate pose topics into nav_msgs/Path for RViz trajectory display.

Handles both geometry_msgs/PoseStamped (e.g. /mavros/*_position/pose,
/mavros/vision_pose/pose) and geometry_msgs/PoseWithCovarianceStamped
(e.g. SVO's /svo/pose_imu) — the message type is auto-detected per source
topic, so the same node draws the EKF trail, the relayed vision trail, and
the RAW SVO odometry trail. Each output Path keeps the source frame_id so
world-frame (SVO) and map-frame (EKF) trails render in their true frames.
"""
import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path


class PoseToPath:
    def __init__(self, in_topic, out_topic, max_points=2000):
        self.path = Path()
        self.max_points = max_points
        self.pub = rospy.Publisher(out_topic, Path, queue_size=5)
        msg_type = self._resolve_type(in_topic)
        rospy.Subscriber(in_topic, msg_type, self.cb, queue_size=5)
        rospy.loginfo("pose_to_path: %s (%s) -> %s",
                      in_topic, msg_type.__name__, out_topic)

    @staticmethod
    def _resolve_type(topic):
        """Pick PoseStamped vs PoseWithCovarianceStamped from the live topic.
        Falls back to PoseStamped if the topic is not advertised yet."""
        try:
            import rostopic
            ttype, _, _ = rostopic.get_topic_type(topic, blocking=False)
        except Exception:
            ttype = None
        if ttype == "geometry_msgs/PoseWithCovarianceStamped":
            return PoseWithCovarianceStamped
        return PoseStamped

    def cb(self, msg):
        ps = PoseStamped()
        ps.header = msg.header
        # PoseWithCovarianceStamped nests pose.pose; PoseStamped is flat.
        ps.pose = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose
        # Adopt the source frame so the trail renders in its true frame
        # (SVO -> "world", EKF -> "map"; world<->map is identity here).
        self.path.header.frame_id = msg.header.frame_id or "map"
        self.path.header.stamp = msg.header.stamp
        self.path.poses.append(ps)
        if len(self.path.poses) > self.max_points:
            self.path.poses = self.path.poses[-self.max_points:]
        self.pub.publish(self.path)


if __name__ == "__main__":
    rospy.init_node("pose_to_path")
    topics = rospy.get_param("~topics",
        "/mavros/local_position/pose:/ekf_path,"
        "/mavros/vision_pose/pose:/vio_path,"
        "/svo/pose_imu:/svo_path")
    for pair in topics.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        src, dst = pair.rsplit(":", 1)
        PoseToPath(src.strip(), dst.strip())
    rospy.spin()
