#!/usr/bin/env python3
"""ee478_localization/landmark_anchor_publisher_node.py

Sparse landmark drift correction publisher.

Pipeline
--------
  /signboards               (ee478_msgs/SignboardArray)
                              from perception (YOLO + RGBD)
  /semantic_map             (ee478_msgs/SemanticMap)
                              from semantic_map_manager (known stores)
  /mavros/local_position/pose
                              current EKF-fused drone pose
            |
            v
  /landmark_anchor_pose     (geometry_msgs/PoseStamped)
                              "drone IS at this world pose"
                              consumed by vio_bridge_node

How it works
------------
A signboard observation gives us position_base, the 3D position of
the signboard centre expressed in the drone's body frame. If the
drone's current pose has drift `e`, then the OBSERVED world position
of the signboard is

    sb_world_obs = drone_pose_world + R(yaw) @ position_base

The CLOSEST known store in /semantic_map gives us the true world
position `sb_world_known`. So the drone's TRUE world pose is

    drone_world_true = sb_world_known - R(yaw) @ position_base

We publish that as /landmark_anchor_pose. vio_bridge_node computes
the offset against its current raw VO pose and SLOWLY (0.2 m/s)
blends it into /mavros/vision_pose/pose. No teleporting.

Assumptions
-----------
- Drone is mostly level (small roll/pitch); we use yaw only for the
  body->world rotation. For the EE478 sim/real-flight envelope (z
  ~0.7 m hover, gentle motion) this is fine.
- Store association is purely SPATIAL (nearest known store within
  ~assoc_radius_m). icon_labels / bundle_id refinement can be added
  later if multiple stores get within range.
"""

import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped

from ee478_msgs.msg import SemanticMap, SignboardArray


def _yaw_from_quat(q):
    """ZYX yaw extraction, robust for the small roll/pitch we expect."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _isfinite_point(p):
    return (math.isfinite(p.x) and math.isfinite(p.y)
            and math.isfinite(p.z))


class LandmarkAnchorPublisher:
    def __init__(self):
        rospy.init_node("landmark_anchor_publisher")
        self.lock = threading.Lock()

        self.signboard_topic = rospy.get_param(
            "~signboard_topic", "/signboards")
        self.semantic_map_topic = rospy.get_param(
            "~semantic_map_topic", "/semantic_map")
        self.drone_pose_topic = rospy.get_param(
            "~drone_pose_topic", "/mavros/local_position/pose")
        self.anchor_topic = rospy.get_param(
            "~anchor_topic", "/landmark_anchor_pose")
        self.world_frame = rospy.get_param("~world_frame", "map")

        # Association radius: a signboard observation is bound to the
        # nearest known store only if the OBSERVED world position is
        # within this radius. Tighter than vio_bridge's 5 m gate so we
        # never emit a misassociation that the bridge then has to
        # reject.
        self.assoc_radius_m = float(
            rospy.get_param("~assoc_radius_m", 3.0))
        # Don't emit more than one anchor per `min_period_s`. A noisy
        # YOLO may produce many obs/sec on the same sign; we don't
        # need to push them all through the EKF blend.
        self.min_period_s = float(rospy.get_param("~min_period_s", 0.5))
        # Reject signboards whose body-frame distance is implausible.
        self.min_range_m = float(rospy.get_param("~min_range_m", 0.5))
        self.max_range_m = float(rospy.get_param("~max_range_m", 8.0))

        self.sem_map = None
        self.drone_pose = None
        self.last_publish_t = 0.0
        self.publish_count = 0

        self.pub = rospy.Publisher(
            self.anchor_topic, PoseStamped, queue_size=5)

        rospy.Subscriber(self.semantic_map_topic, SemanticMap,
                         self.on_map, queue_size=1)
        rospy.Subscriber(self.drone_pose_topic, PoseStamped,
                         self.on_pose, queue_size=5)
        rospy.Subscriber(self.signboard_topic, SignboardArray,
                         self.on_signboards, queue_size=5)

        rospy.loginfo(
            f"[landmark_anchor] {self.signboard_topic} + "
            f"{self.semantic_map_topic} + {self.drone_pose_topic} "
            f"-> {self.anchor_topic} "
            f"(assoc {self.assoc_radius_m:.1f} m, "
            f"min_period {self.min_period_s:.2f} s)")

    def on_map(self, msg):
        with self.lock:
            self.sem_map = msg

    def on_pose(self, msg):
        with self.lock:
            self.drone_pose = msg

    def on_signboards(self, msg):
        with self.lock:
            sem = self.sem_map
            pose = self.drone_pose
            last_t = self.last_publish_t
        if sem is None or pose is None:
            return
        if not sem.stores:
            return

        t_now = rospy.Time.now().to_sec()
        if t_now - last_t < self.min_period_s:
            return

        yaw = _yaw_from_quat(pose.pose.orientation)
        cy, sy = math.cos(yaw), math.sin(yaw)
        dx_w, dy_w, dz_w = (pose.pose.position.x,
                            pose.pose.position.y,
                            pose.pose.position.z)

        best = None  # (store, drone_world_true_xyz, residual)
        for obs in msg.signboards:
            pb = obs.position_base
            if not _isfinite_point(pb):
                continue
            r = math.sqrt(pb.x * pb.x + pb.y * pb.y + pb.z * pb.z)
            if r < self.min_range_m or r > self.max_range_m:
                continue

            # body -> world rotation (yaw only)
            rwx = cy * pb.x - sy * pb.y
            rwy = sy * pb.x + cy * pb.y
            rwz = pb.z

            sb_world_obs = (dx_w + rwx, dy_w + rwy, dz_w + rwz)

            # Nearest known store.
            nearest, nd = None, float("inf")
            for s in sem.stores:
                if not _isfinite_point(s.position_world):
                    continue
                d = math.hypot(
                    sb_world_obs[0] - s.position_world.x,
                    sb_world_obs[1] - s.position_world.y)
                if d < nd:
                    nearest, nd = s, d
            if nearest is None or nd > self.assoc_radius_m:
                continue

            # Drone TRUE world pose: subtract the body-rotated obs
            # from the KNOWN store world position.
            drone_true = (nearest.position_world.x - rwx,
                          nearest.position_world.y - rwy,
                          nearest.position_world.z - rwz)

            if best is None or nd < best[2]:
                best = (nearest, drone_true, nd)

        if best is None:
            return
        store, drone_true, residual = best

        out = PoseStamped()
        # Use the perception message stamp so vio_bridge can reason
        # about staleness; fall back to now if header is zero.
        out.header.stamp = (msg.header.stamp
                            if msg.header.stamp.to_sec() > 0
                            else rospy.Time.now())
        out.header.frame_id = self.world_frame
        out.pose.position.x = drone_true[0]
        out.pose.position.y = drone_true[1]
        out.pose.position.z = drone_true[2]
        # Keep the drone's current orientation; we are only correcting
        # XYZ drift here. yaw alignment is a separate (harder)
        # problem out of scope for the sparse-anchor design.
        out.pose.orientation = pose.pose.orientation
        self.pub.publish(out)

        with self.lock:
            self.last_publish_t = t_now
            self.publish_count += 1

        rospy.loginfo_throttle(
            1.0,
            f"[landmark_anchor] store_id={store.store_id} "
            f"residual={residual:.2f} m -> "
            f"drone_true=({drone_true[0]:.2f},{drone_true[1]:.2f},"
            f"{drone_true[2]:.2f}) "
            f"(emitted={self.publish_count})")


if __name__ == "__main__":
    try:
        LandmarkAnchorPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
