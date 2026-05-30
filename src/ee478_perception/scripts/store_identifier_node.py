#!/usr/bin/env python3
"""ee478_perception/store_identifier_node.py

Sub-task 5: pick the store that matches the mission target.

The drone hovers at a viewing point (default (15, 0, hover_z)) where
all three storefronts (cafe / pharmacy / burger) are inside the
camera FOV. We look up each store's category in /semantic_map (real
drone: filled by YOLO; sim: filled by sim_world_publisher from
gazebo GT), match against /mission_target, and publish

  /target_store_pose  (geometry_msgs/PoseStamped)
        the standoff point in front of the matching store's facade.

mission_orchestrator pipes that into /next_goal for sub-task 6.

Inputs:
  /semantic_map    (ee478_msgs/SemanticMap)
  /mission_target  (std_msgs/String)
Output:
  /target_store_pose  (geometry_msgs/PoseStamped, latched)
"""

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from ee478_msgs.msg import SemanticMap


class StoreIdentifier:
    def __init__(self):
        rospy.init_node("store_identifier")
        self.standoff_m = float(rospy.get_param("~standoff_m", 1.5))
        self.target_z   = float(rospy.get_param("~target_z", 0.7))
        self.world_frame = rospy.get_param("~world_frame", "map")

        self.sem_map = None
        self.target_cat = None
        self.published_for = None  # str: which (cat, store_id) we already emitted

        self.pub = rospy.Publisher(
            "/target_store_pose", PoseStamped, queue_size=1, latch=True)

        rospy.Subscriber("/semantic_map", SemanticMap,
                         self.on_map, queue_size=1)
        rospy.Subscriber("/mission_target", String,
                         self.on_target, queue_size=2)

        rospy.Timer(rospy.Duration(0.5), self.tick)
        rospy.loginfo(
            f"[store_id] semantic_map + mission_target -> "
            f"/target_store_pose (standoff {self.standoff_m:.2f} m)")

    def on_map(self, msg):
        self.sem_map = msg

    def on_target(self, msg):
        self.target_cat = msg.data.strip().upper()
        rospy.loginfo(f"[store_id] target = {self.target_cat}")

    def tick(self, _evt):
        if self.sem_map is None or self.target_cat is None:
            return
        match = None
        for s in self.sem_map.stores:
            if s.category.upper() == self.target_cat:
                match = s
                break
        if match is None:
            rospy.logwarn_throttle(
                2.0,
                f"[store_id] {self.target_cat} not in semantic_map "
                f"yet ({[s.category for s in self.sem_map.stores]})")
            return
        key = f"{self.target_cat}:{match.store_id}"
        if self.published_for == key:
            return
        # Facades face -x (stores at x=21, yaw=-pi/2 in ee478 world).
        # Stand `standoff` in -x of the facade.
        out = PoseStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.world_frame
        out.pose.position.x = match.position_world.x - self.standoff_m
        out.pose.position.y = match.position_world.y
        out.pose.position.z = self.target_z
        out.pose.orientation.w = 1.0
        self.pub.publish(out)
        self.published_for = key
        rospy.loginfo(
            f"[store_id] selected store_id={match.store_id} "
            f"category={match.category} -> approach pose "
            f"({out.pose.position.x:.2f}, {out.pose.position.y:.2f}, "
            f"{out.pose.position.z:.2f})")


if __name__ == "__main__":
    try:
        StoreIdentifier()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
