#!/usr/bin/env python3
"""ee478_agent/land_node.py

Sub-task 8: land at the current xy and disarm.

Triggered by /mission/land_trigger (std_msgs/Bool). On trigger we:
  1. Take the drone's current xy from /mavros/local_position/pose.
  2. Stream /next_goal at (xy, descend_z) at high rate so EGO /
     direct_goal_follower bring the drone down smoothly.
  3. Once gazebo z drops below ground_z, call mavros
     /mavros/cmd/arming with value=False to disarm.
  4. Publish /mission/land_done = True (latched).

Real-drone TODO: prefer AUTO.LAND via mavros set_mode rather than a
streamed setpoint — PX4 handles landing detection more gracefully
than the OFFBOARD-stream approach.
"""

import threading

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from mavros_msgs.srv import CommandBool


class LandNode:
    def __init__(self):
        rospy.init_node("land_node")
        self.lock = threading.Lock()

        self.descend_z   = float(rospy.get_param("~descend_z", 0.05))
        self.ground_z    = float(rospy.get_param("~ground_z",  0.10))
        self.rate_hz     = float(rospy.get_param("~rate_hz",   10.0))
        self.world_frame = rospy.get_param("~world_frame", "map")

        self.active = False
        self.drone_pose = None
        self.landed = False

        self.pub_goal = rospy.Publisher("/next_goal", PoseStamped,
                                        queue_size=5)
        self.pub_done = rospy.Publisher(
            "/mission/land_done", Bool, queue_size=1, latch=True)

        rospy.Subscriber("/mission/land_trigger", Bool,
                         self.on_trigger, queue_size=2)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)

        rospy.wait_for_service("/mavros/cmd/arming", timeout=120.0)
        self.srv_arm = rospy.ServiceProxy(
            "/mavros/cmd/arming", CommandBool)

        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.tick)
        rospy.loginfo(
            f"[land_node] ready; descend_z={self.descend_z:.2f} "
            f"ground_z={self.ground_z:.2f}")

    def on_trigger(self, msg):
        with self.lock:
            if not msg.data:
                return
            if self.active:
                return
            self.active = True
            self.landed = False
        rospy.loginfo("[land_node] land sequence start")

    def on_pose(self, msg):
        with self.lock:
            self.drone_pose = msg

    def tick(self, _evt):
        with self.lock:
            active = self.active
            landed = self.landed
            pose = self.drone_pose
        if not active or landed or pose is None:
            return

        # Stream a descend setpoint at the current xy.
        g = PoseStamped()
        g.header.stamp = rospy.Time.now()
        g.header.frame_id = self.world_frame
        g.pose.position.x = pose.pose.position.x
        g.pose.position.y = pose.pose.position.y
        g.pose.position.z = self.descend_z
        g.pose.orientation.w = 1.0
        self.pub_goal.publish(g)

        if pose.pose.position.z < self.ground_z:
            try:
                res = self.srv_arm(False)
                if res.success:
                    rospy.loginfo("[land_node] disarmed; LAND_DONE")
                    self.pub_done.publish(Bool(data=True))
                    with self.lock:
                        self.landed = True
            except Exception as e:
                rospy.logwarn(f"[land_node] disarm failed: {e}")


if __name__ == "__main__":
    try:
        LandNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
