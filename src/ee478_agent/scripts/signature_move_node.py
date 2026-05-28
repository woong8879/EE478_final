#!/usr/bin/env python3
"""ee478_agent/signature_move_node.py

Mission step 4 (signature): perform a "I'm here" maneuver in front of
the target store. The final-project spec replaces "drop item" with a
clear visual signature so a human reviewer can see arrival succeeded.

Sequence (relative to drone pose at trigger):
  1.  bobble up   +0.5 m
  2.  bobble down -0.5 m (back to nominal)
  3.  yaw spin   +180 deg
  4.  yaw spin   -180 deg (back to nominal)

Each waypoint is published to /next_goal and we wait for /goal_reached
(with the same store_id reused as a label) before moving on.

Trigger:  /mission/signature_trigger  (std_msgs/Int32 store_id)
Done:     /mission/signature_done     (std_msgs/Bool latched true on
                                       completion)
"""

import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32


def _yaw_to_quat(yaw):
    return (math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class SignatureMove:
    def __init__(self):
        rospy.init_node("signature_move")
        self.lock = threading.Lock()

        self.bobble_dz = float(rospy.get_param("~bobble_dz", 0.5))
        self.spin_dyaw = float(rospy.get_param(
            "~spin_dyaw_deg", 180.0)) * math.pi / 180.0
        self.step_timeout = float(rospy.get_param("~step_timeout_s", 8.0))
        self.world_frame = rospy.get_param("~world_frame", "map")
        self.arrival_radius = float(rospy.get_param(
            "~arrival_radius_m", 0.35))

        self.drone_pose = None
        self.active_store_id = None
        self.steps = []
        self.step_idx = 0
        self.last_pub_t = rospy.Time(0)
        self.waiting_arrival = False

        self.pub_goal = rospy.Publisher("/next_goal", PoseStamped,
                                        queue_size=5)
        self.pub_done = rospy.Publisher(
            "/mission/signature_done", Bool, queue_size=1, latch=True)

        rospy.Subscriber("/mission/signature_trigger", Int32,
                         self.on_trigger, queue_size=2)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)

        rospy.Timer(rospy.Duration(0.25), self.tick)
        rospy.loginfo(
            f"[signature_move] ready (bobble_dz={self.bobble_dz:.2f} m, "
            f"spin={math.degrees(self.spin_dyaw):.0f} deg)")

    def on_pose(self, msg):
        with self.lock:
            self.drone_pose = msg

    def on_trigger(self, msg):
        with self.lock:
            if self.active_store_id is not None:
                rospy.logwarn(
                    f"[signature_move] already running for "
                    f"store={self.active_store_id}; ignoring new "
                    f"trigger={msg.data}")
                return
            if self.drone_pose is None:
                rospy.logwarn(
                    "[signature_move] no drone pose yet; ignoring")
                return
            p = self.drone_pose.pose.position
            yaw0 = _yaw_from_quat(self.drone_pose.pose.orientation)
            # NO z bobble: any altitude change while EGO is still
            # active competes with EGO's own z plan and ends with the
            # drone on the floor in sim. The "signature" is purely a
            # 360 deg yaw rotation in place at hover_z.
            hover_z = float(rospy.get_param("~hover_z", 0.7))
            self.steps = [
                (p.x, p.y, hover_z, yaw0 + self.spin_dyaw),
                (p.x, p.y, hover_z, yaw0),
            ]
            self.step_idx = 0
            self.active_store_id = int(msg.data)
            self.waiting_arrival = False
        rospy.loginfo(
            f"[signature_move] start for store={msg.data} "
            f"steps={len(self.steps)}")

    def _publish_step(self):
        with self.lock:
            if not self.steps or self.step_idx >= len(self.steps):
                return
            x, y, z, yaw = self.steps[self.step_idx]
        g = PoseStamped()
        g.header.stamp = rospy.Time.now()
        g.header.frame_id = self.world_frame
        g.pose.position.x = x
        g.pose.position.y = y
        g.pose.position.z = z
        qz, qw = _yaw_to_quat(yaw)
        g.pose.orientation.z = qz
        g.pose.orientation.w = qw
        self.pub_goal.publish(g)
        rospy.loginfo(
            f"[signature_move] step {self.step_idx + 1}/{len(self.steps)}"
            f" -> ({x:.2f},{y:.2f},{z:.2f}) "
            f"yaw_deg={math.degrees(yaw):.0f}")
        self.last_pub_t = rospy.Time.now()
        self.waiting_arrival = True

    def _arrived(self):
        with self.lock:
            if self.drone_pose is None or self.step_idx >= len(self.steps):
                return False
            x, y, z, yaw_t = self.steps[self.step_idx]
            p = self.drone_pose.pose.position
            yaw_now = _yaw_from_quat(self.drone_pose.pose.orientation)
        dxy = math.hypot(p.x - x, p.y - y)
        dz = abs(p.z - z)
        # wrap yaw error to [-pi, pi]
        dyaw = math.atan2(math.sin(yaw_t - yaw_now),
                          math.cos(yaw_t - yaw_now))
        return (dxy < self.arrival_radius
                and dz < self.arrival_radius
                and abs(dyaw) < math.radians(20))

    def tick(self, _evt):
        with self.lock:
            if self.active_store_id is None:
                return
            done = (self.step_idx >= len(self.steps))
        if done:
            self.pub_done.publish(Bool(data=True))
            rospy.loginfo("[signature_move] complete")
            with self.lock:
                self.active_store_id = None
                self.steps = []
                self.step_idx = 0
            return

        if not self.waiting_arrival:
            self._publish_step()
            return

        # arrived or timed out -> advance
        elapsed = (rospy.Time.now() - self.last_pub_t).to_sec()
        if self._arrived() or elapsed > self.step_timeout:
            with self.lock:
                self.step_idx += 1
                self.waiting_arrival = False


if __name__ == "__main__":
    try:
        SignatureMove()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
