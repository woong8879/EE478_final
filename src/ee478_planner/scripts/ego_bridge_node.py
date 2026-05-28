#!/usr/bin/env python3
"""ee478_planner/ego_bridge_node.py

Glue between the EE478 agent and EGO-Planner-V2.

Two directions:

GOAL (agent -> EGO)
  /next_goal              (geometry_msgs/PoseStamped)   -- from agent
    -> /goal_with_id      (quadrotor_msgs/GoalSet)      -- to EGO
  EGO's FSM resets `have_trigger_` to False after every short
  trajectory segment, so we ALSO re-publish the goal at ~1 Hz until
  /goal_reached fires; otherwise the planner parks in WAIT_TARGET and
  the drone stops mid-corridor.

CMD (EGO -> offboard)
  /drone_<id>_planning/pos_cmd  (quadrotor_msgs/PositionCommand 100 Hz)
    -> /offboard/cmd_raw         (mavros_msgs/PositionTarget)
  Forward the planner's full state (pos + vel + accel + yaw + yaw_rate)
  to PX4 via offboard_controller. PositionTarget with type_mask=0 tells
  PX4 to use the feedforward velocities/accelerations, not just chase
  position.

ARRIVAL
  When the drone gets within `arrival_radius_m` of the active goal we
  publish /goal_reached(store_id) so agent_interface advances state.
"""

import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from mavros_msgs.msg import PositionTarget
from quadrotor_msgs.msg import PositionCommand, GoalSet

from ee478_msgs.msg import SemanticMap


class EgoBridgeNode:
    def __init__(self):
        rospy.init_node("ego_bridge")
        self.lock = threading.RLock()

        self.drone_id = int(rospy.get_param("~drone_id", 0))
        self.goal_in_topic   = rospy.get_param("~goal_in_topic",   "/next_goal")
        self.goal_out_topic  = rospy.get_param("~goal_out_topic",  "/goal_with_id")
        self.cmd_in_topic    = rospy.get_param("~cmd_in_topic",
                                               "/drone_0_planning/pos_cmd")
        self.cmd_out_topic   = rospy.get_param("~cmd_out_topic",   "/offboard/cmd_raw")
        self.world_frame     = rospy.get_param("~world_frame",     "map")
        self.arrival_radius  = float(rospy.get_param("~arrival_radius_m", 0.6))
        self.tick_hz         = float(rospy.get_param("~tick_hz", 20.0))
        self.resend_period_s = float(rospy.get_param("~resend_period_s", 1.0))

        self.sem_map = None
        self.drone_pose = None
        self.pending_goal_xy = None
        self.pending_store_id = None
        self.arrived_published = False
        self.last_goal_world = None
        self._last_resend_t = 0.0

        self.pub_goal = rospy.Publisher(self.goal_out_topic, GoalSet, queue_size=2)
        self.pub_cmd  = rospy.Publisher(self.cmd_out_topic,  PositionTarget,
                                        queue_size=10)
        self.pub_reached = rospy.Publisher("/goal_reached", Int32, queue_size=5)

        rospy.Subscriber(self.goal_in_topic, PoseStamped,
                         self.on_goal_in, queue_size=2)
        rospy.Subscriber(self.cmd_in_topic, PositionCommand,
                         self.on_cmd_in, queue_size=10)
        rospy.Subscriber("/semantic_map", SemanticMap, self.on_map, queue_size=1)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)

        rospy.Timer(rospy.Duration(1.0 / self.tick_hz), self.tick)
        rospy.loginfo(
            f"[ego_bridge] {self.goal_in_topic} -> {self.goal_out_topic} "
            f"(GoalSet) | {self.cmd_in_topic} -> {self.cmd_out_topic} "
            f"(PositionTarget @ stream)")

    # ----- inbound -----
    def on_goal_in(self, msg):
        gx = float(msg.pose.position.x)
        gy = float(msg.pose.position.y)
        gz = float(msg.pose.position.z)
        gs = GoalSet()
        gs.drone_id = self.drone_id
        gs.goal = [gx, gy, gz]
        # Burst publish: EGO sometimes drops the first one or two
        # while it's still finishing a previous traj.
        for _ in range(5):
            self.pub_goal.publish(gs)
            rospy.sleep(0.1)
        with self.lock:
            self.last_goal_world = (gx, gy, gz)
            self.pending_goal_xy = (gx, gy)
            self.pending_store_id = self._snap_to_store(gx, gy)
            self.arrived_published = False
            self._last_resend_t = rospy.Time.now().to_sec()
        rospy.loginfo(
            f"[ego_bridge] forwarded goal ({gx:.2f},{gy:.2f},{gz:.2f}) "
            f"-> EGO drone_{self.drone_id}, /goal_reached={self.pending_store_id}")

    def on_cmd_in(self, cmd):
        """Convert EGO PositionCommand -> mavros PositionTarget with
        FULL pos+vel+accel+yaw feedforward (type_mask=0)."""
        out = PositionTarget()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.world_frame
        out.coordinate_frame = PositionTarget.FRAME_LOCAL_NED  # mavros: ENU
        out.type_mask = 0
        out.position.x = cmd.position.x
        out.position.y = cmd.position.y
        out.position.z = cmd.position.z
        out.velocity.x = cmd.velocity.x
        out.velocity.y = cmd.velocity.y
        out.velocity.z = cmd.velocity.z
        out.acceleration_or_force.x = cmd.acceleration.x
        out.acceleration_or_force.y = cmd.acceleration.y
        out.acceleration_or_force.z = cmd.acceleration.z
        out.yaw = cmd.yaw
        out.yaw_rate = cmd.yaw_dot
        self.pub_cmd.publish(out)

    def on_map(self, msg):
        with self.lock:
            self.sem_map = msg

    def on_pose(self, msg):
        with self.lock:
            self.drone_pose = msg
        self._check_arrival()

    # ----- helpers -----
    def _snap_to_store(self, gx, gy):
        with self.lock:
            sem = self.sem_map
        if sem is None:
            return -1
        best, best_d = -1, float("inf")
        pk = sem.pickup_point
        d = math.hypot(gx - pk.x, gy - pk.y)
        if d < best_d:
            best, best_d = -1, d
        for e in sem.stores:
            d = math.hypot(gx - e.position_world.x, gy - e.position_world.y)
            if d < best_d:
                best, best_d = e.store_id, d
        return int(best)

    def _check_arrival(self):
        with self.lock:
            if (self.arrived_published or self.pending_goal_xy is None
                    or self.drone_pose is None):
                return
            gx, gy = self.pending_goal_xy
            sid = self.pending_store_id
            p = self.drone_pose.pose.position
        if math.hypot(p.x - gx, p.y - gy) <= self.arrival_radius:
            with self.lock:
                self.arrived_published = True
            self.pub_reached.publish(Int32(data=sid))
            rospy.loginfo(
                f"[ego_bridge] arrived ({gx:.2f},{gy:.2f}) "
                f"-> /goal_reached={sid}")

    # ----- tick -----
    def tick(self, _evt):
        """Re-publish the active goal periodically until /goal_reached.

        EGO-Planner resets have_trigger_=false after every completed
        local trajectory; without a fresh GoalSet it parks in
        WAIT_TARGET and the drone stops mid-corridor."""
        with self.lock:
            goal = self.last_goal_world
            arrived = self.arrived_published
        if goal is None or arrived:
            return
        t_now = rospy.Time.now().to_sec()
        if t_now - self._last_resend_t < self.resend_period_s:
            return
        gs = GoalSet()
        gs.drone_id = self.drone_id
        gs.goal = [goal[0], goal[1], goal[2]]
        self.pub_goal.publish(gs)
        self._last_resend_t = t_now


if __name__ == "__main__":
    try:
        EgoBridgeNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
