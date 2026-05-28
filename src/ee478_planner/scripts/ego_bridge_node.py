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
        # EGO planner sometimes proposes huge z excursions (it tries
        # to "fly over" obstacles when grid_map shows tightly packed
        # cells near the lane). Clamp z to the mission hover range so
        # the drone stays at ~0.7 m and goes THROUGH the gate aperture
        # instead of over the top bar.
        self.z_min = float(rospy.get_param("~z_min", 0.5))
        self.z_max = float(rospy.get_param("~z_max", 1.0))
        # Hard z lock: completely override EGO's z command with a
        # constant hover altitude. EGO's optimiser otherwise produces
        # downward velocity segments at random during replans, and
        # PX4 follows them to the floor before the next clamp tick.
        # The mission is structured so the drone never NEEDS a z
        # change (gate aperture, store standoff, pickup all sit at
        # the same altitude). signature_move handles its own z bobble
        # by toggling ~z_locked off via a service if needed.
        self.z_locked = bool(rospy.get_param("~z_locked", True))
        self.hover_z = float(rospy.get_param("~hover_z", 0.7))

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
        # Mirror /next_goal to /offboard/goal so offboard_controller's
        # pose-setpoint stream stays at the active FSM goal even when
        # EGO momentarily stops publishing pos_cmd (replan timeouts,
        # planning failures). Without this fallback, PX4 sees a
        # setpoint gap > COM_OF_LOSS_T and falls back to POSCTL,
        # which leaves the drone slowly descending until it lands.
        self.pub_offboard_goal = rospy.Publisher(
            "/offboard/goal", PoseStamped, queue_size=2)
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
        # Mirror to /offboard/goal so offboard_controller has a
        # pose-setpoint anchor even when EGO temporarily can't plan.
        self.pub_offboard_goal.publish(msg)
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
        FULL pos+vel+accel+yaw feedforward (type_mask=0).

        z is clamped to the mission hover band so EGO's
        occasional over-the-top trajectory proposals don't send the
        drone into the gate's top bar or above the corridor walls."""
        out = PositionTarget()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.world_frame
        out.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        out.type_mask = 0
        out.position.x = cmd.position.x
        out.position.y = cmd.position.y
        z_raw = cmd.position.z
        if self.z_locked:
            # Override EGO's z entirely. We always fly at hover_z, the
            # planner only handles xy avoidance.
            z_clamped = self.hover_z
        else:
            z_clamped = max(self.z_min, min(self.z_max, z_raw))
        out.position.z = z_clamped
        out.velocity.x = cmd.velocity.x
        out.velocity.y = cmd.velocity.y
        if self.z_locked:
            # In locked mode, ignore EGO's vz and feedforward a small
            # PD correction back to hover_z. This swamps any rate of
            # descent PX4 might still try to apply.
            # (current_z is unknown here, so just zero the feedforward
            #  and let PX4's position controller do the correction.)
            out.velocity.z = 0.0
            out.acceleration_or_force.x = cmd.acceleration.x
            out.acceleration_or_force.y = cmd.acceleration.y
            out.acceleration_or_force.z = 0.0
            out.yaw = cmd.yaw
            out.yaw_rate = cmd.yaw_dot
            self.pub_cmd.publish(out)
            return
        # KEY FIX: also clamp the velocity-z feedforward. EGO sometimes
        # produces trajectory segments with vz<0 even when the goal
        # is at hover altitude (it's "ducking under" a phantom
        # obstacle above the corridor). If we pass vz<0 through, PX4
        # follows the feedforward and descends ~0.5 m/s — the drone
        # is on the ground in 1.5 s. Force vz to PUSH BACK toward the
        # mid hover band whenever EGO is asking us to leave it.
        mid_z = 0.5 * (self.z_min + self.z_max)
        if z_raw < self.z_min:
            # EGO wants to go below floor; force ascent.
            out.velocity.z = max(0.0, cmd.velocity.z)
        elif z_raw > self.z_max:
            # EGO wants to go above ceiling; force descent.
            out.velocity.z = min(0.0, cmd.velocity.z)
        else:
            # Inside hover band but pass through. Also veto a vz that
            # would CARRY THE DRONE OUT of the band on the next tick.
            if cmd.velocity.z < 0 and z_clamped <= self.z_min + 0.05:
                out.velocity.z = 0.0
            elif cmd.velocity.z > 0 and z_clamped >= self.z_max - 0.05:
                out.velocity.z = 0.0
            else:
                out.velocity.z = cmd.velocity.z
        out.acceleration_or_force.x = cmd.acceleration.x
        out.acceleration_or_force.y = cmd.acceleration.y
        out.acceleration_or_force.z = (cmd.acceleration.z
                                       if z_clamped == z_raw else 0.0)
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
