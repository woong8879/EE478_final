#!/usr/bin/env python3
"""ee478_agent/mission_fsm_node.py

Top-level state machine for the EE478 final-project mission.

Mission flow (5 user-visible steps, per the project image):

  1. READ COMMAND       LLM/keyword classifier emits /mission_target.
  2. SOLVE QUIZ         drone navigates to a hover IN FRONT of the
                        quiz signboard, waits for /quiz/chosen_pose
                        from quiz_solver_node, then flies through it.
  3. NAVIGATE COURSE    fly toward the target store (planner handles
                        avoidance) — picked from /semantic_map by
                        category.
  4. STORE + SIGNATURE  on /goal_reached(store_id), trigger
                        signature_move_node and wait for
                        /mission/signature_done.
  5. RETURN             fly back through the SAME gate used in step 2
                        (no quiz on return), then to pickup_point.

State diagram:
  INIT
   -> AWAIT_CMD        wait for /mission_target
   -> AWAIT_MAP        wait for /semantic_map + /quiz/gates
   -> AWAIT_TAKEOFF    wait for drone z > threshold
   -> APPROACH_QUIZ    fly to a hover pose IN FRONT of the quiz pair
   -> THROUGH_QUIZ     fly through /quiz/chosen_pose
   -> NAV_STORE        fly to the target-category store
   -> SIGNATURE        run signature_move, wait for done
   -> RETURN_GATE      fly back through the SAME chosen gate
   -> RETURN_PICKUP    fly to pickup_point
   -> DONE

Inputs:
  /mission_target           std_msgs/String     "CAFE" / ...
  /semantic_map             ee478_msgs/SemanticMap
  /quiz/chosen_pose         geometry_msgs/PoseStamped
  /quiz/chosen_label        std_msgs/Int32
  /mavros/local_position/pose
  /goal_reached             std_msgs/Int32
  /mission/signature_done   std_msgs/Bool

Outputs:
  /next_goal                geometry_msgs/PoseStamped
  /mission/signature_trigger std_msgs/Int32
  /mission/state            std_msgs/String (latched)
"""

import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32, String

from ee478_msgs.msg import SemanticMap


HOVER_Z = 0.7
TAKEOFF_THRESHOLD_M = 0.30


def _yaw_to_quat(yaw):
    return (math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class MissionFSM:
    def __init__(self):
        rospy.init_node("mission_fsm")
        self.lock = threading.RLock()

        self.world_frame = rospy.get_param("~world_frame", "map")
        self.hover_z = float(rospy.get_param("~hover_z", HOVER_Z))
        # how far IN FRONT of the gate (along -x in world) to hover
        # while waiting for the quiz solver to choose a lane.
        self.quiz_approach_back_m = float(rospy.get_param(
            "~quiz_approach_back_m", 2.0))
        # how far PAST the gate to fly so we're committed to a lane
        # before turning to the store. Same offset reused on return.
        self.gate_pass_forward_m = float(rospy.get_param(
            "~gate_pass_forward_m", 1.0))
        # store-approach standoff (positive = in front of the facade).
        self.store_standoff_m = float(rospy.get_param(
            "~store_standoff_m", 1.2))
        # FSM tick rate.
        self.tick_hz = float(rospy.get_param("~tick_hz", 4.0))

        self.state = "INIT"
        self.drone_pose = None
        self.sem_map = None
        self.target_cat = None
        self.quiz_pose = None
        self.quiz_label = None
        self.outbound_gate_xy = None     # remember for return leg
        self.pending_goal_xyz = None     # PoseStamped.position we last sent
        self.pending_min_dxy = float("inf")  # closest xy approach since publish
        self.pending_min_dz = float("inf")
        self.arrival_radius = float(
            rospy.get_param("~arrival_radius_m", 0.5))
        # Vertical arrival is checked separately because PX4's altitude
        # hold can run a 10-20 cm offset.
        self.arrival_radius_z = float(
            rospy.get_param("~arrival_radius_z_m", 0.4))
        self.signature_done = False
        self.takeoff_latched = False

        self.pub_goal = rospy.Publisher("/next_goal", PoseStamped,
                                        queue_size=5)
        self.pub_sig = rospy.Publisher(
            "/mission/signature_trigger", Int32, queue_size=2)
        self.pub_state = rospy.Publisher(
            "/mission/state", String, queue_size=1, latch=True)

        rospy.Subscriber("/mission_target", String,
                         self.on_target, queue_size=2)
        rospy.Subscriber("/semantic_map", SemanticMap,
                         self.on_map, queue_size=1)
        rospy.Subscriber("/quiz/chosen_pose", PoseStamped,
                         self.on_quiz_pose, queue_size=2)
        rospy.Subscriber("/quiz/chosen_label", Int32,
                         self.on_quiz_label, queue_size=2)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)
        rospy.Subscriber("/goal_reached", Int32,
                         self.on_reached, queue_size=5)
        rospy.Subscriber("/mission/signature_done", Bool,
                         self.on_sig_done, queue_size=2)

        rospy.Timer(rospy.Duration(1.0 / self.tick_hz), self.tick)
        self._set_state("INIT")
        rospy.loginfo("[mission_fsm] ready")

    # ---------- callbacks ----------
    def on_pose(self, msg):
        with self.lock:
            self.drone_pose = msg
            if (not self.takeoff_latched
                    and msg.pose.position.z > TAKEOFF_THRESHOLD_M):
                self.takeoff_latched = True
                rospy.loginfo(
                    f"[mission_fsm] takeoff latched at "
                    f"z={msg.pose.position.z:.2f}")
            # Update closest-approach to the active goal so a fast
            # pass-through still triggers arrival on the next tick.
            if self.pending_goal_xyz is not None:
                gx, gy, gz = self.pending_goal_xyz
                p = msg.pose.position
                dxy = math.hypot(p.x - gx, p.y - gy)
                dz = abs(p.z - gz)
                if dxy < self.pending_min_dxy:
                    self.pending_min_dxy = dxy
                if dz < self.pending_min_dz:
                    self.pending_min_dz = dz

    def on_target(self, msg):
        with self.lock:
            self.target_cat = msg.data.strip().upper()
        rospy.loginfo(
            f"[mission_fsm] target category = {self.target_cat}")

    def on_map(self, msg):
        with self.lock:
            self.sem_map = msg

    def on_quiz_pose(self, msg):
        with self.lock:
            self.quiz_pose = msg
            self.outbound_gate_xy = (msg.pose.position.x,
                                     msg.pose.position.y)

    def on_quiz_label(self, msg):
        with self.lock:
            self.quiz_label = int(msg.data)

    def on_reached(self, msg):
        # /goal_reached is informational only. The FSM uses position-
        # based arrival against pending_goal_xyz so it never has to
        # coordinate label semantics with the goal follower.
        pass

    def on_sig_done(self, msg):
        with self.lock:
            if msg.data:
                self.signature_done = True

    # ---------- helpers ----------
    def _publish_goal(self, x, y, z=None, yaw=0.0, label=None):
        z = self.hover_z if z is None else z
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
        with self.lock:
            self.pending_goal_xyz = (x, y, z)
            # Track the closest the drone has gotten to the goal since
            # publish. Without this, an overshoot at >2 m/s would miss
            # the 0.5 m arrival window between 4 Hz FSM ticks.
            self.pending_min_dxy = float("inf")
            self.pending_min_dz = float("inf")
        rospy.loginfo(
            f"[mission_fsm] goal ({x:.2f},{y:.2f},{z:.2f}) "
            f"yaw_deg={math.degrees(yaw):.0f} label={label}")

    def _pending_reached(self):
        with self.lock:
            if self.pending_goal_xyz is None:
                return False
            return (self.pending_min_dxy < self.arrival_radius
                    and self.pending_min_dz < self.arrival_radius_z)

    def _clear_pending(self):
        with self.lock:
            self.pending_goal_xyz = None
            self.pending_min_dxy = float("inf")
            self.pending_min_dz = float("inf")

    def _find_target_store(self):
        """Return StoreEntry whose category == target_cat, else None."""
        with self.lock:
            sm = self.sem_map
            cat = self.target_cat
        if sm is None or cat is None:
            return None
        for s in sm.stores:
            if s.category.upper() == cat:
                return s
        return None

    def _set_state(self, s):
        with self.lock:
            if s != self.state:
                rospy.loginfo(
                    f"[mission_fsm] state {self.state} -> {s}")
                self.state = s
            self.pub_state.publish(String(data=s))

    # ---------- tick ----------
    def tick(self, _evt):
        with self.lock:
            state = self.state
            pose = self.drone_pose
            sm = self.sem_map
            qp = self.quiz_pose
            tcat = self.target_cat
            sig_done = self.signature_done
            took_off = self.takeoff_latched
            outbound = self.outbound_gate_xy

        if state == "INIT":
            self._set_state("AWAIT_CMD")
        elif state == "AWAIT_CMD":
            if tcat is not None:
                self._set_state("AWAIT_MAP")
        elif state == "AWAIT_MAP":
            if sm is not None and pose is not None:
                self._set_state("AWAIT_TAKEOFF")
        elif state == "AWAIT_TAKEOFF":
            if took_off:
                self._set_state("APPROACH_QUIZ")
        elif state == "APPROACH_QUIZ":
            # Hover BACK from the gate-pair so perception can see both
            # lanes. We use the quiz pose's x and the lane-pair's
            # midline (y=0) — robust fallback if quiz_solver hasn't
            # published yet: aim for x=3 (the only quiz x in our world).
            tx = qp.pose.position.x if qp is not None else 3.0
            self._publish_goal(tx - self.quiz_approach_back_m, 0.0,
                               label=1)
            self._set_state("WAIT_APPROACH_QUIZ")
        elif state == "WAIT_APPROACH_QUIZ":
            if self._pending_reached():
                self._clear_pending()
                if qp is None:
                    rospy.logwarn_throttle(
                        2.0,
                        "[mission_fsm] at approach but no chosen gate "
                        "yet; holding")
                else:
                    self._set_state("THROUGH_QUIZ")
        elif state == "THROUGH_QUIZ":
            # Fly forward through the chosen gate, ending up
            # `gate_pass_forward_m` past it.
            gx = qp.pose.position.x + self.gate_pass_forward_m
            gy = qp.pose.position.y
            self._publish_goal(gx, gy, label=2)
            self._set_state("WAIT_THROUGH_QUIZ")
        elif state == "WAIT_THROUGH_QUIZ":
            if self._pending_reached():
                self._clear_pending()
                self._set_state("NAV_STORE")
        elif state == "NAV_STORE":
            s = self._find_target_store()
            if s is None:
                rospy.logwarn_throttle(
                    2.0,
                    f"[mission_fsm] target {tcat} not in semantic map "
                    f"yet; waiting")
                return
            # stand `standoff` in -x of the facade. Facades face -x in
            # our world (cafe/burger/pharmacy at x=21, rotated -1.57).
            sx, sy = s.position_world.x, s.position_world.y
            self._publish_goal(sx - self.store_standoff_m, sy,
                               label=3)
            self._set_state("WAIT_NAV_STORE")
        elif state == "WAIT_NAV_STORE":
            if self._pending_reached():
                self._clear_pending()
                self._set_state("SIGNATURE")
        elif state == "SIGNATURE":
            s = self._find_target_store()
            sid = s.store_id if s is not None else 0
            self.pub_sig.publish(Int32(data=sid))
            self._set_state("WAIT_SIGNATURE")
        elif state == "WAIT_SIGNATURE":
            if sig_done:
                self._set_state("RETURN_GATE")
        elif state == "RETURN_GATE":
            # Fly back through the SAME chosen gate (no quiz this time).
            if outbound is None:
                rospy.logwarn(
                    "[mission_fsm] no outbound gate memorised; "
                    "returning direct to pickup")
                self._set_state("RETURN_PICKUP")
                return
            self._publish_goal(outbound[0], outbound[1],
                               yaw=math.pi, label=4)
            self._set_state("WAIT_RETURN_GATE")
        elif state == "WAIT_RETURN_GATE":
            if self._pending_reached():
                self._clear_pending()
                self._set_state("RETURN_PICKUP")
        elif state == "RETURN_PICKUP":
            pk = sm.pickup_point if sm is not None else None
            px = pk.x if pk is not None else 0.0
            py = pk.y if pk is not None else 0.0
            self._publish_goal(px, py, yaw=math.pi, label=5)
            self._set_state("WAIT_RETURN_PICKUP")
        elif state == "WAIT_RETURN_PICKUP":
            if self._pending_reached():
                self._clear_pending()
                self._set_state("DONE")
        elif state == "DONE":
            rospy.loginfo_throttle(
                10.0, "[mission_fsm] MISSION DONE")


if __name__ == "__main__":
    try:
        MissionFSM()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
