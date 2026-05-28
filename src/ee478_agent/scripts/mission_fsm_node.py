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
        # How far in front of the gate to ALIGN laterally onto the
        # chosen lane. Without an explicit align waypoint the drone
        # would fly the (approach -> past_gate) leg diagonally and
        # clip the gate post sitting at gate_x.
        self.lane_align_back_m = float(rospy.get_param(
            "~lane_align_back_m", 0.7))
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
        # Waypoint queue + continuation. _enqueue_path() fills these.
        self.nav_queue = []
        self.nav_after_state = None
        self.nav_yaw = 0.0
        # World layout corridors (see ee478_project_simple.world):
        #   central wall at x=13 blocks y in [-2, 2]
        #   cafe wall    at x=16 blocks y in [1, 4]
        #   pharmacy wall at x=16 blocks y in [-4, -1]
        # The safest lateral corridor exists at y=+4.5 (centred in the
        # 1 m gap between cafe wall top y=4 and outer wall y=5, giving
        # 0.5 m clearance on BOTH sides) and y=-4.5 (mirror). y=4.3
        # was tried first but only had 0.3 m to the cafe wall top —
        # too tight against the wall the drone is trying to skip past.
        self.bypass_y_pos = float(rospy.get_param("~bypass_y_pos",  4.5))
        self.bypass_y_neg = float(rospy.get_param("~bypass_y_neg", -4.5))
        # X of the two intermediate waypoints along the bypass corridor.
        # bypass_x_in=6 means the drone climbs to the bypass altitude
        # IMMEDIATELY past the gate, getting above the obstacle field
        # (poles around x=6-10, max y of pole tips ~2.8) instead of
        # threading through it.
        self.bypass_x_in  = float(rospy.get_param("~bypass_x_in",  6.0))
        self.bypass_x_out = float(rospy.get_param("~bypass_x_out", 17.0))
        # When an obstacle-aware planner (EGO) is downstream, we do
        # NOT pre-route the drone through bypass corridors; the
        # planner finds its own path. set ~use_bypass=false in that
        # mode so NAV_STORE / RETURN_PATH publish only the high-level
        # final goal. With the direct goal follower (no avoidance),
        # ~use_bypass=true is required to navigate the corridor walls.
        self.use_bypass = bool(rospy.get_param("~use_bypass", True))
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

    def _bypass_y_for_store(self, sy):
        """Choose the +y or -y bypass corridor to use based on the
        target store's y. Cafe (y=+2.5) -> use +y corridor; pharmacy
        (y=-2.5) -> use -y. Burger (y=0) -> default to +y."""
        return self.bypass_y_pos if sy >= 0 else self.bypass_y_neg

    def _enqueue_path(self, waypoints, after_state, yaw=0.0):
        """Queue a list of (x, y) waypoints; transition to
        `after_state` once the last one is reached. Each waypoint is
        published in sequence as the previous one's arrival is
        detected. yaw is the heading commanded at every waypoint."""
        with self.lock:
            self.nav_queue = [tuple(w) for w in waypoints]
            self.nav_after_state = after_state
            self.nav_yaw = yaw
        self._set_state("FOLLOW_PATH")

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
                    self._set_state("ALIGN_LANE")
        elif state == "ALIGN_LANE":
            # Shift LATERALLY to the chosen-lane y BEFORE the gate so
            # the THROUGH leg is a straight forward pass. Without
            # this, the diagonal from (approach, y=0) to (past_gate,
            # lane_y) clips the gate post at x=gate_x.
            ax = qp.pose.position.x - self.lane_align_back_m
            ay = qp.pose.position.y
            self._publish_goal(ax, ay, label=2)
            self._set_state("WAIT_ALIGN_LANE")
        elif state == "WAIT_ALIGN_LANE":
            if self._pending_reached():
                self._clear_pending()
                self._set_state("THROUGH_QUIZ")
        elif state == "THROUGH_QUIZ":
            # Fly forward through the chosen gate, ending up
            # `gate_pass_forward_m` past it.
            gx = qp.pose.position.x + self.gate_pass_forward_m
            gy = qp.pose.position.y
            self._publish_goal(gx, gy, label=3)
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
            # Facades face -x in our world (stores at x=21, yaw=-1.57).
            # Stand `standoff` in -x of the facade.
            sx, sy = s.position_world.x, s.position_world.y
            store_xy = (sx - self.store_standoff_m, sy)
            if self.use_bypass:
                # Bypass the corridor walls at x=13 and x=16 via the
                # narrow gap between cafe/pharmacy walls and the
                # outer y=±5 walls. Used only with the direct
                # planner; an obstacle-aware planner picks its own.
                by = self._bypass_y_for_store(sy)
                waypoints = [
                    (self.bypass_x_in,  by),
                    (self.bypass_x_out, by),
                    store_xy,
                ]
                rospy.loginfo(
                    f"[mission_fsm] NAV_STORE waypoints "
                    f"(bypass y={by:.1f}): {waypoints}")
            else:
                waypoints = [store_xy]
                rospy.loginfo(
                    f"[mission_fsm] NAV_STORE direct (planner handles "
                    f"avoidance): {waypoints}")
            self._enqueue_path(waypoints, after_state="SIGNATURE",
                               yaw=0.0)
        elif state == "FOLLOW_PATH":
            self._tick_follow_path()
        elif state == "SIGNATURE":
            s = self._find_target_store()
            sid = s.store_id if s is not None else 0
            self.pub_sig.publish(Int32(data=sid))
            self._set_state("WAIT_SIGNATURE")
        elif state == "WAIT_SIGNATURE":
            if sig_done:
                self._set_state("RETURN_PATH")
        elif state == "RETURN_PATH":
            # Reverse the NAV_STORE bypass, drop through outbound gate,
            # land at pickup. All one path so we use the same queue.
            s = self._find_target_store()
            if s is None or outbound is None or sm is None:
                rospy.logwarn(
                    "[mission_fsm] missing context for return; "
                    "jumping straight to pickup")
                px = sm.pickup_point.x if sm is not None else 0.0
                py = sm.pickup_point.y if sm is not None else 0.0
                self._enqueue_path([(px, py)], after_state="DONE",
                                   yaw=math.pi)
                return
            sy = s.position_world.y
            pk = sm.pickup_point
            if self.use_bypass:
                by = self._bypass_y_for_store(sy)
                waypoints = [
                    (self.bypass_x_out, by),
                    (self.bypass_x_in,  by),
                    (outbound[0], outbound[1]),
                    (pk.x, pk.y),
                ]
                rospy.loginfo(
                    f"[mission_fsm] RETURN_PATH waypoints "
                    f"(bypass y={by:.1f}): {waypoints}")
            else:
                waypoints = [
                    (outbound[0], outbound[1]),
                    (pk.x, pk.y),
                ]
                rospy.loginfo(
                    f"[mission_fsm] RETURN_PATH direct (planner "
                    f"handles avoidance): {waypoints}")
            self._enqueue_path(waypoints, after_state="DONE",
                               yaw=math.pi)
        elif state == "DONE":
            rospy.loginfo_throttle(
                10.0, "[mission_fsm] MISSION DONE")

    def _tick_follow_path(self):
        with self.lock:
            queue = list(self.nav_queue)
            after = self.nav_after_state
            yaw = self.nav_yaw
            pending = self.pending_goal_xyz
        if not queue:
            if after is not None:
                self._set_state(after)
            return
        if pending is None:
            # Publish next waypoint.
            wx, wy = queue[0]
            self._publish_goal(wx, wy, yaw=yaw, label=len(queue))
            return
        if self._pending_reached():
            self._clear_pending()
            with self.lock:
                if self.nav_queue:
                    done = self.nav_queue.pop(0)
                else:
                    done = None
            rospy.loginfo(
                f"[mission_fsm] reached waypoint {done}; "
                f"{len(self.nav_queue)} left")


if __name__ == "__main__":
    try:
        MissionFSM()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
