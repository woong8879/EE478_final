#!/usr/bin/env python3
"""precision_land_node.py — find the target box behind the store gate and land
ON TOP of it.

Course geometry (map FLU, +y = drone-left at takeoff):
  Store gates on the x=3.5 wall, passed flying in -x:
    LEFT  gate (facing -x) = tags 271+274, centre (3.5, 6.2)
    RIGHT gate             = tags 275+276, centre (3.5, 8.2)
  Behind them (x < 3.5): 4 boxes, one per YOLO label
  (store/pharmacy/hamburger/cafe). Label images on the box FRONT (faces +x,
  seen by the forward camera) and on the box TOP (seen by the down camera).

State machine:
  PRE_TAKEOFF  wait armed + altitude (sequence cannot start on the bench)
  TO_GATE      /next_goal beyond the chosen gate centre -> EGO flies through
  SEARCH       force delivery RGB+YOLO on; strafe toward the UNSEEN side
               (passed left gate -> boxes to the drone's right = map +y) until
               the front YOLO sees the target label
  APPROACH     front-camera visual servo via /offboard/cmd_raw position steps:
               centre the bbox horizontally, step forward until the bbox is
               wide (close) -> stop at standoff
  MOUNT        climb, then step forward until the DOWN camera sees the box top
  CENTER_TOP   down-camera servo. Down cam: 14.5 cm BEHIND body centre,
               image TOP = drone BACK, image BOTTOM = drone FRONT
               -> bbox toward image bottom = box ahead of camera = step forward
  COMMIT       once centred STABLY (N consecutive frames), proceed even if
               detection drops (too close for YOLO): shift BACK 14.5 cm so the
               BODY centre (not the camera) is over the box centre, descend
               slowly, force-disarm at touchdown height.

All fine motion goes through /offboard/cmd_raw (absorbed by the offboard
velocity-PID: clamped, slewed, hover-hold watchdog if we stop publishing).
"""
import json
import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import PositionTarget, State as MavState
from mavros_msgs.srv import CommandLong
from std_msgs.msg import Bool, String


def _yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PrecisionLand(object):
    def __init__(self):
        rospy.init_node("precision_land")
        self.lock = threading.Lock()

        # ---- mission params ----
        self.target_label = rospy.get_param("~target_label", "cafe")
        self.gate_choice = rospy.get_param("~gate", "left").lower()
        self.left_gate = rospy.get_param("~left_gate", [3.5, 6.2])
        self.right_gate = rospy.get_param("~right_gate", [3.5, 8.2])
        self.hover_z = float(rospy.get_param("~hover_z", 0.7))
        self.pass_through = float(rospy.get_param("~pass_through_m", 1.2))
        self.pass_margin = float(rospy.get_param("~pass_margin_m", 0.8))

        # ---- search / approach ----
        # After the gate: first ADVANCE straight forward this far (clear of the
        # gate / into the box area), THEN strafe a fixed 2.5 m to ONE side while
        # the front YOLO scans (passed LEFT gate -> sweep the drone's RIGHT;
        # RIGHT -> LEFT). Stops early when the target is confirmed.
        # After the gate: advance advance_m forward (obstacles rule out scanning
        # from far back), THEN descend to scan_z. The key is the LOW altitude --
        # the front cam is tilted ~12.8 deg UP, so at hover_z the low 26 cm boxes
        # fall below its frame; dropping to scan_z brings them into the
        # up-tilted FOV even at this closer range.
        # 0: the planner already positioned the drone at the scan spot (0.725,5)
        # over the box row, so precision_land just descends + sweeps (no forward).
        self.advance_m = float(rospy.get_param("~advance_m", 0.0))
        # SEARCH altitude: drop to this before swinging. The front cam is tilted
        # ~12.8 deg UP (for the gate tags), so at hover_z=0.7 the low 26 cm boxes
        # sit BELOW its axis and fall out of frame. Descending lets the boxes
        # rise into the up-tilted view so the front YOLO can find them. Lower =
        # sees boxes from closer, but less ground clearance -- tune in flight.
        self.search_z = float(rospy.get_param("~search_z", 0.1))
        # Box-phase heading: face the box FRONTS. Boxes now sit in a row at
        # y=2.425 facing +y, scanned from (0.725, 5) -> face map -y (yaw=-pi/2).
        # Forcing the heading makes the body frame match the layout, so
        # _step_body(forward)=toward boxes (-y) and _step_body(left/right)=along
        # the row (+-x). Also kills the gate-exit yaw skew.
        self.box_yaw = float(rospy.get_param("~box_yaw_rad", -math.pi / 2))
        # added to every absolute z target (scan/prep/land) from the gate-3
        # anchor's measured EKF-z drift, so real altitude matches intent.
        self.z_bias_enable = bool(rospy.get_param("~z_bias_enable", True))
        self.z_bias = 0.0
        self.search_step = float(rospy.get_param("~search_step_m", 0.3))
        self.search_max = float(rospy.get_param("~search_max_m", 1.0))  # +-1 m
        self.sweep_cycles = int(rospy.get_param("~sweep_cycles", 2))
        self.step_period = float(rospy.get_param("~step_period_s", 1.6))
        self.k_px = float(rospy.get_param("~k_px", 0.0015))   # m per pixel err
        self.max_servo_step = float(rospy.get_param("~max_servo_step_m", 0.30))
        self.fwd_step = float(rospy.get_param("~fwd_step_m", 0.25))
        self.near_frac = float(rospy.get_param("~near_frac", 0.35))
        # YOLO already filters at conf>=0.8 (yolo node default); we keep a floor
        # here too, AND require N consecutive frames so a momentary mis-detect
        # is never acted on.
        # 0.65 (was 0.8): the hamburger label rarely cleared 0.8 -> the target
        # was effectively invisible. The windowed N-hit confirm + AR gate still
        # reject momentary false positives at this threshold.
        self.conf_min = float(rospy.get_param("~conf_min", 0.65))
        self.confirm_n = int(rospy.get_param("~confirm_n", 4))
        self.confirm_window_s = float(rospy.get_param("~confirm_window_s", 1.5))
        # APPROACH gives up and returns to SEARCH after this long without the
        # target (recovery from a stale confirm / weak label).
        self.approach_lost_s = float(rospy.get_param("~approach_lost_s", 6.0))
        # if the target drops out during APPROACH but the last seen offset was
        # within this, treat as aligned-enough and go MOUNT (down cam refines).
        self.rough_align_px = float(rospy.get_param("~rough_align_px", 70.0))
        self._last_err_px = None

        # ---- mount / top-centering / landing ----
        # climb only to ~0.8 m for the down cam: higher (was 1.2 m) put the
        # front IR over open low-texture space -> feature loss -> VIO diverged
        # -> CENTER_TOP lost the box. 0.8 m still clears the 26 cm box top.
        self.climb_dz = float(rospy.get_param("~climb_dz", 0.1))
        # MOUNT creeps forward (after climbing) until the down cam sees the box.
        # From the align spot (~gate+advance) to the boxes can be ~3 m, and the
        # step is SLOW so it doesn't overshoot the box top.
        self.mount_fwd_max = float(rospy.get_param("~mount_fwd_max_m", 3.5))
        self.mount_fwd_step = float(rospy.get_param("~mount_fwd_step_m", 0.15))

        # ---- planner-based box tour (map frame) ----
        # Sweep the box y-positions at a fixed x via the EGO planner; the front
        # cam scans. Target seen -> direct YOLO align -> planner ascends -> goes
        # -x until the DOWN cam catches the box top -> direct landing.
        self.tour_x = float(rospy.get_param("~tour_x", 1.5))
        self.box_ys = [float(y) for y in rospy.get_param(
            "~box_ys", [5.45, 6.72, 7.45, 8.95])]
        # LEFT gate (strafe_sign>0) tours low->high y; RIGHT gate high->low.
        self.tour_flip = bool(rospy.get_param("~tour_flip", False))
        self.forward_max = float(rospy.get_param("~forward_max_m", 3.0))
        self.arrive_tol = float(rospy.get_param("~arrive_tol_m", 0.25))
        self._tour = []
        self._ti = 0
        self._aligned = None
        self._fwd_start = None
        self._yaw = 0.0
        self._goal_sent = None
        self.cam_back_m = float(rospy.get_param("~down_cam_back_m", 0.14))
        # down cam focal length (px) for the 424-wide colour stream (HFOV ~69
        # deg -> 212/tan(34.5)=~308). Used to turn the box's pixel offset into
        # METRES accurately at CENTER_TOP (k_px under-shot).
        self.down_focal_px = float(rospy.get_param("~down_focal_px", 308.0))
        self.center_tol_px = float(rospy.get_param("~center_tol_px", 25.0))
        self.stable_n = int(rospy.get_param("~stable_n", 5))
        # CENTER_TOP: collect this many down-cam detections, average the box
        # offset (trust it), then move + land in ONE shot -- continuous servo
        # oscillated and the long hover let the VIO diverge.
        self.ct_n = int(rospy.get_param("~center_top_n", 8))
        # FACE-ON gate: the same picture is on the box front + top. The facing
        # camera sees the label at ~label_ar; the other face (grazing) looks
        # FLAT (very different AR). We only reject CLEARLY-WRONG ARs so the box
        # is still caught from a fair angle -- max_ar is the cutoff (a label
        # this flat must be a grazing face). label_ar is the true print ratio.
        self.label_ar = float(rospy.get_param("~label_ar", 26.0 / 19.0))
        self.label_ar_tol = float(rospy.get_param("~label_ar_tol", 1.2))
        self.max_ar = float(rospy.get_param("~max_ar", 2.6))
        # down image axes (this mount): image BOTTOM (+v) = drone FRONT; and
        # down cam REMOUNTED (rotated ~180 deg) so it now matches the front cam
        # in BOTH axes: image RIGHT (+u) = drone RIGHT (u_sign=+1), and image
        # BOTTOM (+v) = drone BACK now, so v_sign flips to -1. (Flip both back
        # if remounted again; verify on the bench with yolo_check.)
        self.v_sign = float(rospy.get_param("~v_sign", -1.0))
        self.u_sign = float(rospy.get_param("~u_sign", 1.0))
        # box is 26 cm tall; rise to PREP altitude, centre over it, then descend
        # to land_z (box top + clearance) where we force-disarm.
        self.box_h = float(rospy.get_param("~box_height_m", 0.26))
        self.prep_z = float(rospy.get_param("~prep_z", 0.8))   # no extra climb
        self.land_clear = float(rospy.get_param("~land_clearance_m", 0.06))
        self.land_z = float(rospy.get_param("~land_z", self.box_h + self.land_clear))
        self.descend_step = float(rospy.get_param("~descend_step_m", 0.12))
        # GENTLE landing: slow final descent + press a touch onto the box; no
        # force-disarm (operator cuts throttle).
        self.land_descend_step = float(
            rospy.get_param("~land_descend_step_m", 0.06))
        # motor cutoff fires only when the drone is within this of the target
        # xy at the cut height (box top + 5 cm).
        self.land_xy_tol = float(rospy.get_param("~land_xy_tol_m", 0.1))
        # forward trim applied ONLY to the final landing move (it consistently
        # landed ~10 cm short of the box centre).
        self.land_fwd_trim = float(rospy.get_param("~land_fwd_trim_m", 0.10))

        g = self.left_gate if self.gate_choice == "left" else self.right_gate
        self.gate = list(g)
        # passed LEFT gate (facing -x) -> unseen boxes are to the drone's
        # RIGHT = map +y; passed RIGHT gate -> map -y.
        self.strafe_sign = (1.0 if self.gate_choice == "left" else -1.0) \
            * float(rospy.get_param("~strafe_sign_mult", 1.0))

        # ---- start mode ----
        # auto=True  (s5 standalone): PRE_TAKEOFF -> fly through the gate myself.
        # auto=False (final.launch):  IDLE until /precision_land/start (final_
        #            mission already flew the gate); the message carries the
        #            target label + strafe sign, then I jump straight to SEARCH.
        self.auto = bool(rospy.get_param("~auto", True))

        # ---- state ----
        self.state = "PRE_TAKEOFF" if self.auto else "IDLE"
        self.armed = False
        self.pose = None
        self.front_det = None      # latest (det, stamp) for target label
        self.down_det = None
        self._front_hits = []      # timestamps of recent target detections
        self._down_hits = []
        self.front_shape = (424, 240)
        self.down_shape = (424, 240)
        self._t_step = rospy.Time(0)
        self._adv_dist = 0.0
        self._search_dist = 0.0
        self._sweep_off = 0.0    # signed lateral offset during the SEARCH sweep
        self._sweep_phase = 0    # 0:+max 1:centre 2:-max 3:centre (per cycle)
        self._mount_fwd = 0.0
        self._stable = 0
        self._anchor = None        # latched cruise/servo reference pose
        self._committed = []       # queued blind steps for COMMIT
        self._ct_samples = []      # CENTER_TOP box-offset samples

        self.pub_cmd = rospy.Publisher("/offboard/cmd_raw", PositionTarget,
                                       queue_size=2)
        self.pub_goal = rospy.Publisher("/next_goal", PoseStamped, queue_size=2)
        self.pub_force = rospy.Publisher("/delivery/force", Bool, queue_size=1,
                                         latch=True)
        self.pub_state = rospy.Publisher("/precision_land/state", String,
                                         queue_size=1, latch=True)

        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)
        rospy.Subscriber("/mavros/state", MavState, self.on_mav_state,
                         queue_size=5)
        rospy.Subscriber("/delivery/yolo_front/detections", String,
                         self.on_front, queue_size=2)
        rospy.Subscriber("/delivery/yolo/detections", String,
                         self.on_down, queue_size=2)
        if not self.auto:
            rospy.Subscriber("/precision_land/start", String, self.on_start,
                             queue_size=1)
        rospy.Timer(rospy.Duration(0.2), self.tick)
        rospy.loginfo("[pland] target='%s' gate=%s centre=(%.1f,%.1f) "
                      "strafe=%+.0fy", self.target_label, self.gate_choice,
                      self.gate[0], self.gate[1], self.strafe_sign)

    # ---------------- subscribers ----------------
    def on_pose(self, msg):
        with self.lock:
            self.pose = msg

    def on_mav_state(self, msg):
        self.armed = bool(msg.armed)

    def on_start(self, msg):
        """final_mission handoff after the last gate. JSON: {target, strafe_sign}.
        Force the delivery RGB/YOLO on, latch the current pose, jump to SEARCH."""
        if self.state != "IDLE":
            return
        try:
            d = json.loads(msg.data)
            if d.get("target"):
                self.target_label = d["target"]
            if "strafe_sign" in d:
                self.strafe_sign = float(d["strafe_sign"])
            if self.z_bias_enable:
                # measured EKF-z drift from the gate-3 anchor. ADD it to every
                # absolute z TARGET so the REAL altitude matches the intent
                # (EKF reads z_bias too high -> command z_bias higher).
                self.z_bias = float(d.get("z_bias", 0.0))
        except ValueError:
            pass
        rospy.loginfo("[pland] external START: target='%s' strafe=%+.0f "
                      "z_bias=%+.2f", self.target_label, self.strafe_sign,
                      self.z_bias)
        self.pub_force.publish(Bool(data=True))
        self._latch_anchor()
        self._anchor["yaw"] = self.box_yaw    # face exactly -x for the box phase
        self._goto_state("ADVANCE")

    def _pick(self, msg):
        """Pick the target-label box, rejecting only CLEARLY grazing faces.

        Same picture on the box FRONT + TOP. The facing camera sees ~label_ar;
        the other face (grazing) looks FLAT. We reject ONLY clearly-wrong ARs
        (|AR-label_ar| > label_ar_tol OR AR > max_ar) so the box is still caught
        from a fair side angle -- then take the LARGEST (nearest / below) box.
        (Momentary mis-detects are filtered by the streak in on_front/on_down.)
        """
        try:
            d = json.loads(msg.data)
        except ValueError:
            return None, None
        shape = (d.get("w", 424), d.get("h", 240))
        cand = []
        for det in d.get("dets", []):
            if det["cls"] != self.target_label or det["conf"] < self.conf_min:
                continue
            x1, y1, x2, y2 = det["xyxy"]
            ar = max(x2 - x1, y2 - y1) / (min(x2 - x1, y2 - y1) + 1e-6)
            if ar > self.max_ar or abs(ar - self.label_ar) > self.label_ar_tol:
                continue                       # clearly flat -> grazing face
            cand.append(det)
        if not cand:
            return None, shape
        best = max(cand, key=lambda det: (det["xyxy"][2] - det["xyxy"][0])
                                         * (det["xyxy"][3] - det["xyxy"][1]))
        return best, shape

    def on_front(self, msg):
        det, shape = self._pick(msg)
        if shape:
            self.front_shape = shape
        # TEMPORAL FILTER (windowed): the target is real if seen confirm_n times
        # within the last confirm_window_s. STRICT consecutive frames never
        # confirmed when several boxes are in view (target dropped out every few
        # frames and reset the streak) -- windowed counting tolerates that while
        # still rejecting a 1-2 frame mis-detection.
        if det:
            self._front_hits.append(rospy.Time.now())
            self.front_det = (det, rospy.Time.now())

    def on_down(self, msg):
        det, shape = self._pick(msg)
        if shape:
            self.down_shape = shape
        if det:
            self._down_hits.append(rospy.Time.now())
            self.down_det = (det, rospy.Time.now())

    def _hits_in_window(self, hits):
        now = rospy.Time.now()
        while hits and (now - hits[0]).to_sec() > self.confirm_window_s:
            hits.pop(0)
        return len(hits)

    def _front_ok(self):
        return (self._hits_in_window(self._front_hits) >= self.confirm_n
                and self._fresh(self.front_det))

    def _down_ok(self):
        return (self._hits_in_window(self._down_hits) >= self.confirm_n
                and self._fresh(self.down_det))

    # ---------------- motion helpers ----------------
    def _cmd_pos(self, x, y, z, yaw):
        t = PositionTarget()
        t.header.stamp = rospy.Time.now()
        t.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        t.position.x, t.position.y, t.position.z = x, y, z
        t.yaw = yaw
        self.pub_cmd.publish(t)

    def _step_body(self, fwd, left, dz=0.0):
        """Step the cmd reference by (fwd,left) in the BODY frame + dz."""
        a = self._anchor
        yaw = a["yaw"]
        a["x"] += math.cos(yaw) * fwd - math.sin(yaw) * left
        a["y"] += math.sin(yaw) * fwd + math.cos(yaw) * left
        a["z"] += dz
        self._cmd_pos(a["x"], a["y"], a["z"], yaw)

    def _step_map(self, dx, dy, dz=0.0):
        """Step the cmd reference in the MAP frame (yaw-INDEPENDENT). The boxes
        are laid out in map x/y, so the sweep (+-y) and the forward creep (-x)
        must follow map axes -- a small yaw error otherwise skews the body-frame
        path diagonally (sweep drifts in x, forward drifts in y)."""
        a = self._anchor
        a["x"] += dx
        a["y"] += dy
        a["z"] += dz
        self._cmd_pos(a["x"], a["y"], a["z"], a["yaw"])

    def _latch_anchor(self):
        with self.lock:
            p = self.pose
        self._anchor = {"x": p.pose.position.x, "y": p.pose.position.y,
                        "z": self.hover_z, "yaw": _yaw_of(p.pose.orientation)}

    def _latch_here(self):
        """Latch the servo anchor at the CURRENT pose (incl. z) -- used when
        handing over from the planner to direct cmd_raw control."""
        with self.lock:
            p = self.pose
        self._anchor = {"x": p.pose.position.x, "y": p.pose.position.y,
                        "z": p.pose.position.z, "yaw": _yaw_of(p.pose.orientation)}

    def _goto_xyz(self, x, y, z):
        """Fly to an ABSOLUTE position via the offboard velocity-PID (direct
        cmd_raw). We drive the tour/forward waypoints ourselves instead of the
        EGO planner -- EGO crashed (Eigen assert) on near-zero goals and its
        mandatory-stop doesn't recover, so the planner handoff was too fragile.
        The offboard PID already flies smoothly to a position setpoint."""
        self._cmd_pos(x, y, z, self._yaw)

    def _dist_xy(self, p, x, y):
        return math.hypot(p.pose.position.x - x, p.pose.position.y - y)

    def _build_tour(self):
        ys = sorted(self.box_ys)              # low -> high  (LEFT gate default)
        if self.strafe_sign < 0:              # RIGHT gate -> high -> low
            ys.reverse()
        if self.tour_flip:
            ys.reverse()
        self._tour = [(self.tour_x, y) for y in ys]
        self._ti = 0
        rospy.loginfo("[pland] box tour x=%.1f, y order: %s",
                      self.tour_x, [round(y, 2) for y in ys])

    def _due(self):
        if (rospy.Time.now() - self._t_step).to_sec() >= self.step_period:
            self._t_step = rospy.Time.now()
            return True
        return False

    def _fresh(self, slot, max_age=1.0):
        return (slot is not None
                and (rospy.Time.now() - slot[1]).to_sec() <= max_age)

    def _goto_state(self, s):
        rospy.loginfo("[pland] -> %s", s)
        self.state = s
        self.pub_state.publish(String(data=s))

    # ---------------- state machine ----------------
    def tick(self, _evt):
        with self.lock:
            p = self.pose

        if self.state == "PRE_TAKEOFF":
            if (self.armed and p is not None
                    and p.pose.position.z >= self.hover_z - 0.15):
                g = PoseStamped()
                g.header.frame_id = "map"
                g.pose.position.x = self.gate[0] - self.pass_through  # -x pass
                g.pose.position.y = self.gate[1]
                g.pose.position.z = self.hover_z
                g.pose.orientation.w = 1.0
                self.pub_goal.publish(g)
                rospy.loginfo("[pland] takeoff OK -> through %s gate to "
                              "(%.2f, %.2f)", self.gate_choice,
                              g.pose.position.x, g.pose.position.y)
                self._goto_state("TO_GATE")

        elif self.state == "TO_GATE":
            if p is not None and \
                    p.pose.position.x < self.gate[0] - self.pass_margin:
                rospy.loginfo("[pland] gate passed (x=%.2f)",
                              p.pose.position.x)
                self.pub_force.publish(Bool(data=True))   # RGB + YOLO ON
                self._latch_anchor()
                self._anchor["yaw"] = self.box_yaw    # face exactly -x
                self._goto_state("ADVANCE")

        elif self.state == "ADVANCE":
            # Clear the gate by advance_m, then DESCEND to scan_z (low, so the
            # up-tilted front cam catches the low boxes), then sweep.
            a = self._anchor
            if self._adv_dist < self.advance_m:          # 1) short clearance
                if self._due():
                    self._step_body(self.fwd_step, 0.0)   # forward = box_yaw dir
                    self._adv_dist += self.fwd_step
                return
            scan_tgt = self.search_z + self.z_bias       # drift-compensated
            if a["z"] > scan_tgt + 0.05:                 # 2) descend to scan_z
                if self._due():
                    self._step_body(0.0, 0.0,
                                    max(-self.descend_step, scan_tgt - a["z"]))
                return
            rospy.loginfo("[pland] scan pose ready (advanced %.1f m, z=%.2f) "
                          "-> SEARCH", self._adv_dist, a["z"])
            self._sweep_off, self._sweep_phase = 0.0, 0
            self._goto_state("SEARCH")

        elif self.state == "SEARCH":
            if self._front_ok():
                rospy.loginfo("[pland] '%s' confirmed by front cam "
                              "(%d hits in %.1fs)", self.target_label,
                              self._hits_in_window(self._front_hits),
                              self.confirm_window_s)
                self._goto_state("APPROACH")
                return
            # ANY recent target hit -> PAUSE the sweep and wait for the confirm
            # window to fill. Sweeping on after the first hit moved the box out
            # of frame, so it was only ever seen once. If no further hit comes
            # (mis-detection), freshness expires and the sweep resumes.
            if self._fresh(self.front_det, self.confirm_window_s):
                rospy.loginfo_throttle(2.0, "[pland] '%s' glimpsed -- pausing "
                                       "sweep to confirm (%d/%d hits)",
                                       self.target_label,
                                       self._hits_in_window(self._front_hits),
                                       self.confirm_n)
                return
            if self._due():
                # sweep AROUND the scan centre: +max -> 0 -> -max -> 0, returning
                # to centre between sides. Phase-target based so it is HARD
                # bounded to +-search_max (the old reverse-at-threshold ran past).
                if self._sweep_phase >= 4 * self.sweep_cycles:
                    rospy.logwarn_throttle(5.0, "[pland] swept +-%.1f m x%d, no "
                                           "'%s' -- hovering", self.search_max,
                                           self.sweep_cycles, self.target_label)
                    return
                tgt = (self.search_max, 0.0, -self.search_max, 0.0)[
                    self._sweep_phase % 4]
                if abs(tgt - self._sweep_off) <= self.search_step + 1e-6:
                    self._sweep_off = tgt          # snap, advance to next phase
                    self._sweep_phase += 1
                    return
                d = self.search_step if tgt > self._sweep_off \
                    else -self.search_step
                self._step_body(0.0, -self.strafe_sign * d)
                self._sweep_off += d

        elif self.state == "APPROACH":
            # ALIGN ONLY -- no forward (front cam tilted UP -> advancing drops
            # the low box out of frame). Centre L/R, then climb + down cam.
            if not self._due():
                return
            # SEARCH already confirmed the target (N consecutive). Here, align on
            # ANY recent detection (within 2 s) -- requiring the full streak made
            # APPROACH just hold forever when detection is intermittent.
            if not self._fresh(self.front_det, 2.0):
                # RECOVERY: if the target stays unseen, go back to SEARCH and
                # resume the sweep -- holding forever just hovered all day.
                lost_s = (rospy.Time.now() - self.front_det[1]).to_sec() \
                    if self.front_det else 1e9
                if lost_s > self.approach_lost_s:
                    # If we were already ROUGHLY aligned when the label dropped
                    # out (it weakens up close), proceed to MOUNT -- the down
                    # cam re-centres precisely in CENTER_TOP anyway. Only
                    # re-scan when we were still far off.
                    if self._last_err_px is not None and \
                            abs(self._last_err_px) <= self.rough_align_px:
                        rospy.logwarn("[pland] target lost but last err %.0fpx "
                                      "(rough-aligned) -> MOUNT",
                                      self._last_err_px)
                        self._goto_state("MOUNT")
                        return
                    rospy.logwarn("[pland] target unseen %.0fs in APPROACH -> "
                                  "re-scan", lost_s)
                    self._sweep_off, self._sweep_phase = 0.0, 0
                    self._adv_dist = self.advance_m   # skip forward, just z
                    # latch at the CURRENT pose: _latch_anchor would latch
                    # z=hover_z and climb to 0.7 (that was the surprise ascent).
                    # ADVANCE then re-descends to the scan altitude.
                    self._latch_here()
                    self._anchor["yaw"] = self.box_yaw
                    self._goto_state("ADVANCE")
                    return
                rospy.logwarn_throttle(2.0, "[pland] no recent front det -- hold")
                return
            det, _ = self.front_det
            w, _h = self.front_shape
            x1, _y1, x2, _y2 = det["xyxy"]
            err_px = 0.5 * (x1 + x2) - 0.5 * w
            self._last_err_px = err_px
            if abs(err_px) <= self.center_tol_px:
                rospy.loginfo("[pland] aligned L/R (no forward) -> MOUNT")
                self._goto_state("MOUNT")
                return
            side = max(-self.max_servo_step,
                       min(self.max_servo_step, -self.k_px * err_px))
            self._step_body(0.0, side)                 # centre L/R (body frame)

        elif self.state == "MOUNT":
            if not self._due():
                return
            a = self._anchor
            # 1) climb (drift-compensated) so the down cam can look down.
            climb_to = self.hover_z + self.climb_dz + self.z_bias
            if a["z"] < climb_to - 0.05:
                self._step_body(0.0, 0.0, min(0.25, climb_to - a["z"]))
                return
            # 2) creep forward SLOWLY until the down cam sees the box top.
            if self._down_ok():
                rospy.loginfo("[pland] box top in down cam -> CENTER_TOP")
                self._stable = 0
                self._ct_samples = []
                self._goto_state("CENTER_TOP")
                return
            if self._mount_fwd >= self.mount_fwd_max:
                rospy.logwarn_throttle(5.0, "[pland] mount sweep exhausted -- "
                                       "hovering")
                return
            self._step_body(self.mount_fwd_step, 0.0)   # forward = box_yaw dir
            self._mount_fwd += self.mount_fwd_step

        elif self.state == "CENTER_TOP":
            # COLLECT ct_n down-cam detections, AVERAGE the box offset (trust
            # it), then move the BODY centre over the box in ONE shot and land.
            # No continuous servo (it oscillated; the long hover diverged VIO).
            if not self._fresh(self.down_det, 2.0):
                rospy.logwarn_throttle(2.0, "[pland] top not seen -- hold")
                return
            det, _ = self.down_det
            w, h = self.down_shape
            x1, y1, x2, y2 = det["xyxy"]
            du = 0.5 * (x1 + x2) - 0.5 * w   # +: box right in image
            dv = 0.5 * (y1 + y2) - 0.5 * h   # +: box toward image bottom
            self._ct_samples.append((dv, du))
            if len(self._ct_samples) < self.ct_n:
                return
            adv = sum(s[0] for s in self._ct_samples) / len(self._ct_samples)
            adu = sum(s[1] for s in self._ct_samples) / len(self._ct_samples)
            a = self._anchor
            # TRUE metres-per-pixel at the box plane = (cam altitude - box top)
            # / focal. k_px (a servo gain) under-shot, so it landed slightly off.
            mpp = max(0.05, a["z"] - self.box_h) / self.down_focal_px
            # v_sign: +dv = box toward image-bottom; u_sign: +du = box image-
            # right. Add -cam_back so the BODY (not the cam) ends over the box.
            # land_fwd_trim: measured landing bias -- it consistently touched
            # down ~10 cm SHORT, so push the final spot that much forward.
            fwd = self.v_sign * mpp * adv - self.cam_back_m + self.land_fwd_trim
            left = -self.u_sign * mpp * adu
            rospy.loginfo("[pland] top converged (%d dets, du=%.0f dv=%.0f px) "
                          "-> move body (fwd=%.2f,left=%.2f) + land",
                          len(self._ct_samples), adu, adv, fwd, left)
            self._committed = [("body", fwd, left, 0.0)]   # one move, then descend
            self._goto_state("COMMIT")

        elif self.state == "COMMIT":
            # detection no longer required from here (too close for YOLO).
            if not self._due():
                return
            if self._committed:
                kind, f, l, d = self._committed.pop(0)
                self._step_body(f, l, d)
                return
            a = self._anchor
            # Descend slowly to 5 cm above the box top, then CUT MOTORS -- but
            # only once the drone is ACTUALLY on the target xy (within
            # land_xy_tol of the commanded spot). Cutting while still sliding
            # sideways would drop it off the box.
            cut_z = self.box_h + 0.05 + self.z_bias
            if a["z"] > cut_z:
                self._step_body(0.0, 0.0, -self.land_descend_step)   # slow
                return
            with self.lock:
                p = self.pose
            xy_err = math.hypot(p.pose.position.x - a["x"],
                                p.pose.position.y - a["y"]) if p else 1e9
            if xy_err > self.land_xy_tol:
                rospy.loginfo_throttle(2.0, "[pland] at cut height, waiting "
                                       "xy (err %.2f m > %.2f)", xy_err,
                                       self.land_xy_tol)
                self._cmd_pos(a["x"], a["y"], cut_z, a["yaw"])   # hold + converge
                return
            rospy.loginfo("[pland] xy on target (err %.2f m), z=box+5cm -> "
                          "MOTOR CUTOFF", xy_err)
            try:
                rospy.wait_for_service("/mavros/cmd/command", timeout=2.0)
                cmd = rospy.ServiceProxy("/mavros/cmd/command", CommandLong)
                # MAV_CMD_COMPONENT_ARM_DISARM (400), param1=0 (disarm),
                # param2=21196 (force, in-air allowed)
                cmd(command=400, param1=0.0, param2=21196.0)
            except Exception as e:
                rospy.logerr("[pland] disarm failed: %s", e)
                return
            self._goto_state("LANDED")

        # LANDED: nothing to do.


if __name__ == "__main__":
    try:
        PrecisionLand()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
