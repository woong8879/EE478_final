#!/usr/bin/env python3
"""final_mission_node.py — full EE478 final pipeline orchestrator.

Sequence (map FLU, +x fwd / +y left, origin = takeoff spot):

  PRE_TAKEOFF  wait armed + at altitude.
  For each of the 3 gate pairs (a monitor sits between the two openings of a
  pair; solve it with ChatGPT vision to pick LEFT or RIGHT):
    NAV    fly the gate's pre-waypoints via /next_goal (EGO avoids obstacles);
           the last leg orients the drone to face the gate so the front camera
           sees the monitor and "left/right" is the drone's own left/right.
    QUIZ   hover at the quiz point, photograph /camera/infra1, ask gpt-4o-mini
           (2-step: transcribe -> reason) LEFT or RIGHT. Hover while waiting.
    THROUGH go to a goal BEYOND the chosen opening (so the path passes THROUGH,
           not stops in it); refine the centre live from the two gate-tag
           midpoint (/apriltag/detections); confirmed passed by crossing the
           wall + pass_margin along the gate's pass axis.
  After the LAST gate -> hand off to precision_land_node (/precision_land/start
  with the chosen strafe sign + target label): it forces RGB+YOLO on, searches
  the target box, approaches, mounts and lands on top.

Gate geometry derived from the surveyed tag coords:
  Gate1 wall x=3.5, cross +x:  L=(3.5, 1.0) 281/282 | R=(3.5,-1.0) 283/284
  Gate2 wall y=3.5, cross +y:  L=(8.9,3.5) 266/267  | R=(10.9,3.5) 268/269
  Gate3 wall x=3.5, cross -x:  L=(3.5,6.2) 271/274  | R=(3.5,8.2) 275/276
  (L/R = the drone's left/right given the approach heading of that gate.)
"""
import base64
import json
import math
import threading

import cv2
import requests
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State as MavState
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


def _yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# Each gate: pre_wps (last = quiz point), pass axis/sign, the two openings
# (centre + tag pair) as the drone's LEFT / RIGHT.
# quiz point = the monitor (exact midpoint of the pair's two openings),
# photographed from 1.5 m IN FRONT of the gate wall along the approach axis
# (1 m was too close -- nearly clipped the gate-2 monitor).
GATES = [
    {"name": "gate1",      # wall x=3.5; quiz 1.7 m before (20 cm farther back
     # than the others: shallower diagonal to the RIGHT opening -> more
     # clearance from the centre pillar)
     "pre_wps": [[1.8, 0.0]], "quiz": [1.8, 0.0],
     "axis": "x", "sign": +1.0,
     "left": {"c": [3.5, 1.0], "tags": [281, 282]},
     "right": {"c": [3.5, -1.0], "tags": [283, 284]}},
    {"name": "gate2",      # wall y=3.5, cross +y -> quiz 1.5 m before = y 2.0
     "pre_wps": [[9.9, 0.0], [9.9, 2.0]], "quiz": [9.9, 2.0],
     "axis": "y", "sign": +1.0,
     "left": {"c": [8.9, 3.5], "tags": [266, 267]},
     "right": {"c": [10.9, 3.5], "tags": [268, 269]}},
    {"name": "gate3",      # wall x=3.5, cross -x -> quiz 1.5 m before = x 5.0
     # pass_through 1.5: a DESK sits between/behind the two openings, so go
     # 1.5 m straight from the opening centre (x=2.0) BEFORE turning left
     # toward the box scan position (cutting the corner clips the desk).
     "pre_wps": [[9.9, 6.5], [9.9, 7.2], [5.0, 7.2]], "quiz": [5.0, 7.2],
     "pass_through": 1.5,
     "axis": "x", "sign": -1.0,
     "left": {"c": [3.5, 6.2], "tags": [271, 274]},
     "right": {"c": [3.5, 8.2], "tags": [275, 276]}},
]


class FinalMission(object):
    def __init__(self):
        rospy.init_node("final_mission")
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        self.hover_z = float(rospy.get_param("~hover_z", 0.7))
        self.target = rospy.get_param("~target", "cafe")
        self.arrive_r = float(rospy.get_param("~arrive_radius_m", 0.5))
        # the GATE waypoint (= quiz point, the crossing start) is held TIGHT so
        # the through-diagonal starts from a precise spot; intermediate
        # waypoints keep the loose radius (no need to be exact there).
        self.gate_arrive_r = float(rospy.get_param("~gate_arrive_radius_m", 0.2))
        # Gate crossing goes via a LINE-UP point square in front of the chosen
        # opening, with a TIGHT arrival tolerance, then straight through the
        # middle -- the direct quiz-point->through diagonal skimmed the centre
        # pillar (~0.26 m clearance).
        # crossing goes VIA a point exactly this far in front of the (tag-
        # refined) opening centre, then straight through -- the direct
        # quiz->through diagonal skimmed the centre pillar when VIO drift
        # pushed it sideways. Tag refine keeps this point drift-proof.
        self.lineup_back = float(rospy.get_param("~lineup_back_m", 0.5))
        self.lineup_tol = float(rospy.get_param("~lineup_tol_m", 0.2))
        # After the last gate, the planner flies to this spot IN FRONT of the
        # box row (boxes at y=2.425 facing +y) before handing off to the box
        # scan/land. Centre of the 4 box x-positions = 0.725.
        self.box_scan_pos = [float(v) for v in
                             rospy.get_param("~box_scan_pos", [0.725, 5.0])]
        self.box_arrive_tol = float(rospy.get_param("~box_arrive_tol_m", 0.3))
        self.pass_through = float(rospy.get_param("~pass_through_m", 1.2))
        self.pass_margin = float(rospy.get_param("~pass_margin_m", 0.8))
        self.settle_s = float(rospy.get_param("~quiz_settle_s", 3.0))
        self.refine_min = float(rospy.get_param("~refine_min_step_m", 0.15))
        self.refine_max = float(rospy.get_param("~refine_max_shift_m", 0.8))
        self.api_key = rospy.get_param("~api_key", "")
        # Most accurate available: gpt-4.1 for BOTH vision/OCR and reasoning.
        # Later knowledge cutoff (~mid-2024) so it knows recent facts (e.g.
        # ICRA 2026 = Vienna, not Atlanta=2025) AND reads the monitor well.
        self.model = rospy.get_param("~model", "gpt-4.1")
        self.reason_model = rospy.get_param("~reason_model", self.model)
        self.retry_s = float(rospy.get_param("~retry_s", 5.0))

        # ~gates: override the gate list (e.g. a single gate in a LOCAL frame
        # for the "take off in front of the last gate" test). Default = the full
        # 3-gate course. ~start_gate (1-based) picks the first gate to run.
        self.gates = rospy.get_param("~gates", GATES)
        self.gi = max(0, int(rospy.get_param("~start_gate", 1)) - 1)
        self.gi = min(self.gi, len(self.gates) - 1)
        self.wi = 0                # current pre-waypoint index
        self.state = "PRE_TAKEOFF"
        self.armed = False
        self.pose = None
        self.last_img = None             # LEFT IR
        self.last_img2 = None            # RIGHT IR
        self.gate = None           # chosen opening dict for the active gate
        self.gate_tags = None
        self.answer = None
        self._arrive_t = None
        self._asking = False
        self._last_ask_t = rospy.Time(0)
        self._last_refine_t = rospy.Time(0)
        self._goal_sent = None     # last (x,y) goal published (dedupe)

        self.pub_goal = rospy.Publisher("/next_goal", PoseStamped, queue_size=2)
        self.pub_force = rospy.Publisher("/delivery/force", Bool,
                                         queue_size=1, latch=True)
        self.pub_start = rospy.Publisher("/precision_land/start", String,
                                         queue_size=1, latch=True)
        self.pub_state = rospy.Publisher("/final_mission/state", String,
                                         queue_size=1, latch=True)

        # ---- one-shot XYZ drift anchor off the final gate's tags ----
        # When >=anchor_min_tags of these tags are seen within anchor_range and
        # their back-computed drone positions agree (<anchor_spread), publish a
        # SINGLE /landmark_anchor_pose. svo_vision_relay / vio_bridge blend it
        # into vision_pose at 0.2 m/s -> drift accumulated between gates 2-3 is
        # corrected once, smoothly (no continuous EKF tug = stays stable).
        self.pub_anchor = rospy.Publisher("/landmark_anchor_pose", PoseStamped,
                                          queue_size=1)
        self.anchor_enable = bool(rospy.get_param("~anchor_enable", True))
        self.anchor_range = float(rospy.get_param("~anchor_range_m", 2.0))
        self.anchor_spread = float(rospy.get_param("~anchor_spread_m", 0.3))
        self.anchor_min_tags = int(rospy.get_param("~anchor_min_tags", 2))
        self._z_bias = 0.0       # measured EKF-z drift, handed to precision_land
        # tag world positions (map FLU, from apriltag_anchors.yaml).
        self.anchor_tag_world = {281: (3.5, 1.7, 1.45), 282: (3.5, 0.3, 1.45),
                                 283: (3.5, -0.3, 1.45), 284: (3.5, -1.7, 1.45),
                                 271: (3.5, 5.5, 1.45), 274: (3.5, 6.9, 1.45),
                                 275: (3.5, 7.5, 1.45), 276: (3.5, 8.9, 1.45)}
        at = rospy.get_param("~anchor_tags", None)
        if isinstance(at, dict):
            self.anchor_tag_world = {int(k): tuple(v) for k, v in at.items()}
        # One-shot anchor per PAIR (each fires at most once). Default: only the
        # gate-3 central pair 274+275 (head-on from the quiz line = least
        # off-axis solvePnP noise) -> pre-landing drift fix + z_bias measure.
        # Gate crossing drift is handled geometrically instead (LINE_UP point
        # 0.5 m in front of the tag-refined opening centre).
        self.anchor_pairs = [frozenset(int(t) for t in pr) for pr in
                             rospy.get_param("~anchor_pairs", [[274, 275]])]
        self._anchored_pairs = set()

        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)
        rospy.Subscriber("/mavros/state", MavState, self.on_mav, queue_size=5)
        rospy.Subscriber("/camera/infra1/image_rect_raw", Image, self.on_img,
                         queue_size=1, buff_size=2 ** 22)        # LEFT IR
        rospy.Subscriber("/camera/infra2/image_rect_raw", Image, self.on_img2,
                         queue_size=1, buff_size=2 ** 22)        # RIGHT IR
        rospy.Subscriber("/apriltag/detections", String, self.on_det,
                         queue_size=5)
        rospy.Timer(rospy.Duration(0.2), self.tick)
        if not self.api_key:
            rospy.logwarn("[final] NO OPENAI_API_KEY -- will hover at each quiz "
                          "point forever!")
        rospy.loginfo("[final] target='%s', 3 gates, %d quiz points",
                      self.target, len(self.gates))

    # ---------------- subscribers ----------------
    def on_pose(self, msg):
        with self.lock:
            self.pose = msg

    def on_mav(self, msg):
        self.armed = bool(msg.armed)

    def on_img(self, msg):
        with self.lock:
            self.last_img = msg          # LEFT IR (infra1)

    def on_img2(self, msg):
        with self.lock:
            self.last_img2 = msg         # RIGHT IR (infra2)

    def _try_anchor(self, seen, p):
        """One-shot XYZ drift anchor off the final gate's tags. Back-compute the
        drone's TRUE world pose from each visible anchor tag (drone_true =
        tag_world - R(yaw)*tag_body); if >=min_tags agree, publish one anchor."""
        if not self.anchor_enable or not self.armed:
            return
        yaw = _yaw_of(p.pose.orientation)
        c, s = math.cos(yaw), math.sin(yaw)
        for pair in self.anchor_pairs:
            if pair in self._anchored_pairs:           # each pair fires ONCE
                continue
            ests = []
            for tid in pair:
                tw = self.anchor_tag_world.get(tid)
                d = seen.get(tid)
                if tw is None or d is None or not d.get("body"):
                    break
                if float(d.get("range", 1e9)) > self.anchor_range:
                    break
                tx, ty, tz = tw
                bx, by, bz = d["body"][0], d["body"][1], d["body"][2]
                ests.append((tx - (c * bx - s * by),   # drone_true x
                             ty - (s * bx + c * by),   # drone_true y
                             tz - bz))                 # drone_true z
            if len(ests) < self.anchor_min_tags:
                continue
            spread = max(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                                   + (a[2] - b[2]) ** 2)
                         for i, a in enumerate(ests) for b in ests[i + 1:])
            if spread > self.anchor_spread:
                rospy.logwarn_throttle(2.0, "[final] anchor pair %s disagree "
                                       "(spread %.2f m) -- waiting",
                                       sorted(pair), spread)
                continue
            n = len(ests)
            ax = sum(e[0] for e in ests) / n
            ay = sum(e[1] for e in ests) / n
            az = sum(e[2] for e in ests) / n
            a = PoseStamped()
            a.header.stamp = rospy.Time.now()
            a.header.frame_id = "map"
            a.pose.position.x, a.pose.position.y, a.pose.position.z = ax, ay, az
            a.pose.orientation = p.pose.orientation   # position-only anchor
            self.pub_anchor.publish(a)
            self._anchored_pairs.add(pair)
            # z is NOT injected into the EKF (svo_vision_relay drops it -- noisy
            # solvePnP z destabilised altitude before), but it IS a one-shot
            # MEASUREMENT of EKF-z drift: z_bias = EKF_z - true_z, handed to
            # precision_land to correct its altitude targets. The LAST anchor
            # (gate-3) overwrites earlier ones -- freshest before landing.
            self._z_bias = max(-0.5, min(0.5, p.pose.position.z - az))
            rospy.loginfo("[final] XYZ ANCHOR (pair %s) -> (%.2f,%.2f,%.2f); "
                          "drift correction (%+.2f,%+.2f,%+.2f), z_bias=%+.2f",
                          sorted(pair), ax, ay, az, ax - p.pose.position.x,
                          ay - p.pose.position.y, az - p.pose.position.z,
                          self._z_bias)
            return

    def on_det(self, msg):
        """One-shot drift anchor + live gate-centre refine during THROUGH."""
        try:
            dets = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            p = self.pose
        if p is None:
            return
        seen = {int(d["id"]): d for d in dets}

        self._try_anchor(seen, p)   # fires once when final-gate tags are close

        if self.state not in ("LINE_UP", "THROUGH") or self.gate_tags is None:
            return
        chosen = [seen[t] for t in self.gate_tags
                  if t in seen and seen[t].get("body")]
        if len(chosen) != 2:
            return
        now = rospy.Time.now()
        if (now - self._last_refine_t).to_sec() < 1.0:
            return
        bx = 0.5 * (chosen[0]["body"][0] + chosen[1]["body"][0])
        by = 0.5 * (chosen[0]["body"][1] + chosen[1]["body"][1])
        yaw = _yaw_of(p.pose.orientation)
        cx = p.pose.position.x + math.cos(yaw) * bx - math.sin(yaw) * by
        cy = p.pose.position.y + math.sin(yaw) * bx + math.cos(yaw) * by
        c = self.gate["c"]
        d = math.hypot(cx - c[0], cy - c[1])
        if d > self.refine_max or d < self.refine_min:
            return
        self.gate["c"] = [cx, cy]
        self._last_refine_t = now
        if self.state == "LINE_UP":
            self._publish_goal(*self._lineup_goal())
        else:
            self._publish_goal(*self._through_goal())
        rospy.loginfo("[final] gate centre refined -> (%.2f,%.2f)", cx, cy)

    # ---------------- helpers ----------------
    def _publish_goal(self, x, y, force=False):
        if not force and self._goal_sent and \
                math.hypot(x - self._goal_sent[0], y - self._goal_sent[1]) < 0.05:
            return
        g = PoseStamped()
        g.header.stamp = rospy.Time.now()
        g.header.frame_id = "map"
        g.pose.position.x, g.pose.position.y = x, y
        g.pose.position.z = self.hover_z
        g.pose.orientation.w = 1.0
        self.pub_goal.publish(g)
        self._goal_sent = (x, y)

    def _near(self, x, y, r=None):
        with self.lock:
            p = self.pose
        if p is None:
            return False
        if r is None:
            r = self.arrive_r
        return math.hypot(p.pose.position.x - x,
                          p.pose.position.y - y) <= r

    def _through_goal(self):
        g = self.gates[self.gi]
        c = self.gate["c"]
        pt = float(g.get("pass_through", self.pass_through))
        if g["axis"] == "x":
            return c[0] + g["sign"] * pt, c[1]
        return c[0], c[1] + g["sign"] * pt

    def _lineup_goal(self):
        """Point square IN FRONT of the chosen opening (lineup_back before the
        wall, on the opening's centreline)."""
        g = self.gates[self.gi]
        c = self.gate["c"]
        if g["axis"] == "x":
            return c[0] - g["sign"] * self.lineup_back, c[1]
        return c[0], c[1] - g["sign"] * self.lineup_back

    def _passed(self):
        g = self.gates[self.gi]
        with self.lock:
            p = self.pose
        if p is None:
            return False
        c = self.gate["c"]
        if g["axis"] == "x":
            d = (p.pose.position.x - c[0]) * g["sign"]
        else:
            d = (p.pose.position.y - c[1]) * g["sign"]
        return d > self.pass_margin

    def _strafe_sign_for_last_gate(self):
        """Box-search strafe after the final gate: chose LEFT opening -> sweep
        the drone's RIGHT (+1 in precision_land's convention); RIGHT -> LEFT."""
        return 1.0 if self.answer == "LEFT" else -1.0

    def _goto(self, s):
        rospy.loginfo("[final] -> %s", s)
        self.state = s
        self.pub_state.publish(String(data="%s gate%d" % (s, self.gi + 1)))

    # ---------------- LLM (2-step, same as gate_quiz) ----------------
    def _enc(self, img):
        """ROS mono Image -> base64 JPEG (None if unavailable/bad)."""
        if img is None:
            return None
        try:
            gray = self.bridge.imgmsg_to_cv2(img, desired_encoding="mono8")
            ok, jpg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                return base64.b64encode(jpg.tobytes()).decode()
        except Exception:
            pass
        return None

    def _grab_frames(self, per_cam=2, gap_s=0.15):
        """Capture per_cam frames from EACH IR camera (left infra1 + right
        infra2), temporally spread -> robust to a single blurry frame AND to a
        bad viewing angle on one eye."""
        b64s = []
        for _ in range(per_cam):
            with self.lock:
                il, ir = self.last_img, self.last_img2
            for img in (il, ir):
                b = self._enc(img)
                if b is not None:
                    b64s.append(b)
            rospy.sleep(gap_s)
        return b64s

    def _ask(self):
        try:
            b64s = self._grab_frames()
            if not b64s:
                return
            url = "https://api.openai.com/v1/chat/completions"
            hdr = {"Authorization": "Bearer " + self.api_key}
            imgs = [{"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + b, "detail": "high"}}
                for b in b64s]
            r1 = requests.post(url, headers=hdr, json={
                "model": self.model, "max_tokens": 400, "messages": [{
                    "role": "user", "content": [
                        {"type": "text", "text":
                         "다음은 같은 모니터를 드론 카메라로 연속 촬영한 %d장의 "
                         "사진이야 (일부는 흔들렸을 수 있음 -- 여러 장을 "
                         "교차검증해서 가장 정확하게 읽어).\n\n"
                         "화면 구조: 맨 위에 '문제'가 있고, 그 아래에 보기 "
                         "여러 개가 '좌우로 나란히' 놓여 있어. 그리고 각 보기 "
                         "'바로 밑(같은 세로 칸)'에 그 보기 전용의 'Go to "
                         "Left' 또는 'Go to Right' 지시문이 따로 적혀 있어. "
                         "즉 보기와 지시문은 세로 칼럼(열)으로 짝지어져 있다.\n\n"
                         "**가장 중요**: 각 'Go to ...' 지시문은 반드시 '그 바로 "
                         "위에 있는 보기'와 짝지어 적어. 절대 한쪽으로 몰지 말고, "
                         "보이는 칼럼을 하나도 빠뜨리지 마(특히 오른쪽 칼럼의 "
                         "'Go to' 를 누락하지 마). 해석하지 말고 보이는 글자만.\n\n"
                         "아래 형식으로, 화면 왼쪽 칼럼부터 순서대로:\n"
                         "QUESTION: <문제 전체>\n"
                         "COL1(왼쪽): 보기=\"<그 칼럼 보기 텍스트>\" | 바로아래="
                         "\"<그 칼럼의 Go to ...>\"\n"
                         "COL2(오른쪽): 보기=\"<...>\" | 바로아래=\"<Go to ...>\"\n"
                         "(보기가 더 많으면 COL3, COL4 ... 계속)"
                         % len(imgs)}] + imgs}]}, timeout=45)
            r1.raise_for_status()
            ocr = r1.json()["choices"][0]["message"]["content"].strip()
            rospy.loginfo("[final] gate%d monitor text:\n%s", self.gi + 1, ocr)
            r2 = requests.post(url, headers=hdr, json={
                "model": self.reason_model, "max_tokens": 400, "messages": [{
                    "role": "user", "content":
                    "드론이 왼쪽/오른쪽 게이트 중 하나를 골라 통과해야 해. "
                    "모니터를 칼럼(보기 + 그 바로 아래 지시문)별로 전사한 "
                    "결과야:\n\n" + ocr +
                    "\n\n규칙:\n"
                    "(0) 문제에 '연도/버전(예: 2026, ICRA 2026)'이 있으면 그 "
                    "특정 연도에 정확히 맞는 답을 골라 -- 비슷한 다른 연도의 "
                    "답과 혼동하지 마 (예: ICRA 2025=Atlanta, 2026=Vienna). "
                    "보기 둘 중 어느 쪽이 그 연도에 맞는지 각각 따져봐.\n"
                    "(1) 문제(QUESTION)를 풀어 정답인 '보기'가 무엇인지 정한다.\n"
                    "(2) 그 정답 보기가 들어 있는 칼럼(COL)을 찾는다.\n"
                    "(3) 바로 그 칼럼의 '바로아래' 지시문을 읽는다. 경로 지시문은 "
                    "항상 '정답 보기 바로 밑 같은 칼럼'에 있다. 다른 칼럼의 "
                    "지시문이나 칼럼의 좌우 위치는 절대 답이 아니다.\n"
                    "(4) 그 지시문이 'Go to Left'면 LEFT, 'Go to Right'면 RIGHT. "
                    "(예: 정답 보기가 왼쪽 칼럼이어도 그 아래가 'Go to Right'면 "
                    "답은 RIGHT, 오른쪽 칼럼이어도 그 아래가 'Go to Left'면 "
                    "LEFT.)\n"
                    "단계별로: 정답 보기 -> 그 보기의 칼럼 -> 그 칼럼 바로아래 "
                    "Go to -> LEFT/RIGHT. 마지막 줄엔 정확히 LEFT 또는 RIGHT "
                    "한 단어만."}]},
                timeout=45)
            r2.raise_for_status()
            txt = r2.json()["choices"][0]["message"]["content"].strip()
            rospy.loginfo("[final] gate%d reasoning:\n%s", self.gi + 1, txt)
            up = txt.upper()
            li, ri = up.rfind("LEFT"), up.rfind("RIGHT")
            if li < 0 and ri < 0:
                rospy.logwarn("[final] no LEFT/RIGHT -- retry")
            else:
                self.answer = "LEFT" if li > ri else "RIGHT"
        except Exception as e:
            rospy.logwarn("[final] LLM failed: %s (retry %.0fs)", e, self.retry_s)
        finally:
            self._asking = False

    # ---------------- state machine ----------------
    def tick(self, _evt):
        with self.lock:
            p = self.pose

        if self.state == "PRE_TAKEOFF":
            if (self.armed and p is not None
                    and p.pose.position.z >= self.hover_z - 0.15):
                rospy.loginfo("[final] takeoff OK -> mission START")
                self._goto("NAV")
            return

        g = self.gates[self.gi]

        if self.state == "NAV":
            wp = g["pre_wps"][self.wi]
            self._publish_goal(wp[0], wp[1])
            last = self.wi >= len(g["pre_wps"]) - 1
            # tight radius ONLY at the gate (quiz) waypoint; loose in between
            if self._near(wp[0], wp[1],
                          self.gate_arrive_r if last else self.arrive_r):
                if not last:
                    self.wi += 1
                else:
                    self._arrive_t = None
                    self.answer = None
                    self._goto("QUIZ")

        elif self.state == "QUIZ":
            # hover at the quiz point (re-assert goal); ask once settled
            qz = g["quiz"]
            self._publish_goal(qz[0], qz[1])
            if self.answer is not None:
                self.gate = dict(g["left" if self.answer == "LEFT" else "right"])
                self.gate_tags = set(self.gate["tags"])
                rospy.loginfo("[final] gate%d answer=%s -> (%.1f,%.1f) tags %s",
                              self.gi + 1, self.answer, self.gate["c"][0],
                              self.gate["c"][1], sorted(self.gate_tags))
                # Cross VIA a point 0.5 m square in front of the (tag-refined)
                # opening centre, then straight through. The direct diagonal
                # skimmed the centre pillar (confirmed: EGO saw the thin pillar
                # only 0.19 m out, VIO died on impact).
                self._publish_goal(*self._lineup_goal(), force=True)
                self._goto("LINE_UP")
                return
            if self._arrive_t is None:
                self._arrive_t = rospy.Time.now()
            settled = (rospy.Time.now() - self._arrive_t).to_sec() >= self.settle_s
            if (settled and not self._asking and self.api_key
                    and (rospy.Time.now() - self._last_ask_t).to_sec()
                    >= self.retry_s):
                self._last_ask_t = rospy.Time.now()
                self._asking = True
                rospy.loginfo("[final] gate%d: asking quiz", self.gi + 1)
                threading.Thread(target=self._ask, daemon=True).start()

        elif self.state == "LINE_UP":
            # reach the opening's centreline TIGHTLY before committing through:
            # crossing then happens square through the middle of the opening.
            lx, ly = self._lineup_goal()
            self._publish_goal(lx, ly)
            if p is not None and math.hypot(p.pose.position.x - lx,
                                            p.pose.position.y - ly) \
                    <= self.lineup_tol:
                rospy.loginfo("[final] lined up at (%.2f,%.2f) +-%.2f -> THROUGH",
                              lx, ly, self.lineup_tol)
                self._publish_goal(*self._through_goal(), force=True)
                self._goto("THROUGH")

        elif self.state == "THROUGH":
            if self._passed():
                if self.gi >= len(self.gates) - 1:
                    # LAST gate: do NOT turn toward the box scan pos yet -- a
                    # desk sits behind/between the openings. Finish the FULL
                    # straight run to the through point (1.5 m past the centre)
                    # first, then turn left.
                    tx, ty = self._through_goal()
                    if not self._near(tx, ty, 0.3):
                        return
                    rospy.loginfo("[final] gate%d PASSED + straight run done "
                                  "(%.1f,%.1f) -> box scan pos (%.2f,%.2f)",
                                  self.gi + 1, tx, ty, self.box_scan_pos[0],
                                  self.box_scan_pos[1])
                    self._goto("BOX_NAV")
                else:
                    rospy.loginfo("[final] gate%d PASSED", self.gi + 1)
                    # Turn the delivery YOLO/RGB ON after the second-to-last
                    # gate so the streams are fully warmed up before the box
                    # search at the final gate (on-demand was too slow to start).
                    if self.gi == len(self.gates) - 2:
                        self.pub_force.publish(Bool(data=True))
                        rospy.loginfo("[final] gate%d passed -> delivery YOLO "
                                      "ON early (warm up for final gate)",
                                      self.gi + 1)
                    self.gi += 1
                    self.wi = 0
                    self.gate = None
                    self.gate_tags = None
                    self._goto("NAV")

        elif self.state == "BOX_NAV":
            # planner-fly to the box scan spot, then hand off to precision_land.
            sx, sy = self.box_scan_pos
            self._publish_goal(sx, sy)
            if p is not None and math.hypot(p.pose.position.x - sx,
                                            p.pose.position.y - sy) \
                    < self.box_arrive_tol:
                msg = json.dumps({"target": self.target,
                                  "strafe_sign":
                                  self._strafe_sign_for_last_gate(),
                                  "z_bias": round(self._z_bias, 3)})
                self.pub_start.publish(String(data=msg))
                rospy.loginfo("[final] at box scan pos -> precision_land (%s)",
                              msg)
                self._goto("DELIVERY")

        # DELIVERY: precision_land_node owns it now; nothing to do here.


if __name__ == "__main__":
    try:
        FinalMission()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
