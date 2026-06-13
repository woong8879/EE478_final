#!/usr/bin/env python3
"""gate_quiz_node.py — quiz-gated gate selection + verified passage.

Mission (first gate pair):
  1. waypoint_loop flies the drone to the QUIZ POINT (1.5, 0) and stops there
     (its last waypoint), so the offboard watchdog hovers.
  2. This node detects arrival, settles, grabs an infra1 frame and asks
     ChatGPT (gpt-4o-mini, vision): solve the problem on the monitor -> LEFT or
     RIGHT? The drone keeps hovering while waiting (no goal published).
  3. LEFT  -> /next_goal = left  gate centre (3.5,  1)   tags 281(L) 282(R)
     RIGHT -> /next_goal = right gate centre (3.5, -1)   tags 283(L) 284(R)
  4. NOT waypoint-only: while approaching, /apriltag/detections (with body-frame
     tag positions) is used to (a) VERIFY the correct tag pair is what we are
     flying at, and (b) REFINE the goal to the midpoint actually measured
     between the two gate tags -- so VIO drift cannot put us through the wrong
     opening. Wrong pair sighted close + chosen pair absent -> loud warning.
  5. Passage confirmed when x > gate_x + pass_margin -> /gate_quiz/passed.
  6. Then the remaining course (6.5,0) -> (9.9,0); hover at the final point.

Needs OPENAI_API_KEY (env) or ~api_key param. With no key/answer the drone
just keeps hovering at the quiz point -- fail-safe.
"""
import base64
import json
import os
import threading

import cv2
import requests
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State as MavState
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

import math


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GateQuiz(object):
    def __init__(self):
        rospy.init_node("gate_quiz")
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        # --- geometry (map FLU: +x fwd, +y LEFT) ---
        self.quiz_pt = rospy.get_param("~quiz_point", [1.5, 0.0])
        self.left_gate = rospy.get_param("~left_gate", [3.5, 1.0])
        self.right_gate = rospy.get_param("~right_gate", [3.5, -1.0])
        self.left_tags = set(rospy.get_param("~left_tags", [281, 282]))
        self.right_tags = set(rospy.get_param("~right_tags", [283, 284]))
        self.hover_z = float(rospy.get_param("~hover_z", 0.7))
        # course AFTER the gate; last one is the terminus (hover there)
        self.post_wps = rospy.get_param("~post_waypoints", [[6.5, 0.0],
                                                            [9.9, 0.0]])
        self.arrive_r = float(rospy.get_param("~arrive_radius_m", 0.5))
        self.pass_margin = float(rospy.get_param("~pass_margin_m", 0.8))
        # The GOAL is placed THIS far BEYOND the gate centre (same y), so the
        # path goes THROUGH the opening. Goaling the centre itself made the
        # drone arrive there and hover IN the gate, never passing. Must be
        # > pass_margin so the pass check fires en route.
        self.pass_through = float(rospy.get_param("~pass_through_m", 1.2))
        self.settle_s = float(rospy.get_param("~settle_s", 2.0))
        self.refine_min_step = float(rospy.get_param("~refine_min_step_m", 0.15))
        self.refine_max_shift = float(rospy.get_param("~refine_max_shift_m", 0.8))

        # --- LLM ---
        self.api_key = rospy.get_param("~api_key",
                                       os.getenv("OPENAI_API_KEY", ""))
        self.model = rospy.get_param("~model", "gpt-4o-mini")
        # TWO-STEP query (gpt-4o-mini is unreliable doing vision+reasoning in
        # one shot -- it read "go to RIGHT" yet answered LEFT). Step 1 only
        # TRANSCRIBES the monitor text; step 2 reasons over the TEXT alone,
        # which the mini model does reliably.
        self.q_transcribe = rospy.get_param(
            "~q_transcribe",
            "모니터 화면 사진이야. 보이는 모든 텍스트를 빠짐없이 그대로 "
            "전사해. 해석하지 말고 텍스트만.")
        self.q_reason = rospy.get_param(
            "~q_reason",
            "드론이 왼쪽/오른쪽 게이트 중 하나를 골라 통과해야 해. 모니터에 "
            "쓰인 텍스트는 다음과 같아:\n\n{ocr}\n\n문제를 풀고, 지시문에 "
            "따라 어느 게이트로 가야 하는지 단계별로 추론해. 마지막 줄에는 "
            "정확히 LEFT 또는 RIGHT 한 단어만 출력해.")
        self.retry_s = float(rospy.get_param("~retry_s", 5.0))

        # --- state ---
        # PRE_TAKEOFF -> TO_QUIZ -> ASKING -> TO_GATE -> POST -> DONE
        # The whole sequence is LOCKED until takeoff has actually succeeded
        # (armed + altitude reached): nothing runs on the bench / pre-arm.
        self.state = "PRE_TAKEOFF"
        self.armed = False
        self.pose = None
        self.last_img = None
        self.answer = None         # "LEFT"/"RIGHT"
        self.gate = None           # chosen [x, y]
        self.gate_tags = None      # chosen tag id pair (set)
        self.post_i = 0
        self._arrive_t = None
        self._asking = False
        self._last_refine_t = rospy.Time(0)

        self.pub_goal = rospy.Publisher("/next_goal", PoseStamped, queue_size=2)
        self.pub_ans = rospy.Publisher("/gate_quiz/answer", String,
                                       queue_size=1, latch=True)
        self.pub_passed = rospy.Publisher("/gate_quiz/passed", Bool,
                                          queue_size=1, latch=True)
        self.pub_passed.publish(Bool(data=False))

        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)
        rospy.Subscriber("/mavros/state", MavState, self.on_state,
                         queue_size=5)
        rospy.Subscriber("/camera/infra1/image_rect_raw", Image,
                         self.on_img, queue_size=1, buff_size=2 ** 22)
        rospy.Subscriber("/apriltag/detections", String, self.on_det,
                         queue_size=5)
        rospy.Timer(rospy.Duration(0.2), self.tick)

        if not self.api_key:
            rospy.logwarn("[gate_quiz] NO OPENAI_API_KEY -- will hover at the "
                          "quiz point until a key/answer exists!")
        rospy.loginfo("[gate_quiz] quiz@(%.1f,%.1f) L=%s%s R=%s%s post=%s",
                      self.quiz_pt[0], self.quiz_pt[1],
                      self.left_gate, sorted(self.left_tags),
                      self.right_gate, sorted(self.right_tags), self.post_wps)

    # ---------------- subscribers ----------------
    def on_pose(self, msg):
        with self.lock:
            self.pose = msg

    def on_state(self, msg):
        self.armed = bool(msg.armed)

    def on_img(self, msg):
        with self.lock:
            self.last_img = msg

    def on_det(self, msg):
        """Tag-pair verification + goal refinement while heading to the gate."""
        if self.state != "TO_GATE" or self.gate_tags is None:
            return
        try:
            dets = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            pose = self.pose
        if pose is None:
            return

        seen = {int(d["id"]): d for d in dets}
        chosen = [seen[t] for t in self.gate_tags if t in seen]
        wrong_set = (self.right_tags if self.gate_tags is self.left_tags
                     else self.left_tags)
        wrong = [seen[t] for t in wrong_set if t in seen]

        # (a) VERIFY: wrong pair close ahead while the chosen pair is unseen.
        if (len(wrong) == 2 and not chosen
                and all(d["range"] < 2.5 for d in wrong)):
            rospy.logwarn_throttle(
                1.0, "[gate_quiz] !! WRONG gate pair %s ahead (chosen %s not "
                "seen) -- check the gate!", sorted(wrong_set),
                sorted(self.gate_tags))
            return

        # (b) REFINE: both chosen tags visible w/ body coords -> measured
        # midpoint becomes the goal (drift-proof gate centring).
        if len(chosen) != 2 or any(d.get("body") is None for d in chosen):
            return
        now = rospy.Time.now()
        if (now - self._last_refine_t).to_sec() < 1.0:
            return
        bx = 0.5 * (chosen[0]["body"][0] + chosen[1]["body"][0])
        by = 0.5 * (chosen[0]["body"][1] + chosen[1]["body"][1])
        yaw = _yaw(pose.pose.orientation)
        cy, sy = math.cos(yaw), math.sin(yaw)
        gx = pose.pose.position.x + cy * bx - sy * by
        gy = pose.pose.position.y + sy * bx + cy * by
        # sanity: stay near the surveyed centre
        if math.hypot(gx - self.gate[0], gy - self.gate[1]) > self.refine_max_shift:
            return
        if math.hypot(gx - self.gate[0], gy - self.gate[1]) < self.refine_min_step:
            return
        self.gate = [gx, gy]
        self._last_refine_t = now
        # keep goaling BEYOND the (refined) centre -> fly through, not to, it
        self._publish_goal(gx + self.pass_through, gy)
        rospy.loginfo("[gate_quiz] gate centre refined from tags -> "
                      "(%.2f, %.2f); goal (%.2f, %.2f)", gx, gy,
                      gx + self.pass_through, gy)

    # ---------------- helpers ----------------
    def _publish_goal(self, x, y):
        g = PoseStamped()
        g.header.stamp = rospy.Time.now()
        g.header.frame_id = "map"
        g.pose.position.x = x
        g.pose.position.y = y
        g.pose.position.z = self.hover_z
        g.pose.orientation.w = 1.0
        self.pub_goal.publish(g)

    def _near(self, x, y):
        with self.lock:
            p = self.pose
        if p is None:
            return False
        return math.hypot(p.pose.position.x - x,
                          p.pose.position.y - y) <= self.arrive_r

    # ---------------- LLM ----------------
    def _ask_llm(self):
        """Runs in a thread: image -> gpt-4o-mini -> LEFT/RIGHT."""
        try:
            with self.lock:
                img = self.last_img
            if img is None:
                rospy.logwarn("[gate_quiz] no camera frame yet")
                return
            cvimg = self.bridge.imgmsg_to_cv2(img, desired_encoding="mono8")
            ok, jpg = cv2.imencode(".jpg", cvimg,
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                rospy.logwarn("[gate_quiz] jpeg encode failed")
                return
            b64 = base64.b64encode(jpg.tobytes()).decode()
            rospy.loginfo("[gate_quiz] asking %s (%d KB image)...",
                          self.model, len(b64) // 1024)
            url = "https://api.openai.com/v1/chat/completions"
            hdr = {"Authorization": "Bearer " + self.api_key}
            # STEP 1: vision -> verbatim transcription only
            r1 = requests.post(url, headers=hdr, json={
                "model": self.model,
                "max_tokens": 300,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.q_transcribe},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/jpeg;base64," + b64,
                            "detail": "high"}},
                    ],
                }],
            }, timeout=45)
            r1.raise_for_status()
            ocr = r1.json()["choices"][0]["message"]["content"].strip()
            rospy.loginfo("[gate_quiz] monitor text:\n%s", ocr)
            # STEP 2: text-only reasoning over the transcription
            r2 = requests.post(url, headers=hdr, json={
                "model": self.model,
                "max_tokens": 300,
                "messages": [{"role": "user",
                              "content": self.q_reason.format(ocr=ocr)}],
            }, timeout=45)
            r2.raise_for_status()
            txt = r2.json()["choices"][0]["message"]["content"].strip()
            rospy.loginfo("[gate_quiz] LLM reasoning:\n%s", txt)
            # FINAL verdict = the LAST occurrence of LEFT/RIGHT.
            up = txt.upper()
            li, ri = up.rfind("LEFT"), up.rfind("RIGHT")
            if li < 0 and ri < 0:
                rospy.logwarn("[gate_quiz] no LEFT/RIGHT in answer -- retrying")
            else:
                self.answer = "LEFT" if li > ri else "RIGHT"
                rospy.loginfo("[gate_quiz] final answer: %s", self.answer)
        except Exception as e:
            rospy.logwarn("[gate_quiz] LLM query failed: %s (retry in %.0fs)",
                          e, self.retry_s)
        finally:
            self._asking = False

    # ---------------- state machine ----------------
    def tick(self, _evt):
        if self.state == "PRE_TAKEOFF":
            # SEQUENCE GATE: only after takeoff has SUCCEEDED (armed + at
            # altitude) does the quiz/tag sequence start. Bench handling or a
            # pre-arm tag/pose can never kick anything off.
            with self.lock:
                p = self.pose
            if (self.armed and p is not None
                    and p.pose.position.z >= self.hover_z - 0.15):
                rospy.loginfo("[gate_quiz] takeoff success (armed, z=%.2f) -> "
                              "quiz sequence STARTED", p.pose.position.z)
                self.state = "TO_QUIZ"
            return

        if self.state == "TO_QUIZ":
            if self._near(self.quiz_pt[0], self.quiz_pt[1]):
                if self._arrive_t is None:
                    self._arrive_t = rospy.Time.now()
                elif (rospy.Time.now() - self._arrive_t).to_sec() >= self.settle_s:
                    rospy.loginfo("[gate_quiz] at quiz point -- hovering, "
                                  "asking the quiz")
                    self.state = "ASKING"
                    self._arrive_t = None
            else:
                self._arrive_t = None

        elif self.state == "ASKING":
            if self.answer is not None:
                self.pub_ans.publish(String(data=self.answer))
                if self.answer == "LEFT":
                    self.gate = list(self.left_gate)
                    self.gate_tags = self.left_tags
                else:
                    self.gate = list(self.right_gate)
                    self.gate_tags = self.right_tags
                rospy.loginfo("[gate_quiz] answer=%s -> gate (%.2f, %.2f), "
                              "tags %s", self.answer, self.gate[0],
                              self.gate[1], sorted(self.gate_tags))
                self.state = "TO_GATE"
                # goal BEYOND the gate so the path passes THROUGH the opening
                self._publish_goal(self.gate[0] + self.pass_through,
                                   self.gate[1])
            elif not self._asking and self.api_key:
                # hover is automatic (no goal published); ask/retry
                if (rospy.Time.now() - getattr(self, "_last_ask_t",
                                               rospy.Time(0))).to_sec() \
                        >= self.retry_s:
                    self._last_ask_t = rospy.Time.now()
                    self._asking = True
                    threading.Thread(target=self._ask_llm,
                                     daemon=True).start()

        elif self.state == "TO_GATE":
            # NOTE: the goal is published on entry + on each tag refine only --
            # ego_bridge resends the pending goal at 1 Hz by itself, and
            # flooding /next_goal would back up its burst-publish callback.
            with self.lock:
                p = self.pose
            if p is not None and \
                    p.pose.position.x > self.gate[0] + self.pass_margin:
                self.pub_passed.publish(Bool(data=True))
                rospy.loginfo("[gate_quiz] GATE PASSED (x=%.2f > %.2f)",
                              p.pose.position.x,
                              self.gate[0] + self.pass_margin)
                self.state = "POST"
                self._post_goal_sent = False

        elif self.state == "POST":
            if self.post_i >= len(self.post_wps):
                rospy.loginfo("[gate_quiz] course DONE -- hovering at "
                              "(%.1f, %.1f)", self.post_wps[-1][0],
                              self.post_wps[-1][1])
                self.state = "DONE"
                return
            wp = self.post_wps[self.post_i]
            if not getattr(self, "_post_goal_sent", False):
                self._publish_goal(wp[0], wp[1])
                self._post_goal_sent = True
            if self._near(wp[0], wp[1]):
                rospy.loginfo("[gate_quiz] post wp %d (%.1f, %.1f) reached",
                              self.post_i, wp[0], wp[1])
                self.post_i += 1
                self._post_goal_sent = False


if __name__ == "__main__":
    try:
        GateQuiz()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
