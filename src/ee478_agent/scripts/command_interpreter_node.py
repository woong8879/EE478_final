#!/usr/bin/env python3
"""ee478_agent/command_interpreter_node.py

Mission step 1: parse the natural-language delivery command into a
single store category.

Input
-----
  /mission_command   (std_msgs/String)
                      Free-form English: "Deliver to the cafe",
                      "Bring the pills to the pharmacy", ...
  --or-- ~initial_command parameter at boot for one-shot mode.

Output
------
  /mission_target    (std_msgs/String)
                      One of CAFE / PHARMACY / CONVENIENCE / FASTFOOD.
                      Latched so the FSM sees the target whenever it
                      subscribes.

Strategy
--------
1. KEYWORD MATCH on the lowered command — fast, works offline. This
   covers ~95% of expected inputs since the four categories use a
   small fixed vocabulary.
2. LLM FALLBACK (gpt_llm_client) only if keyword match is ambiguous
   AND the OPENAI_API_KEY env var is set. We never let the node hang
   on a network call: the LLM path has a hard 5 s timeout and falls
   back to the highest-frequency keyword on failure.

Robustness: a one-shot ~initial_command means the node can be used
from a launch file without any external publisher.
"""

import os
import re
import threading

import rospy
from std_msgs.msg import String


CATEGORY_KEYWORDS = {
    "CAFE":        ["cafe", "coffee", "espresso", "latte", "cappuccino"],
    "PHARMACY":    ["pharmacy", "drug", "drugstore", "medic", "pill",
                    "prescription"],
    "CONVENIENCE": ["convenience", "convenient", "mart", "store",
                    "snack", "groceries", "grocery"],
    "FASTFOOD":    ["fastfood", "fast food", "burger", "fries",
                    "mcdonald", "fast-food"],
}


def keyword_classify(text):
    """Return (category, confidence) or (None, 0.0) if ambiguous.

    Confidence = (matches_of_best / max(1, total_matches)).
    A clean single-category match is 1.0; two categories matched
    equally is 0.5 -> caller treats as ambiguous and may LLM.
    """
    t = text.lower()
    hits = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                hits[cat] = hits.get(cat, 0) + 1
    if not hits:
        return None, 0.0
    best_cat = max(hits, key=hits.get)
    best_n = hits[best_cat]
    total = sum(hits.values())
    return best_cat, best_n / float(total)


def llm_classify(text, timeout_s=5.0):
    """Optional OpenAI fallback. Returns category or None. Never raises.

    Imported lazily so the node still loads when the openai package
    isn't installed (typical Jetson minimal image).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    try:
        client = OpenAI(timeout=timeout_s)
        prompt = (
            "Classify the following delivery command into EXACTLY ONE "
            "of: CAFE, PHARMACY, CONVENIENCE, FASTFOOD. Respond with "
            "only the single label, no explanation.\n\n"
            f"Command: {text}\nLabel:")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
        )
        out = resp.choices[0].message.content.strip().upper()
        # Be forgiving about minor formatting issues.
        out = re.sub(r"[^A-Z]", "", out)
        if out in CATEGORY_KEYWORDS:
            return out
    except Exception as e:
        rospy.logwarn(f"[command_interpreter] LLM call failed: {e}")
    return None


class CommandInterpreter:
    def __init__(self):
        rospy.init_node("command_interpreter")
        self.lock = threading.Lock()

        self.in_topic = rospy.get_param("~in_topic", "/mission_command")
        self.out_topic = rospy.get_param("~out_topic", "/mission_target")
        # Confidence below this triggers the LLM fallback.
        self.ambiguity_threshold = float(
            rospy.get_param("~ambiguity_threshold", 0.6))
        self.allow_llm = bool(rospy.get_param("~allow_llm", True))

        self.last_target = None

        self.pub = rospy.Publisher(self.out_topic, String,
                                   queue_size=1, latch=True)
        rospy.Subscriber(self.in_topic, String, self.on_command,
                         queue_size=2)

        # One-shot mode: classify the command supplied via param at
        # boot. Useful for launch files that hard-code the mission.
        initial = rospy.get_param("~initial_command", "").strip()
        if initial:
            rospy.loginfo(
                f"[command_interpreter] initial_command='{initial}'")
            self._classify_and_publish(initial)

        rospy.loginfo(
            f"[command_interpreter] {self.in_topic} -> {self.out_topic} "
            f"(LLM fallback={'on' if self.allow_llm else 'off'})")

    def on_command(self, msg):
        self._classify_and_publish(msg.data)

    def _classify_and_publish(self, text):
        cat, conf = keyword_classify(text)
        if cat is not None and conf >= self.ambiguity_threshold:
            self._publish(cat, source=f"keyword({conf:.2f})", text=text)
            return
        if self.allow_llm:
            llm_cat = llm_classify(text)
            if llm_cat is not None:
                self._publish(llm_cat, source="llm", text=text)
                return
        if cat is not None:
            # Best keyword guess even if ambiguous — better than no answer.
            self._publish(cat, source=f"keyword-ambiguous({conf:.2f})",
                          text=text)
            return
        rospy.logwarn(
            f"[command_interpreter] could not classify '{text}'; "
            f"no output published")

    def _publish(self, cat, source, text):
        with self.lock:
            self.last_target = cat
        self.pub.publish(String(data=cat))
        rospy.loginfo(
            f"[command_interpreter] '{text}' -> {cat} (via {source})")


if __name__ == "__main__":
    try:
        CommandInterpreter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
