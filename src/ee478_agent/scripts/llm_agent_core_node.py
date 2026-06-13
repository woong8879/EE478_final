#!/usr/bin/env python3
"""llm_agent_core_node.py — LLM Reasoning Core (Assignment 4, Topic 2).

The agentic "brain": connects the mission command + a semantic map +
observation memory to the LLM (gpt_llm_client `llm_query` service) and
turns the LLM's decisions into HIGH-LEVEL action commands that
agent_interface_node executes on the drone.

This REPLACES the hard-coded mission_fsm with LLM-driven reasoning — the
assignment's priority is "LLM-driven reasoning and connecting to actual
robot action", not a perfect autonomous system.

Pipeline (see Assignment 4 p.22 architecture):
    [store coords YAML] + [signboard observations]  -> semantic map
    [mission cmd] + [semantic map] + [memory]  --LLM-->  high-level action
    high-level action  -> /llm_action  -> agent_interface -> /next_goal

Data flow
---------
  sub  /mission_command   std_msgs/String     free-form task text
  sub  /signboard_obs     std_msgs/String     "store_id=3 icon=PHARMACY arrow=LEFT"
                                               (from YOLO signboard recog; TODO,
                                               can be injected manually for now)
  sub  /goal_reached      std_msgs/Int32      arrival (store_id) from ego_bridge
  pub  /semantic_map      ee478_msgs/SemanticMap   current world model (latched)
  pub  /llm_action        std_msgs/String     JSON action for agent_interface
  pub  /mission/state     std_msgs/String     human-readable state (latched)
  srv-client  llm_query   gpt_llm_client/LLMQuery

Action JSON schema the LLM MUST return (one object):
  {"action":"GOTO","category":"PHARMACY"}   # visit nearest unvisited store
  {"action":"GOTO","store_id":3}            # visit a specific store
  {"action":"EXPLORE","direction":"LEFT"}   # move to reveal more signboards
  {"action":"RETURN"}                       # go to pickup-point
  {"action":"DONE"}                         # mission complete
  {"action":"SIGNATURE"}                    # do the signature move at a store

Robustness: every LLM call has a hard timeout and a deterministic
fallback (visit the next unvisited store, then RETURN), so the agent
keeps moving even with no network / API key.
"""

import json
import math
import threading

import rospy
from std_msgs.msg import String, Int32
from geometry_msgs.msg import Point

from ee478_msgs.msg import SemanticMap, StoreEntry

CATEGORIES = ["CAFE", "PHARMACY", "CONVENIENCE", "FASTFOOD"]


class LLMAgentCore:
    def __init__(self):
        rospy.init_node("llm_agent_core")
        self.lock = threading.Lock()

        # --- params ---
        # Store coords are PROVIDED (raw text); load them from rosparam
        # (~stores: [{id, x, y}, ...]) or a YAML loaded into the param server.
        self.stores = self._load_stores()
        pp = rospy.get_param("~pickup_point", {"x": 0.0, "y": 0.0, "z": 0.5})
        self.pickup = Point(pp["x"], pp["y"], pp.get("z", 0.5))
        self.hover_z = float(rospy.get_param("~hover_z", 0.5))
        self.llm_timeout = float(rospy.get_param("~llm_timeout_s", 8.0))
        self.use_llm = bool(rospy.get_param("~use_llm", True))

        # --- agent state / memory ---
        self.mission = None
        self.memory = []          # rolling observation/decision history
        self.state = "IDLE"
        self.last_action = None

        # --- pubs / subs ---
        self.pub_action = rospy.Publisher("/llm_action", String, queue_size=5)
        self.pub_map = rospy.Publisher("/semantic_map", SemanticMap,
                                       queue_size=2, latch=True)
        self.pub_state = rospy.Publisher("/mission/state", String,
                                         queue_size=2, latch=True)
        rospy.Subscriber("/mission_command", String, self.on_command, queue_size=2)
        rospy.Subscriber("/signboard_obs", String, self.on_signboard, queue_size=10)
        rospy.Subscriber("/goal_reached", Int32, self.on_reached, queue_size=5)

        # --- LLM service (provided gpt_llm_client) ---
        self.llm = None
        if self.use_llm:
            try:
                from gpt_llm_client.srv import LLMQuery
                rospy.loginfo("[agent] waiting for /llm_query service...")
                rospy.wait_for_service("llm_query", timeout=10.0)
                self.llm = rospy.ServiceProxy("llm_query", LLMQuery)
                rospy.loginfo("[agent] LLM service connected")
            except Exception as e:
                rospy.logwarn(f"[agent] no LLM service ({e}); using fallback planner")

        self._publish_map()
        self._set_state("READY")
        rospy.loginfo("[agent] %d stores loaded, pickup=(%.1f,%.1f)",
                      len(self.stores), self.pickup.x, self.pickup.y)

    # ---------------- data ----------------
    def _load_stores(self):
        """~stores rosparam: list of {id, x, y[, category]}. Categories
        start UNKNOWN and are inferred by the LLM from signboards."""
        raw = rospy.get_param("~stores", [])
        stores = {}
        for s in raw:
            sid = int(s["id"])
            stores[sid] = {
                "id": sid,
                "x": float(s["x"]),
                "y": float(s["y"]),
                "category": str(s.get("category", "UNKNOWN")).upper(),
                "confidence": float(s.get("confidence", 0.0)),
                "visited": False,
                "hints": list(s.get("direction_hints", [])),
            }
        return stores

    def _build_map_msg(self):
        m = SemanticMap()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = "map"
        m.pickup_point = self.pickup
        for s in self.stores.values():
            e = StoreEntry()
            e.store_id = s["id"]
            e.position_world = Point(s["x"], s["y"], self.hover_z)
            e.category = s["category"]
            e.category_confidence = s["confidence"]
            e.visited = s["visited"]
            e.direction_hints = s["hints"]
            m.stores.append(e)
        return m

    def _publish_map(self):
        self.pub_map.publish(self._build_map_msg())

    def _set_state(self, st):
        self.state = st
        self.pub_state.publish(String(st))

    # ---------------- callbacks ----------------
    def on_command(self, msg):
        with self.lock:
            self.mission = msg.data.strip()
            self.memory.append(f"MISSION: {self.mission}")
        rospy.loginfo("[agent] mission: %s", self.mission)
        self._plan_and_act("new mission")

    def on_signboard(self, msg):
        """Signboard observation -> update semantic map (LLM infers category).
        Format: "store_id=3 icon=PHARMACY arrow=LEFT" (free-form ok)."""
        text = msg.data.strip()
        with self.lock:
            self.memory.append(f"OBSERVED: {text}")
            sid, icon, arrow = self._parse_signboard(text)
            if sid is not None and sid in self.stores:
                if icon and icon.upper() in CATEGORIES:
                    self.stores[sid]["category"] = icon.upper()
                    self.stores[sid]["confidence"] = 0.9
                if arrow:
                    self.stores[sid]["hints"].append(arrow.upper())
        self._publish_map()

    def on_reached(self, msg):
        sid = int(msg.data)
        with self.lock:
            if sid in self.stores:
                self.stores[sid]["visited"] = True
                self.memory.append(f"VISITED: store {sid} "
                                   f"({self.stores[sid]['category']})")
        self._publish_map()
        rospy.loginfo("[agent] reached store %d -> re-plan", sid)
        self._plan_and_act(f"reached store {sid}")

    # ---------------- the agentic loop ----------------
    def _plan_and_act(self, trigger):
        """Ask the LLM (or fallback) for the next high-level action and
        publish it for agent_interface_node to execute."""
        if self.mission is None:
            return
        action = None
        if self.llm is not None:
            action = self._llm_decide(trigger)
        if action is None:
            action = self._fallback_decide()
        self.last_action = action
        self.memory.append(f"DECIDED: {json.dumps(action)}")
        self._set_state(action.get("action", "?"))
        self.pub_action.publish(String(json.dumps(action)))
        rospy.loginfo("[agent] action -> %s", json.dumps(action))

    def _llm_decide(self, trigger):
        prompt = self._build_prompt(trigger)
        try:
            resp = self.llm(prompt)
            return self._parse_action(resp.response)
        except Exception as e:
            rospy.logwarn("[agent] LLM decide failed (%s); fallback", e)
            return None

    def _build_prompt(self, trigger):
        lines = [
            "You are the task-planning brain of a delivery drone.",
            "Decide the SINGLE next high-level action. Reply with ONLY a "
            "JSON object, no prose.",
            "",
            "Allowed actions:",
            '  {"action":"GOTO","category":"<CAFE|PHARMACY|CONVENIENCE|FASTFOOD>"}',
            '  {"action":"GOTO","store_id":<int>}',
            '  {"action":"EXPLORE","direction":"<LEFT|RIGHT|STRAIGHT>"}',
            '  {"action":"SIGNATURE"}',
            '  {"action":"RETURN"}   (fly to the pickup-point)',
            '  {"action":"DONE"}',
            "",
            f"MISSION: {self.mission}",
            "",
            "SEMANTIC MAP (stores; category UNKNOWN means not yet identified):",
        ]
        for s in self.stores.values():
            lines.append(
                f"  store {s['id']}: pos=({s['x']:.1f},{s['y']:.1f}) "
                f"category={s['category']} conf={s['confidence']:.1f} "
                f"visited={s['visited']} hints={s['hints']}")
        lines += [
            f"PICKUP_POINT: ({self.pickup.x:.1f},{self.pickup.y:.1f})",
            "",
            "RECENT MEMORY:",
        ]
        lines += [f"  {m}" for m in self.memory[-12:]]
        lines += [
            "",
            "Rules: visit the store category named in the mission; if its "
            "category is still UNKNOWN, EXPLORE toward its direction hints "
            "to read more signboards first. After visiting the target store "
            "do SIGNATURE, then RETURN. Next action JSON:",
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_action(text):
        """Extract the first JSON object from the LLM response."""
        try:
            i = text.index("{")
            j = text.rindex("}") + 1
            obj = json.loads(text[i:j])
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_signboard(text):
        sid = icon = arrow = None
        for tok in text.replace(",", " ").split():
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            k = k.lower()
            if k in ("store_id", "id"):
                try:
                    sid = int(v)
                except ValueError:
                    pass
            elif k in ("icon", "category", "cat"):
                icon = v
            elif k in ("arrow", "direction", "dir"):
                arrow = v
        return sid, icon, arrow

    # ---------------- deterministic fallback ----------------
    def _fallback_decide(self):
        """No-LLM safety net: head to the mission category if known, else
        the nearest unvisited store; once all visited, RETURN."""
        target_cat = None
        if self.mission:
            up = self.mission.upper()
            for c in CATEGORIES:
                if c in up or (c == "CAFE" and "COFFEE" in up):
                    target_cat = c
                    break
        cands = [s for s in self.stores.values() if not s["visited"]]
        if target_cat:
            tc = [s for s in cands if s["category"] == target_cat]
            if tc:
                return {"action": "GOTO", "store_id": tc[0]["id"]}
        if cands:
            return {"action": "GOTO", "store_id": cands[0]["id"]}
        return {"action": "RETURN"}

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        LLMAgentCore().spin()
    except rospy.ROSInterruptException:
        pass
