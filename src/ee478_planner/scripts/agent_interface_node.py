#!/usr/bin/env python3
"""agent_interface_node.py — Agent Interface (Assignment 4, Topic 2).

The node that "connects LLM outputs to actual robot actions". It consumes
the high-level JSON actions emitted by llm_agent_core, looks up the
semantic map, and turns each action into a CONCRETE goal/command that the
existing motion stack already executes:

    /llm_action (JSON)  -->  /next_goal (PoseStamped)  -->  ego_bridge
                        -->  EGO local planner  -->  offboard_controller  -->  PX4

So the LLM never talks to MAVROS directly — it speaks intents, this node
grounds them into map-frame goals (and the signature trigger).

Data flow
---------
  sub  /llm_action     std_msgs/String          JSON from llm_agent_core
  sub  /semantic_map   ee478_msgs/SemanticMap    store positions + categories
  sub  /mavros/local_position/pose  PoseStamped  current pose (EXPLORE / nearest)
  pub  /next_goal      geometry_msgs/PoseStamped  goal for ego_bridge
  pub  /mission/signature_trigger   std_msgs/Int32   fires SIGNATURE move
  pub  /mission/land_trigger        std_msgs/Bool    fires landing on DONE

Action handling
---------------
  GOTO{category}  -> nearest UNVISITED store of that category -> /next_goal
  GOTO{store_id}  -> that store -> /next_goal
  RETURN          -> pickup_point -> /next_goal
  EXPLORE{dir}    -> relative step (LEFT/RIGHT/STRAIGHT) from current pose
  SIGNATURE       -> /mission/signature_trigger
  DONE            -> /mission/land_trigger
"""

import json
import math

import rospy
from std_msgs.msg import String, Int32, Bool
from geometry_msgs.msg import PoseStamped

from ee478_msgs.msg import SemanticMap


class AgentInterface:
    def __init__(self):
        rospy.init_node("agent_interface")
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.hover_z = float(rospy.get_param("~hover_z", 0.5))
        self.explore_step = float(rospy.get_param("~explore_step_m", 1.0))

        self.smap = None
        self.pose = None

        self.pub_goal = rospy.Publisher("/next_goal", PoseStamped, queue_size=2, latch=True)
        self.pub_sig = rospy.Publisher("/mission/signature_trigger", Int32, queue_size=2)
        self.pub_land = rospy.Publisher("/mission/land_trigger", Bool, queue_size=2)

        rospy.Subscriber("/llm_action", String, self.on_action, queue_size=5)
        rospy.Subscriber("/semantic_map", SemanticMap, self.on_map, queue_size=2)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)
        rospy.loginfo("[agent_if] ready: /llm_action -> /next_goal")

    def on_map(self, msg):
        self.smap = msg

    def on_pose(self, msg):
        self.pose = msg

    def on_action(self, msg):
        try:
            act = json.loads(msg.data)
        except Exception:
            rospy.logwarn("[agent_if] bad action json: %s", msg.data)
            return
        a = str(act.get("action", "")).upper()
        if a == "GOTO":
            self._handle_goto(act)
        elif a == "RETURN":
            self._return()
        elif a == "EXPLORE":
            self._explore(str(act.get("direction", "STRAIGHT")).upper())
        elif a == "SIGNATURE":
            rospy.loginfo("[agent_if] SIGNATURE")
            self.pub_sig.publish(Int32(7))
        elif a == "DONE":
            rospy.loginfo("[agent_if] DONE -> land")
            self.pub_land.publish(Bool(True))
        else:
            rospy.logwarn("[agent_if] unknown action: %s", a)

    # ---------------- handlers ----------------
    def _handle_goto(self, act):
        store = None
        if "store_id" in act:
            store = self._store_by_id(int(act["store_id"]))
        elif "category" in act:
            store = self._nearest_unvisited(str(act["category"]).upper())
        if store is None:
            rospy.logwarn("[agent_if] GOTO target not found in semantic map: %s", act)
            return
        self._send_goal(store.position_world.x, store.position_world.y,
                        label=f"store {store.store_id} ({store.category})")

    def _return(self):
        if self.smap is None:
            rospy.logwarn("[agent_if] RETURN but no semantic map yet")
            return
        p = self.smap.pickup_point
        self._send_goal(p.x, p.y, label="pickup-point")

    def _explore(self, direction):
        """Relative step in the body-yaw-agnostic map frame. LEFT=+y,
        RIGHT=-y, STRAIGHT=+x (map x = initial-forward)."""
        if self.pose is None:
            rospy.logwarn("[agent_if] EXPLORE but no pose yet")
            return
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        if direction == "LEFT":
            y += self.explore_step
        elif direction == "RIGHT":
            y -= self.explore_step
        else:  # STRAIGHT
            x += self.explore_step
        self._send_goal(x, y, label=f"explore {direction}")

    # ---------------- helpers ----------------
    def _store_by_id(self, sid):
        if self.smap is None:
            return None
        for s in self.smap.stores:
            if s.store_id == sid:
                return s
        return None

    def _nearest_unvisited(self, category):
        if self.smap is None:
            return None
        cands = [s for s in self.smap.stores
                 if s.category == category and not s.visited]
        if not cands:
            return None
        if self.pose is None:
            return cands[0]
        px, py = self.pose.pose.position.x, self.pose.pose.position.y
        return min(cands, key=lambda s: math.hypot(
            s.position_world.x - px, s.position_world.y - py))

    def _send_goal(self, x, y, label=""):
        g = PoseStamped()
        g.header.frame_id = self.frame_id
        g.header.stamp = rospy.Time.now()
        g.pose.position.x = x
        g.pose.position.y = y
        g.pose.position.z = self.hover_z
        g.pose.orientation.w = 1.0
        self.pub_goal.publish(g)
        rospy.loginfo("[agent_if] -> /next_goal (%.2f,%.2f) %s", x, y, label)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        AgentInterface().spin()
    except rospy.ROSInterruptException:
        pass
