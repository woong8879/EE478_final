#!/usr/bin/env python3
"""ee478_perception/sim_world_publisher_node.py

SIM-ONLY shim that fabricates the perception outputs the rest of the
stack needs, using ground-truth gazebo poses. This lets us
end-to-end-test the mission FSM, quiz solver, and signature move
WITHOUT a working YOLO+TensorRT pipeline.

Real-drone replacement: a perception node that publishes the same
topics from camera frames.

Inputs
------
  /gazebo/model_states           (gazebo_msgs/ModelStates)

Outputs (all latched once populated)
------------------------------------
  /semantic_map  (ee478_msgs/SemanticMap)
                    pickup_point + StoreEntry[] (cafe / pharmacy /
                    burger / convenience if present in the world).
  /quiz/gates    (ee478_msgs/QuizGateArray)
                    The question and a 3-lane gate array centred on
                    gate_pair_0 (left/center/right lanes at y=+1.8 / 0
                    / -1.8 relative to the model pose). Labels are
                    parameters so a different question can be set up
                    from the launch file.

Why this exists: in sim we still want to validate the FSM transitions
(read command -> quiz -> store -> signature -> return). On the real
drone, the same /semantic_map + /quiz/gates topics will be filled by
a YOLO-based recogniser and an AprilTag/numeric OCR pass on the
gate-pair scoreboard.
"""

import math
import threading

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point

from ee478_msgs.msg import (
    SemanticMap, StoreEntry, QuizGate, QuizGateArray,
)


# World categories per ee478_project_simple.world. Mapping is
# (gazebo_model_name -> store category). Extend if more buildings
# are added.
DEFAULT_CATEGORY_MAP = {
    "cafe":     "CAFE",
    "pharmacy": "PHARMACY",
    "burger":   "FASTFOOD",
    "convenience": "CONVENIENCE",
}


class SimWorldPublisher:
    def __init__(self):
        rospy.init_node("sim_world_publisher")
        self.lock = threading.Lock()

        self.publish_rate = float(rospy.get_param("~publish_rate_hz", 1.0))
        self.world_frame = rospy.get_param("~world_frame", "map")
        # Pickup point (where the drone takes off) — fixed by the
        # px4_sitl spawn pose. EE478 world: (0, 0, 0).
        self.pickup_xyz = [
            float(rospy.get_param("~pickup_x", 0.0)),
            float(rospy.get_param("~pickup_y", 0.0)),
            float(rospy.get_param("~pickup_z", 0.0)),
        ]
        # Quiz pair gazebo model name.
        self.gate_model = rospy.get_param("~gate_model", "gate_pair_0")
        # 3 lane labels (left, center, right) parameterised so a launch
        # file can match the texture/sign content.
        self.gate_labels = [
            int(rospy.get_param("~gate_left_label", 14)),
            int(rospy.get_param("~gate_center_label", 7)),
            int(rospy.get_param("~gate_right_label", 9)),
        ]
        self.lane_offsets_y = [1.8, 0.0, -1.8]  # from model.sdf
        self.gate_z = float(rospy.get_param("~gate_z", 0.7))
        # Question can be a literal expression OR an empty string in
        # which case a sane default is used.
        self.question = rospy.get_param("~question", "5 + 9 = ?")

        self.latest_models = None

        self.pub_map = rospy.Publisher(
            "/semantic_map", SemanticMap, queue_size=1, latch=True)
        self.pub_quiz = rospy.Publisher(
            "/quiz/gates", QuizGateArray, queue_size=1, latch=True)

        self.category_map = dict(DEFAULT_CATEGORY_MAP)
        # Optional override from launch as a YAML mapping.
        override = rospy.get_param("~category_map_override", {})
        if isinstance(override, dict):
            self.category_map.update(override)

        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self.on_states, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.tick)

        rospy.loginfo(
            f"[sim_world_publisher] "
            f"question='{self.question}' labels={self.gate_labels} "
            f"categories={sorted(self.category_map.keys())}")

    def on_states(self, msg):
        with self.lock:
            self.latest_models = msg

    def _pose_of(self, name):
        with self.lock:
            ms = self.latest_models
        if ms is None:
            return None
        try:
            i = ms.name.index(name)
        except ValueError:
            return None
        return ms.pose[i]

    def tick(self, _evt):
        ms = self.latest_models
        if ms is None:
            return

        # ---- SemanticMap ----
        sm = SemanticMap()
        sm.header.stamp = rospy.Time.now()
        sm.header.frame_id = self.world_frame
        sm.pickup_point = Point(*self.pickup_xyz)
        next_id = 1
        for model_name, cat in self.category_map.items():
            p = self._pose_of(model_name)
            if p is None:
                continue
            s = StoreEntry()
            s.store_id = next_id
            next_id += 1
            s.position_world = Point(p.position.x, p.position.y,
                                     # Stores are buildings; aim for
                                     # the facade-height the camera
                                     # would see (~1 m off the ground).
                                     1.0)
            s.category = cat
            s.category_confidence = 1.0  # GT
            s.visited = False
            sm.stores.append(s)
        self.pub_map.publish(sm)

        # ---- QuizGateArray ----
        gpose = self._pose_of(self.gate_model)
        if gpose is None:
            rospy.logwarn_throttle(
                5.0,
                f"[sim_world_publisher] gazebo model "
                f"'{self.gate_model}' not seen; quiz topic not yet "
                f"populated")
            return
        gx = gpose.position.x
        # The lane y-offsets are in the model frame; the gate pair has
        # yaw=0 in ee478_project_simple.world so model y == world y.
        qa = QuizGateArray()
        qa.header.stamp = rospy.Time.now()
        qa.header.frame_id = self.world_frame
        qa.question = self.question
        for label, dy in zip(self.gate_labels, self.lane_offsets_y):
            g = QuizGate()
            g.label = label
            g.center_world = Point(gx, gpose.position.y + dy, self.gate_z)
            qa.gates.append(g)
        self.pub_quiz.publish(qa)


if __name__ == "__main__":
    try:
        SimWorldPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
