#!/usr/bin/env python3
"""ee478_agent/quiz_solver_node.py

Mission step 2: choose which gate of the quiz_pair the drone should
fly through.

Input
-----
  /quiz/gates  (ee478_msgs/QuizGateArray)
                A QuizGateArray published when perception has read
                BOTH the question text and all candidate gates'
                world-frame centers and their integer labels.

Output
------
  /quiz/chosen_pose  (geometry_msgs/PoseStamped, latched)
                      Goal point inside the chosen gate aperture, in
                      the map frame, suitable for /next_goal.
  /quiz/chosen_label (std_msgs/Int32, latched)
                      The integer that solves the question.

Algorithm
---------
- safe_eval the question (single arithmetic expression of `+ - * / //`
  and integer literals; nothing else permitted).
- pick the gate whose label equals the answer.
- if no exact match, pick the closest label (perception OCR error).

We deliberately keep the math here: it's deterministic and >1000x
cheaper than an LLM round trip, and the EE478 quiz format is
explicitly a small-integer arithmetic problem per the spec image.
"""

import ast
import operator
import math
import re

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32

from ee478_msgs.msg import QuizGateArray


_BINOPS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
}
_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(expr):
    """Evaluate a single arithmetic expression. Raises ValueError on
    anything outside +,-,*,/,//,%,**, parens, and int/float literals.
    """
    # Normalise: drop a trailing "= ?" / "=" / "?" possibly separated
    # by whitespace. rstrip() can't do this because the chars may be
    # interleaved with spaces.
    expr = re.sub(r"\s*=?\s*\??\s*$", "", expr.strip())
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(
                node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_eval(node.left),
                                           _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](_eval(node.operand))
        raise ValueError(f"disallowed node {type(node).__name__}")

    return _eval(tree)


class QuizSolver:
    def __init__(self):
        rospy.init_node("quiz_solver")

        self.in_topic = rospy.get_param(
            "~in_topic", "/quiz/gates")
        self.pose_topic = rospy.get_param(
            "~pose_topic", "/quiz/chosen_pose")
        self.label_topic = rospy.get_param(
            "~label_topic", "/quiz/chosen_label")
        self.world_frame = rospy.get_param("~world_frame", "map")
        self.target_z = float(rospy.get_param("~target_z", 0.7))

        self.solved = False

        self.pub_pose = rospy.Publisher(self.pose_topic, PoseStamped,
                                        queue_size=1, latch=True)
        self.pub_label = rospy.Publisher(self.label_topic, Int32,
                                         queue_size=1, latch=True)

        rospy.Subscriber(self.in_topic, QuizGateArray, self.on_quiz,
                         queue_size=2)

        rospy.loginfo(
            f"[quiz_solver] {self.in_topic} -> {self.pose_topic} + "
            f"{self.label_topic} (target_z {self.target_z:.2f})")

    def on_quiz(self, msg):
        if self.solved:
            return
        if not msg.gates:
            rospy.logwarn_throttle(
                2.0, "[quiz_solver] empty QuizGateArray, ignoring")
            return
        try:
            answer = safe_eval(msg.question)
        except Exception as e:
            rospy.logerr(
                f"[quiz_solver] cannot eval question '{msg.question}': "
                f"{e}")
            return

        answer_int = int(round(float(answer)))
        # Pick exact match if present; otherwise nearest label.
        exact = [g for g in msg.gates if g.label == answer_int]
        if exact:
            chosen = exact[0]
            via = "exact"
        else:
            chosen = min(msg.gates,
                         key=lambda g: abs(g.label - answer_int))
            via = f"nearest({chosen.label} vs {answer_int})"

        out = PoseStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.world_frame
        out.pose.position.x = chosen.center_world.x
        out.pose.position.y = chosen.center_world.y
        out.pose.position.z = self.target_z
        out.pose.orientation.w = 1.0
        self.pub_pose.publish(out)
        self.pub_label.publish(Int32(data=chosen.label))
        self.solved = True

        rospy.loginfo(
            f"[quiz_solver] '{msg.question}' = {answer_int} "
            f"-> gate label {chosen.label} at "
            f"({chosen.center_world.x:.2f},{chosen.center_world.y:.2f}) "
            f"[{via}]")


if __name__ == "__main__":
    try:
        QuizSolver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
