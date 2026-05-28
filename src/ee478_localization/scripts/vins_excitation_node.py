#!/usr/bin/env python3
"""ee478_localization/vins_excitation_node.py

Pre-flight VINS-Fusion excitation injector.

VINS-Fusion mono+IMU cannot resolve metric scale from a STATIC drone
(takeoff is a single-axis vertical climb that gives the optimiser
only one acceleration direction). To bootstrap, the drone needs a
few seconds of 3-axis acceleration / pose variation. We do that by:

1. Waiting until PX4 is armed in OFFBOARD with vio_bridge feeding
   GT vision_pose (so the drone CAN fly).
2. Publishing a small XY+Z lissajous waypoint sequence on /next_goal
   for `excitation_duration_s` seconds (5 s by default). The waypoints
   stay within a ~0.6 m cube around the takeoff point so we never
   leave the pickup zone or hit anything.
3. After excitation, publishing a "ready" String to
   /vins_excitation/done. mission_fsm waits for that before
   transitioning out of AWAIT_TAKEOFF, so the actual mission cannot
   start until VINS has had time to converge.

This is the same approach used on real drones (wave the drone around
by hand for a few seconds before takeoff) — we just automate it.
"""

import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from mavros_msgs.msg import State


class VinsExcitation:
    def __init__(self):
        rospy.init_node("vins_excitation")
        self.lock = threading.Lock()

        self.duration_s = float(
            rospy.get_param("~excitation_duration_s", 5.0))
        self.amplitude_xy = float(
            rospy.get_param("~amplitude_xy_m", 0.3))
        self.amplitude_z  = float(
            rospy.get_param("~amplitude_z_m", 0.15))
        self.center_z = float(rospy.get_param("~center_z_m", 0.7))
        self.world_frame = rospy.get_param("~world_frame", "map")
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))

        self.armed_offboard = False
        self.takeoff_seen = False
        self.drone_pose = None
        self.done = False
        self.start_time = None

        self.pub_goal = rospy.Publisher("/next_goal", PoseStamped,
                                        queue_size=10)
        self.pub_done = rospy.Publisher(
            "/vins_excitation/done", Bool, queue_size=1, latch=True)

        rospy.Subscriber("/mavros/state", State,
                         self.on_state, queue_size=5)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)

        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.tick)
        rospy.loginfo(
            f"[vins_excite] waiting for arm+offboard+takeoff, will then "
            f"shake xy={self.amplitude_xy:.2f} z={self.amplitude_z:.2f} "
            f"for {self.duration_s:.1f} s")

    def on_state(self, msg):
        with self.lock:
            self.armed_offboard = (msg.armed and msg.mode == "OFFBOARD")

    def on_pose(self, msg):
        with self.lock:
            self.drone_pose = msg
            if msg.pose.position.z > 0.3:
                self.takeoff_seen = True

    def tick(self, _evt):
        with self.lock:
            done = self.done
            armed = self.armed_offboard
            took_off = self.takeoff_seen
            pose = self.drone_pose
            start = self.start_time

        if done:
            return
        if not (armed and took_off and pose is not None):
            return

        now = rospy.Time.now()
        if start is None:
            with self.lock:
                self.start_time = now
            self.center_x = pose.pose.position.x
            self.center_y = pose.pose.position.y
            rospy.loginfo(
                f"[vins_excite] start at ({self.center_x:.2f},"
                f"{self.center_y:.2f}); excitation for "
                f"{self.duration_s:.1f} s")
            start = now

        t = (now - start).to_sec()
        if t >= self.duration_s:
            with self.lock:
                self.done = True
            # Latched /vins_excitation/done = True so the FSM keeps
            # seeing it even after this node exits.
            self.pub_done.publish(Bool(data=True))
            rospy.loginfo(
                "[vins_excite] excitation complete; signalled "
                "/vins_excitation/done = true; node shutting down to "
                "release /next_goal publisher")
            # Hard shutdown so no lingering tick can race with the
            # FSM's APPROACH_QUIZ goal publication.
            rospy.Timer(rospy.Duration(0.2),
                        lambda _e: rospy.signal_shutdown("done"),
                        oneshot=True)
            return

        # Lissajous-ish path: x = a*sin(2*pi*f*t), y = a*sin(2*pi*f*t+pi/2)
        # z bobbles independently. f=1 Hz so we get ~5 full cycles in 5 s.
        f = 1.0
        dx = self.amplitude_xy * math.sin(2.0 * math.pi * f * t)
        dy = self.amplitude_xy * math.sin(2.0 * math.pi * f * t
                                          + math.pi / 2.0)
        dz = self.amplitude_z  * math.sin(2.0 * math.pi * f * 1.3 * t)
        self._publish(self.center_x + dx,
                      self.center_y + dy,
                      self.center_z + dz)

    def _publish(self, x, y, z):
        g = PoseStamped()
        g.header.stamp = rospy.Time.now()
        g.header.frame_id = self.world_frame
        g.pose.position.x = x
        g.pose.position.y = y
        g.pose.position.z = z
        g.pose.orientation.w = 1.0
        self.pub_goal.publish(g)


if __name__ == "__main__":
    try:
        VinsExcitation()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
