#!/usr/bin/env python3
"""delivery_trigger_node.py

Delivery-phase camera gate.

Watches /apriltag/detections (JSON [{"id":271,"range":0.43},...] from
apriltag_anchor_node, always-on). When one of the STORE tags (271/274/275/276)
is detected within trigger_range (1.0 m), the delivery phase becomes ACTIVE:

  * /delivery/active   (std_msgs/Bool, latched)        False -> True
  * /delivery/down_rgb (sensor_msgs/Image, 10 Hz)      <- /down_camera/color/...
  * /delivery/main_rgb (sensor_msgs/Image, 10 Hz)      <- /camera/color/...

Before the trigger NOTHING is republished -- the RGB "streams start" at the
trigger, as the mission spec asks (the driver-level streams run from boot
because realsense ros1 cannot start a stream at runtime; in-flight camera node
restarts risk the VIO camera's USB).

YOLOv11 (yolo_delivery_node) subscribes to /delivery/down_rgb, so it computes
only while active.

~hold_s: keep active this long after the LAST qualifying detection
         (0 = stay active forever once triggered). Default 0.
"""
import json

import rospy
from mavros_msgs.msg import State as MavState
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image


class DeliveryTrigger(object):
    def __init__(self):
        rospy.init_node("delivery_trigger")

        ids = rospy.get_param("~trigger_ids", [271, 274, 275, 276])
        self.trigger_ids = set(int(i) for i in ids)
        # 1.12 m: the store tags sit +-1 m to the sides of the gate, so a
        # 1.12 m tag range = the drone is 0.5 m IN FRONT of the gate wall
        # (x=4.0 for gate3 at x=3.5; x=1.5 for the local last-gate at x=2).
        # -> YOLO/RGB turn on 0.5 m before the gate so box detection is already
        # running as the drone crosses.
        self.trigger_range = float(rospy.get_param("~trigger_range_m", 1.12))
        self.hold_s = float(rospy.get_param("~hold_s", 0.0))
        self.rate_hz = float(rospy.get_param("~rgb_rate_hz", 10.0))

        self.down_in = rospy.get_param("~down_rgb_in",
                                       "/down_camera/color/image_raw")
        self.main_in = rospy.get_param("~main_rgb_in",
                                       "/camera/color/image_raw")

        self.active = False
        self.armed = False
        # Sequence order: the trigger only fires IN FLIGHT (armed). A store tag
        # carried past the bench / seen pre-arm can never start the streams.
        self.require_armed = bool(rospy.get_param("~require_armed", True))
        self.last_hit_t = None
        self._last_down_pub = rospy.Time(0)
        self._last_main_pub = rospy.Time(0)

        self.pub_active = rospy.Publisher("/delivery/active", Bool,
                                          queue_size=1, latch=True)
        self.pub_down = rospy.Publisher("/delivery/down_rgb", Image,
                                        queue_size=2)
        self.pub_main = rospy.Publisher("/delivery/main_rgb", Image,
                                        queue_size=2)
        self.pub_active.publish(Bool(data=False))

        rospy.Subscriber("/apriltag/detections", String, self.on_det,
                         queue_size=5)
        rospy.Subscriber("/mavros/state", MavState, self.on_state,
                         queue_size=5)
        # External force-activation (e.g. precision_land after gate passage,
        # where the tag-range trigger may not fire exactly).
        rospy.Subscriber("/delivery/force", Bool, self.on_force, queue_size=2)
        rospy.Subscriber(self.down_in, Image, self.on_down, queue_size=1,
                         buff_size=2 ** 22)
        rospy.Subscriber(self.main_in, Image, self.on_main, queue_size=1,
                         buff_size=2 ** 22)
        if self.hold_s > 0.0:
            rospy.Timer(rospy.Duration(0.5), self.check_hold)

        rospy.loginfo(
            "[delivery_trigger] tags %s within %.2f m -> RGB gates open "
            "(%.0f Hz; hold_s=%s)", sorted(self.trigger_ids),
            self.trigger_range, self.rate_hz,
            "forever" if self.hold_s <= 0 else str(self.hold_s))

    def on_state(self, msg):
        self.armed = bool(msg.armed)

    def on_force(self, msg):
        if msg.data and not self.active:
            self.active = True
            self.last_hit_t = rospy.Time.now()
            self.pub_active.publish(Bool(data=True))
            rospy.loginfo("[delivery_trigger] FORCED active -> RGB streams ON")
        elif not msg.data and self.active:
            self.active = False
            self.pub_active.publish(Bool(data=False))
            rospy.loginfo("[delivery_trigger] FORCED inactive")

    def on_det(self, msg):
        if self.require_armed and not self.armed:
            return
        try:
            dets = json.loads(msg.data)
        except ValueError:
            return
        hit = [d for d in dets if int(d.get("id", -1)) in self.trigger_ids
               and float(d.get("range", 1e9)) <= self.trigger_range]
        if not hit:
            return
        self.last_hit_t = rospy.Time.now()
        if not self.active:
            self.active = True
            self.pub_active.publish(Bool(data=True))
            rospy.loginfo("[delivery_trigger] ACTIVE: tag %s at %.2f m -> "
                          "RGB streams ON", hit[0]["id"], hit[0]["range"])

    def check_hold(self, _evt):
        if (self.active and self.last_hit_t is not None
                and (rospy.Time.now() - self.last_hit_t).to_sec() > self.hold_s):
            self.active = False
            self.pub_active.publish(Bool(data=False))
            rospy.loginfo("[delivery_trigger] inactive (no near tag for %.1fs)"
                          " -> RGB streams OFF", self.hold_s)

    # gated 10 Hz relays ----------------------------------------------------
    def on_down(self, msg):
        if not self.active:
            return
        now = rospy.Time.now()
        if (now - self._last_down_pub).to_sec() >= 1.0 / self.rate_hz:
            self._last_down_pub = now
            self.pub_down.publish(msg)

    def on_main(self, msg):
        if not self.active:
            return
        now = rospy.Time.now()
        if (now - self._last_main_pub).to_sec() >= 1.0 / self.rate_hz:
            self._last_main_pub = now
            self.pub_main.publish(msg)


if __name__ == "__main__":
    try:
        DeliveryTrigger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
