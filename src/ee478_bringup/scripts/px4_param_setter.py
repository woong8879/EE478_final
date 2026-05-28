#!/usr/bin/env python3
"""ee478_drone_control/px4_param_setter.py

One-shot node that waits for MAVROS, sets a handful of PX4 parameters
that constrain horizontal and vertical velocity (so the drone tracks
its setpoint stream conservatively in the cramped drone_worlds map),
and then exits.

Defaults are tuned for a ~10 m x 25 m course with 3 m tall walls and a
~2 m wide gap. With MPC_XY_VEL_MAX = 1.5 m/s the drone takes the full
~20 m traverse in ~15 s and PX4's controller overshoot stays under
~0.3 m — comfortably inside the 0.45 m obstacle inflation buffer.

If you ever push these higher, also widen the obstacle radii in
ee478_drone_control/config/obstacles.yaml accordingly.
"""

import rospy
from mavros_msgs.srv import ParamGet, ParamSet, ParamSetRequest
from mavros_msgs.msg import ParamValue


PARAMS = {
    # Lower than the PX4 defaults (12 / 5 / 3 m/s) so that controller
    # overshoot in the 2 m wide wall_3m gap stays under 0.3 m — earlier
    # 1.8 m/s allowed up to 1 m lateral drift through the gap.
    "MPC_XY_VEL_MAX":   0.3,   # max horizontal velocity (m/s) — slow
                               # so ICP keeps frame-to-frame motion
                               # within its registration window.
    "MPC_XY_CRUISE":    0.2,   # cruise horizontal velocity (m/s)
    "MPC_Z_VEL_MAX_UP": 0.7,   # max climb rate (m/s)
    "MPC_Z_VEL_MAX_DN": 0.5,   # max descent rate (m/s)
    "MPC_TKO_SPEED":    0.7,   # takeoff vertical speed (m/s)
    "MPC_LAND_SPEED":   0.4,   # landing approach speed (m/s)
    "MIS_TAKEOFF_ALT":  0.6,   # default mission takeoff altitude (m).
                               # PX4 briefly enters AUTO.TAKEOFF on arm
                               # then hands over to OFFBOARD; during the
                               # handover the drone targets this value.
                               # Old 1.7 m overshot ABOVE the gate top
                               # bar (1.35 m). 0.6 m matches hover_z.
}


def set_param(srv, name, value):
    req = ParamSetRequest()
    req.param_id = name
    pv = ParamValue()
    if isinstance(value, int):
        pv.integer = value
        pv.real = 0.0
    else:
        pv.integer = 0
        pv.real = float(value)
    req.value = pv
    res = srv.call(req)
    if not res.success:
        rospy.logwarn(f"[px4_params] FAILED to set {name} = {value}")
    else:
        rospy.loginfo(f"[px4_params] set {name} = {value}")
    return res.success


def get_param(srv_get, name):
    """Return current value of `name` (or None if get fails)."""
    try:
        res = srv_get.call(param_id=name)
        if not res.success:
            return None
        return res.value.real if res.value.real != 0.0 else float(res.value.integer)
    except Exception:
        return None


def main():
    rospy.init_node("px4_param_setter")
    rospy.loginfo("[px4_params] waiting for /mavros/param/{set,get}...")
    rospy.wait_for_service("/mavros/param/set", timeout=120.0)
    rospy.wait_for_service("/mavros/param/get", timeout=120.0)
    srv = rospy.ServiceProxy("/mavros/param/set", ParamSet)
    srv_get = rospy.ServiceProxy("/mavros/param/get", ParamGet)

    # Wait until MAVROS is actually connected to the FCU — PX4
    # rejects param set requests issued before the heartbeat handshake.
    from mavros_msgs.msg import State
    state_pkt = {"got": False}

    def on_state(msg):
        if msg.connected:
            state_pkt["got"] = True

    sub = rospy.Subscriber("/mavros/state", State, on_state)
    deadline = rospy.Time.now() + rospy.Duration(60.0)
    rate = rospy.Rate(2.0)
    while not state_pkt["got"] and rospy.Time.now() < deadline and not rospy.is_shutdown():
        rate.sleep()
    sub.unregister()
    if not state_pkt["got"]:
        rospy.logwarn("[px4_params] MAVROS connection not seen; trying anyway")
    else:
        rospy.loginfo("[px4_params] MAVROS connected; waiting 8 s for PX4 to "
                      "finish populating its parameter table")
        # Even after connected=True, PX4's parameter daemon takes a few
        # seconds to populate values. Calling ParamSet too early returns
        # success=False from MAVROS. Empirically 8 s of grace is enough.
        rospy.sleep(8.0)

    # Try each param a few times — first attempts may fail while PX4's
    # internal state is still settling after arming, etc. After each
    # set, READ THE VALUE BACK to confirm PX4 actually accepted it
    # (the success=True from ParamSet can lie when the param table is
    # still loading).
    ok_count = 0
    for name, value in PARAMS.items():
        success = False
        for attempt in range(10):
            try:
                set_param(srv, name, value)
            except Exception as e:
                rospy.logwarn(f"[px4_params] exception setting {name} "
                              f"(attempt {attempt+1}): {e}")
                rospy.sleep(1.0)
                continue
            # Verify
            rospy.sleep(0.3)
            cur = get_param(srv_get, name)
            if cur is not None and abs(cur - float(value)) < 1e-3:
                rospy.loginfo(f"[px4_params] verified {name} = {cur:.3f}")
                success = True
                break
            rospy.logwarn(f"[px4_params] {name} read back as {cur} "
                          f"(want {value}); retry {attempt+1}/10")
            rospy.sleep(1.0)
        if success:
            ok_count += 1
        else:
            rospy.logerr(f"[px4_params] gave up on {name} after 10 attempts — "
                         f"DRONE WILL OVERSHOOT corridor waypoints if vel "
                         f"caps not applied. Set manually with "
                         f"`rosservice call /mavros/param/set` and restart "
                         f"topic2_agent.")
    rospy.loginfo(f"[px4_params] done — {ok_count}/{len(PARAMS)} parameters verified")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rospy.logerr(f"[px4_params] fatal: {e}")
