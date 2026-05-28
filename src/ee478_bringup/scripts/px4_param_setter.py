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
    # MPC velocity caps. Need to be loose enough that EGO-Planner's
    # commanded velocities aren't clipped (max ~1 m/s in our setup).
    "MPC_XY_VEL_MAX":   1.5,   # max horizontal velocity (m/s)
    "MPC_XY_CRUISE":    1.0,   # cruise horizontal velocity (m/s)
    "MPC_Z_VEL_MAX_UP": 0.7,
    "MPC_Z_VEL_MAX_DN": 0.5,
    "MPC_TKO_SPEED":    0.7,
    "MPC_LAND_SPEED":   0.4,
    "MIS_TAKEOFF_ALT":  0.6,
    # EKF vision-pose tuning. In sim the gt_vision_bridge feeds
    # PERFECT ground-truth pose at 30 Hz, so we tell EKF to trust it
    # absolutely. Without these tight noises, EKF rejects vision-pose
    # samples after fast maneuvers (large delta vs IMU integration)
    # and dead-reckons on IMU, diverging by several metres — exactly
    # what made the drone "reach the cafe but land on ground at
    # mavros-reported xy=(12.86, 1.61) when gazebo xy=(19.88, 2.73)".
    "EKF2_EVP_NOISE":   0.02,  # vision position noise (m). very tight.
    "EKF2_EVA_NOISE":   0.05,  # vision angle noise (rad). very tight.
    "EKF2_EVV_NOISE":   0.05,  # vision velocity noise (m/s).
    # GATE: how many sigmas off before EKF rejects. Larger => harder
    # to reject vision-pose. 5 means up to 5*EVP_NOISE jump is still
    # accepted (= 10 cm).
    "EKF2_EV_GATE":     5.0,
    # NOAID_TMOUT: how long EKF will fly without ANY aiding before
    # giving up. 5 s instead of the default 1 s gives more grace
    # during brief vision-pose stalls (e.g. our gt_bridge missing a
    # sample). Param is microseconds.
    "EKF2_NOAID_TOUT":  5000000,
    # DISARM PREVENTION. PX4 auto-disarms if it thinks the drone has
    # landed (LNDMC_TRIG_TIME after touchdown) or has been idle on
    # the ground. In sim, any brief contact with the ground during
    # transit triggers this and the whole mission falls over. We
    # want the FSM in full control of when the drone arms/disarms.
    "COM_DISARM_LAND":     -1.0,   # never auto-disarm after landing
    "COM_DISARM_PRFLT":    -1.0,   # never auto-disarm pre-flight
    "COM_DISARM_MAN":       0.0,   # disable joystick auto-disarm
    # Land detection: bump trigger time to 30 s and lower required
    # thrust so the drone doesn't get flagged as landed during
    # transient low-z dips.
    "LNDMC_TRIG_TIME":     30.0,
    # Don't switch out of OFFBOARD just because the setpoint stream
    # had a brief gap. EGO can stall for several seconds when its
    # grid_map fills with phantom obstacles during fast manoeuvres;
    # we want to keep OFFBOARD throughout.
    "COM_OF_LOSS_T":       10.0,
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
