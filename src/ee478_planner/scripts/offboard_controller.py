#!/usr/bin/env python3
"""ee478_drone_control/offboard_controller.py

Generic offboard position controller for PX4 via MAVROS.

Provides a high-level Python API and a ROS interface:

  Topics (subscribed):
    ~goal       (geometry_msgs/PoseStamped)  -- desired pose in local ENU frame
    ~cmd_raw    (mavros_msgs/PositionTarget) -- full trajectory feedforward

  Topics (published):
    /mavros/setpoint_position/local  (geometry_msgs/PoseStamped)
    /mavros/setpoint_raw/local       (mavros_msgs/PositionTarget)
  Services consumed:
    /mavros/cmd/arming
    /mavros/set_mode

Behavior:
  * Streams setpoints at 20 Hz (PX4 requires >= 2 Hz before OFFBOARD will be acc
epted).
  * On startup: takes off to ~takeoff_alt (default 1.5 m) above current xy, sets
 OFFBOARD, arms.
  * Whenever a new ~goal arrives, becomes the new target setpoint.
  * Pure position controller -- PX4's onboard cascade handles the rest.

Run with QGroundControl open for manual override / arming if PX4 refuses auto-ar
m.
"""

import math as _math
import threading

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode


class OffboardController:
    def __init__(self):
        rospy.init_node("offboard_controller")
        self.lock = threading.Lock()

        # ---- params ----
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        # YAW = RATE control (like everything else: P on the error -> a RATE
        # command, clamped). We previously commanded an ABSOLUTE yaw angle,
        # which PX4's attitude loop snaps to at up to MC_YAWRATE_MAX
        # (~200 deg/s) -- QGC's MPC_YAWRAUTO_MAX only limits AUTO-mode yaw,
        # NOT offboard setpoints; that's why the post-quiz turn was violent.
        # Now: yaw_rate = clamp(yaw_kp * yaw_err, +-yaw_rate_max).
        self.yaw_rate_max = _math.radians(
            float(rospy.get_param("~yaw_rate_max_dps", 30.0)))
        self.yaw_kp = float(rospy.get_param("~yaw_kp", 1.0))
        self.takeoff_alt = float(rospy.get_param("~takeoff_alt", 1.5))
        self.auto_arm = bool(rospy.get_param("~auto_arm", True))
        self.frame_id = rospy.get_param("~frame_id", "map")
        # Cap for the velocity feedforward injected on top of position
        # setpoints. Keeps long position steps from sending PX4 into
        # huge climbs/overshoots.
        self.max_step_vel = float(rospy.get_param("~max_step_vel", 1.0))
        # Vertical velocity feedforward cap (m/s). Separately specified
        # because the SITL altitude loop diverges if we let xy and z
        # share a single horizon-aware velocity cap (z gets squeezed to
        # near-zero on long horizontal steps and the drone drifts up).
        self.max_step_vel_z = float(
            rospy.get_param("~max_step_vel_z", 0.5))
        # Within this distance of the setpoint we HOLD with a pure position
        # setpoint (no velocity feedforward) and let PX4's native controller
        # do the work -- same as a human holding POSCTL hands-off. Beyond it
        # we add a capped velocity feedforward to slew smoothly to the goal.
        self.hold_radius_m = float(rospy.get_param("~hold_radius_m", 0.4))

        # ---- custom outer-loop position-hold PID ----
        # Instead of handing PX4 a POSITION setpoint and trusting its MPC, we
        # close the position loop ourselves: velocity_cmd = Kp*pos_err
        # + Ki*∫pos_err - Kd*vel_meas, and send it as a VELOCITY setpoint
        # (position masked) so PX4 only runs the inner velocity->thrust loop.
        # Easier to tune than PX4's MPC gains, and the D term uses the (good)
        # VIO velocity to damp hover drift.
        self.use_pid_hold = bool(rospy.get_param("~use_pid_hold", True))
        self.kp_xy = float(rospy.get_param("~hold_kp_xy", 1.0))
        self.ki_xy = float(rospy.get_param("~hold_ki_xy", 0.10))
        self.kd_xy = float(rospy.get_param("~hold_kd_xy", 0.30))
        self.kp_z = float(rospy.get_param("~hold_kp_z", 1.2))
        self.ki_z = float(rospy.get_param("~hold_ki_z", 0.20))
        self.kd_z = float(rospy.get_param("~hold_kd_z", 0.20))
        self.i_limit = float(rospy.get_param("~hold_i_limit", 0.4))  # xy integral clamp (m/s)
        # Separate (larger) z integral clamp: the EKF still reports a phantom
        # vertical velocity, so the z integral needs more room to bias the
        # command up and pull the drone to the target altitude (else it holds
        # ~0.15 m low). Defaults to i_limit if unset.
        self.i_limit_z = float(rospy.get_param("~hold_i_limit_z", self.i_limit))
        # Output slew-rate (accel) limit: caps how fast the commanded velocity
        # may change per cycle so the drone ramps smoothly instead of jerking
        # ("팍 올렸다 팍 내렸다"). m/s^2. Lower = smoother/softer.
        self.accel_lim = float(rospy.get_param("~hold_accel_lim", 1.5))
        # Separate (tighter) z slew: the EKF vertical velocity is a noisy
        # phantom, so the z command swings ("확 내렸다 확 올라가"); a tighter z
        # slew smooths it. Defaults to accel_lim if unset.
        self.accel_lim_z = float(rospy.get_param("~hold_accel_lim_z",
                                                 self.accel_lim))
        # D-term velocity low-pass factor (0..1): smaller = smoother (more lag).
        self.vel_lp_alpha = float(rospy.get_param("~hold_vel_lp", 0.4))
        self.vel = None                # measured velocity (vx,vy,vz), LP-filtered
        self.vio_z = None              # raw VIO z (vision_pose) for z diagnostics
        self._last_zdiag_t = None
        self.int_x = self.int_y = self.int_z = 0.0
        self.last_vx = self.last_vy = self.last_vz = 0.0  # last cmd (for slew)
        self._last_pid_t = None
        # Takeoff climb RATE (m/s): the z setpoint is RAMPED from the ground
        # up to takeoff_alt at this rate instead of jumping the full step at
        # once. A jump makes PX4 climb at its max (MPC_Z_VEL_MAX_UP) and the
        # fast vertical motion blurs the VIO / outruns the EKF before it
        # settles -> divergence right after takeoff. A gentle ramp keeps the
        # estimate converged. Tune up once stable.
        self.takeoff_climb_rate = float(
            rospy.get_param("~takeoff_climb_rate", 0.25))
        # Soft-start: ease the climb rate 0 -> full over this long so liftoff
        # is extra gentle. self.takeoff_ramp_t0 marks ramp start.
        self.soft_start_s = float(rospy.get_param("~takeoff_soft_start_s", 2.0))
        # LEASH: cap how far the climbing z setpoint may lead the drone's ACTUAL
        # altitude. Keeps the position error tiny during motor spool-up so no
        # windup builds -> no violent liftoff surge. Small = gentler liftoff.
        self.takeoff_lookahead = float(rospy.get_param("~takeoff_lookahead", 0.1))
        self.takeoff_ramp_t0 = None
        # Convergence HOLD: after arming, sit on the ground holding the
        # ground-level setpoint for this long BEFORE starting the climb
        # ramp, so the EKF attitude + SVO position/yaw fully converge while
        # stationary. Climbing before the estimate has settled is what
        # caused the post-takeoff divergence. (This is separate from
        # settle_s, which defers horizontal GOALS after takeoff.)
        # NOW used as the CONVERGENCE GATE: the estimate must stay stable
        # (see converge_vel_thresh) for this long BEFORE the code arms.
        self.converge_hold_s = float(rospy.get_param("~converge_hold_s", 6.0))
        # The EKF velocity must stay below this (m/s) while on the ground for
        # the estimate to count as "converged". A diverging/drifting estimate
        # (bad VIO, gyro bias) shows up as a non-zero velocity even when the
        # drone is sitting still -> we refuse to arm. This is exactly what a
        # human does in slam_assist: wait until the estimate is solid before
        # flying.
        self.converge_vel_thresh = float(
            rospy.get_param("~converge_vel_thresh", 0.15))
        self.require_convergence = bool(
            rospy.get_param("~require_convergence", True))
        # Hold takeoff setpoint for this long after arming so PX4 EKF z
        # converges before any horizontal navigation goals are honoured.
        self.settle_s = float(rospy.get_param("~settle_s", 120.0))
        self.armed_at = None
        # takeoff ramp state
        self.takeoff_z_target = None
        self.takeoff_vio_ref = None
        self.takeoff_ramp_active = False
        # convergence-gate state
        self.cur_speed = None          # latest EKF speed magnitude (m/s)
        self.stable_since = None       # time speed last went below thresh
        self.last_unstable_log = rospy.Time(0)

        # ---- state ----
        self.state = State()
        self.current_pose = None
        self.setpoint = PoseStamped()
        self.setpoint.header.frame_id = self.frame_id
        self.setpoint.pose.position.x = 0.0
        self.setpoint.pose.position.y = 0.0
        self.setpoint.pose.position.z = self.takeoff_alt
        self.setpoint.pose.orientation.w = 1.0
        self.has_origin = False
        self.use_raw = False
        self.raw_setpoint = PositionTarget()
        # EGO trajectory velocity feedforward, added to the velocity-PID output
        # while EGO is actively streaming (decays to 0 when its cmds go stale).
        self.ego_ff = [0.0, 0.0, 0.0]
        # cmd_raw watchdog: when EGO stops feeding commands (course complete, or
        # a long replan/SAFETY stall), re-engage the tight custom PID hover
        # instead of riding the stale raw setpoint (PX4-native hold lets the
        # altitude sink). Forgiving of normal replan hiccups (<1 s).
        self.last_raw_t = None
        self.raw_timeout = float(rospy.get_param("~raw_timeout_s", 1.0))

        # ---- pubs / subs / services ----
        self.pub_sp = rospy.Publisher("/mavros/setpoint_position/local",
                                      PoseStamped, queue_size=10)
        self.pub_sp_raw = rospy.Publisher("/mavros/setpoint_raw/local",
                                          PositionTarget, queue_size=10)
        rospy.Subscriber("/mavros/state", State, self.on_state, queue_size=5)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped,
                         self.on_pose, queue_size=5)
        rospy.Subscriber("/mavros/local_position/velocity_local",
                         TwistStamped, self.on_velocity, queue_size=5)
        rospy.Subscriber("/mavros/vision_pose/pose", PoseStamped,
                         self.on_vision, queue_size=5)
        # In "odom" EV mode the relay publishes to /mavros/odometry/out, NOT
        # vision_pose -> subscribe here too so the Z-DIAG vio field isn't NA.
        rospy.Subscriber("/mavros/odometry/out", Odometry,
                         self.on_vision_odom, queue_size=5)
        rospy.Subscriber("~goal", PoseStamped, self.on_goal, queue_size=5)
        rospy.Subscriber("~cmd_raw", PositionTarget, self.on_cmd_raw, queue_size=5)

        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        self.srv_arm = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.srv_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        rospy.loginfo("[offboard] ready; auto_arm=%s takeoff_alt=%.2f",
                      self.auto_arm, self.takeoff_alt)

    # ---------------- callbacks ----------------
    def on_state(self, msg):
        self.state = msg

    def on_vision(self, msg):
        # Raw VIO z straight from SVO (before EKF fusion) -- lets the Z-DIAG
        # log compare what the camera sees vs what the EKF believes.
        self.vio_z = msg.pose.position.z

    def on_vision_odom(self, msg):
        # Same purpose for "odom" EV mode (relay -> /mavros/odometry/out).
        self.vio_z = msg.pose.pose.position.z

    def on_velocity(self, msg):
        v = msg.twist.linear
        # Low-pass (EMA) the velocity used as the PID D-term: raw VIO velocity
        # is noisy, and Kd amplifies that noise into jittery commands. Smoothing
        # it lets us use a higher Kd for real damping (kills overshoot) without
        # injecting jitter. alpha small => smoother but more lag.
        a = self.vel_lp_alpha
        if self.vel is None:
            self.vel = (v.x, v.y, v.z)
        else:
            self.vel = (a * v.x + (1.0 - a) * self.vel[0],
                        a * v.y + (1.0 - a) * self.vel[1],
                        a * v.z + (1.0 - a) * self.vel[2])
        self.cur_speed = (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5
        now = rospy.Time.now()
        if self.cur_speed > self.converge_vel_thresh:
            self.stable_since = None        # estimate moving/diverging -> reset
        elif self.stable_since is None:
            self.stable_since = now         # just became stable

    def _converged(self):
        """True once the EKF estimate has stayed stable (low speed while on
        the ground) for converge_hold_s -- our 'is it safe to fly' gate."""
        if not self.require_convergence:
            return True
        if self.cur_speed is None or self.stable_since is None:
            return False
        if not self.has_origin:             # need a valid local position too
            return False
        return (rospy.Time.now() - self.stable_since).to_sec() >= self.converge_hold_s

    def on_pose(self, msg):
        self.current_pose = msg
        # While NOT armed, keep the warm-up setpoint glued to the CURRENT
        # position. We do NOT latch the takeoff reference here: the estimate
        # may still be jumping at startup. The real takeoff origin is latched
        # at ARM (after convergence) in _latch_takeoff_origin(), so takeoff is
        # always relative to the settled position -- startup jumps / baseline
        # resets no longer bias the start altitude.
        if not self.has_origin:
            self.has_origin = True       # a valid local position now exists
        if self.armed_at is None:        # pre-arm: track current
            with self.lock:
                self.setpoint.pose.position.x = msg.pose.position.x
                self.setpoint.pose.position.y = msg.pose.position.y
                self.setpoint.pose.position.z = msg.pose.position.z
                self.setpoint.pose.orientation = msg.pose.orientation

    def _yaw_rate_cmd(self, desired):
        """P-controller on the yaw error -> a yaw RATE command, clamped to
        yaw_rate_max. Same philosophy as the position axes: PX4 only ever sees
        a bounded rate, so a setpoint-yaw jump can never spin it violently."""
        cp = self.current_pose
        if cp is None:
            return 0.0
        q = cp.pose.orientation
        cur = _math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        err = (desired - cur + _math.pi) % (2.0 * _math.pi) - _math.pi
        return max(-self.yaw_rate_max,
                   min(self.yaw_rate_max, self.yaw_kp * err))

    def _latch_takeoff_origin(self):
        """Called once at ARM: freeze the (now-converged) current position as
        the takeoff reference and arm the climb ramp. Takeoff is hover_z above
        wherever the drone actually is right now."""
        cp = self.current_pose
        if cp is None:
            return
        with self.lock:
            self.setpoint.pose.position.x = cp.pose.position.x
            self.setpoint.pose.position.y = cp.pose.position.y
            self.setpoint.pose.position.z = cp.pose.position.z
            self.setpoint.pose.orientation = cp.pose.orientation
            self.takeoff_z_target = cp.pose.position.z + self.takeoff_alt
            # VIO altitude at arm -> takeoff completion is the RELATIVE climb in
            # the VIO frame (VIO = true height; the EKF lags, so finishing on the
            # EKF overshoots by the lag). climb = vio_z - takeoff_vio_ref.
            self.takeoff_vio_ref = (self.vio_z if self.vio_z is not None
                                    else cp.pose.position.z)
            self.takeoff_ramp_active = True
            self.takeoff_ramp_t0 = None                  # soft-start clock resets
            self.int_x = self.int_y = self.int_z = 0.0   # fresh PID integrals
            self.last_vx = self.last_vy = self.last_vz = 0.0  # slew starts at 0
        rospy.loginfo("[offboard] takeoff origin latched at arm (%.2f,%.2f,%.2f); "
                      "target z=%.2f, ramp %.2f m/s",
                      cp.pose.position.x, cp.pose.position.y,
                      cp.pose.position.z, self.takeoff_z_target,
                      self.takeoff_climb_rate)

    def on_goal(self, msg):
        # /offboard/goal is the FINAL course goal (ego_bridge mirrors /next_goal
        # here). We deliberately DO NOT drive the PID setpoint to it: doing so
        # made the drone fly STRAIGHT to a far goal with NO obstacle avoidance
        # whenever EGO could not produce a cmd_raw (A* fail) -- it drove forward
        # into a low-texture region, SVO diverged, and it crashed. The drone now
        # moves ONLY along EGO's planned cmd_raw; if EGO stops planning, the
        # cmd_raw watchdog holds a hover at the CURRENT position. Goal is logged
        # for visibility only.
        if self.armed_at is None:
            return
        rospy.loginfo_throttle(
            2.0, "[offboard] course goal (%.2f,%.2f,%.2f) — followed via EGO "
            "cmd_raw, NOT driven directly (hover-hold if EGO can't plan)",
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def on_cmd_raw(self, msg):
        if self.armed_at is None:
            return
        held = (rospy.Time.now() - self.armed_at).to_sec()
        if held < self.settle_s:
            return
        with self.lock:
            # Let the takeoff finish on its own (VIO-altitude) condition; don't
            # let an early EGO cmd cut it short.
            if self.takeoff_ramp_active:
                return
            # DO NOT forward the EGO POSITION to PX4 (that makes PX4's position
            # loop chase the laggy/jumpy EKF and surge -- the takeoff problem).
            # Instead feed the EGO target into our OWN velocity-PID: the EGO
            # position becomes the PID setpoint and the EGO velocity becomes a
            # feedforward. The PID hold below then tracks it and emits a pure
            # VELOCITY setpoint -- so takeoff, hover AND course are all the same
            # velocity-control path and PX4 never runs its position loop.
            self.use_raw = False
            self.last_raw_t = rospy.Time.now()
            self.setpoint.pose.position.x = msg.position.x
            self.setpoint.pose.position.y = msg.position.y
            self.setpoint.pose.position.z = msg.position.z
            # face along the trajectory (EGO yaw) -> sp.orientation, which the
            # PID hold reads for the yaw setpoint.
            half = 0.5 * msg.yaw
            self.setpoint.pose.orientation.x = 0.0
            self.setpoint.pose.orientation.y = 0.0
            self.setpoint.pose.orientation.z = _math.sin(half)
            self.setpoint.pose.orientation.w = _math.cos(half)
            self.ego_ff = [msg.velocity.x, msg.velocity.y, msg.velocity.z]
        rospy.loginfo_throttle(1.0, "[offboard] new raw cmd: (%.2f,%.2f,%.2f)",
                               msg.position.x, msg.position.y,
                               msg.position.z)

    # ---------------- main loop ----------------
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        # warm up: stream setpoints before OFFBOARD switch
        rospy.loginfo("[offboard] streaming initial setpoints...")
        for _ in range(int(self.rate_hz * 2.0)):
            if rospy.is_shutdown():
                return
            self._publish()
            rate.sleep()

        last_req = rospy.Time(0)
        while not rospy.is_shutdown():
            now = rospy.Time.now()

            in_offboard = (self.state.mode == "OFFBOARD")

            # Arm as soon as the external controller switches to OFFBOARD.
            # We never call set_mode("OFFBOARD") ourselves — that stays with
            # the operator. But once OFFBOARD is active we arm immediately
            # so the drone is ready to follow setpoints without delay.
            if in_offboard and not self.state.armed:
                if not self._converged():
                    # OFFBOARD is on but the estimate isn't solid yet -> do
                    # NOT arm. This is the convergence gate that stops the
                    # drone from flying on a diverging estimate (spin/flip).
                    if (now - self.last_unstable_log) > rospy.Duration(1.5):
                        sp = "n/a" if self.cur_speed is None else f"{self.cur_speed:.2f}"
                        rospy.logwarn(
                            "[offboard] OFFBOARD set, WAITING for convergence "
                            "before arming (speed=%s m/s, need <%.2f for %.0fs)",
                            sp, self.converge_vel_thresh, self.converge_hold_s)
                        self.last_unstable_log = now
                elif (now - last_req) > rospy.Duration(0.5):
                    try:
                        res = self.srv_arm(True)
                        if res.success:
                            self.armed_at = rospy.Time.now()
                            self._latch_takeoff_origin()   # freeze converged pos as home
                            rospy.loginfo("[offboard] converged + OFFBOARD — armed")
                    except Exception as e:
                        rospy.logwarn(f"[offboard] arming failed: {e}")
                    last_req = now

            # Detect arming that happened before we connected (edge case).
            if self.armed_at is None and self.state.armed:
                self.armed_at = rospy.Time.now()
                self._latch_takeoff_origin()
                rospy.loginfo("[offboard] already armed on connect — settle timer started")

            # sim-only: also switch to OFFBOARD automatically when auto_arm=True.
            if self.auto_arm and not in_offboard:
                if (now - last_req) > rospy.Duration(2.0):
                    try:
                        res = self.srv_mode(custom_mode="OFFBOARD")
                        if res.mode_sent:
                            rospy.loginfo("[offboard] requested OFFBOARD (sim)")
                    except Exception as e:
                        rospy.logwarn(f"set_mode: {e}")
                    last_req = now

            # Gentle takeoff: convergence is already verified BEFORE arming
            # (the gate above), so once armed we ramp the z setpoint straight
            # up to the takeoff target at takeoff_climb_rate (m/s) -- gentle
            # enough that the climb never outruns the estimate.
            if (self.takeoff_ramp_active and self.state.armed
                    and self.takeoff_z_target is not None):
                with self.lock:
                    # SOFT START: ease the climb rate from 0 up to full over
                    # takeoff_soft_start_s so the instant of liftoff (motor
                    # spin-up + its vibration/feature-loss) is as gentle as
                    # possible and the climb never outruns the estimate.
                    if self.takeoff_ramp_t0 is None:
                        self.takeoff_ramp_t0 = rospy.Time.now()
                    elapsed = (rospy.Time.now() - self.takeoff_ramp_t0).to_sec()
                    soft = (1.0 if self.soft_start_s <= 0.0
                            else min(1.0, elapsed / self.soft_start_s))
                    dz = self.takeoff_climb_rate * soft / self.rate_hz
                    z = min(self.setpoint.pose.position.z + dz,
                            self.takeoff_z_target)
                    # FINISH on the ACTUAL climb (VIO), not the ramp timer. Pure
                    # velocity control climbs smoothly; when the drone has really
                    # risen takeoff_alt (in the VIO frame, which doesn't lag like
                    # the EKF), we stop and hold -> no infinite climb, no early
                    # hand-off yank.
                    climb = None
                    if self.vio_z is not None and self.takeoff_vio_ref is not None:
                        climb = self.vio_z - self.takeoff_vio_ref
                    elif self.current_pose is not None:
                        climb = (self.current_pose.pose.position.z
                                 - (self.takeoff_z_target - self.takeoff_alt))
                    # Stop a little early, in proportion to the climb speed, so
                    # the drone's upward momentum coasts it to ~target instead of
                    # overshooting (the velocity-PID hold then pins it there).
                    stop_margin = max(0.03, 0.5 * self.takeoff_climb_rate)
                    if climb is not None and climb >= self.takeoff_alt - stop_margin:
                        z = self.takeoff_z_target
                        self.takeoff_ramp_active = False
                        # Latch the CURRENT xy as the position-hold target.
                        if self.current_pose is not None:
                            self.setpoint.pose.position.x = self.current_pose.pose.position.x
                            self.setpoint.pose.position.y = self.current_pose.pose.position.y
                        rospy.loginfo("[offboard] takeoff complete: VIO climb=%.2f"
                                      " (target %.2f), position-hold engaged",
                                      climb, self.takeoff_alt)
                    self.setpoint.pose.position.z = z

            # cmd_raw watchdog: if EGO stopped feeding commands (course complete,
            # or a long replan/SAFETY stall), re-engage the tight custom PID hold
            # at the CURRENT position so the drone HOVERS (altitude-safe) instead
            # of coasting on the stale raw setpoint / sinking on PX4-native hold.
            if (self.use_raw and self.armed_at is not None
                    and self.last_raw_t is not None
                    and (now - self.last_raw_t).to_sec() > self.raw_timeout):
                with self.lock:
                    self.use_raw = False
                    cp = self.current_pose
                    if cp is not None:
                        self.setpoint.pose.position.x = cp.pose.position.x
                        self.setpoint.pose.position.y = cp.pose.position.y
                        self.setpoint.pose.position.z = cp.pose.position.z
                        self.setpoint.pose.orientation = cp.pose.orientation
                    self.int_x = self.int_y = self.int_z = 0.0
                    self.last_vx = self.last_vy = self.last_vz = 0.0
                rospy.loginfo_throttle(
                    2.0, "[offboard] cmd_raw stale %.1fs -> PID hover-hold",
                    (now - self.last_raw_t).to_sec())

            self._publish()
            rate.sleep()

    def _publish(self):
        """Stream the active setpoint to PX4.

        VELOCITY-ONLY policy: every live path below (takeoff / PID hold /
        fallback) publishes a raw setpoint EVERY cycle, so the OFFBOARD
        keepalive is satisfied by the raw stream alone. The POSITION setpoint
        (/mavros/setpoint_position/local) is published ONLY as a last-resort
        keepalive when there is NO pose at all (every raw path needs
        current_pose) -- otherwise the position stream would interleave with
        the velocity stream at 20 Hz and PX4 (which uses the most recent
        setpoint) would keep half-running its position loop, defeating the
        whole velocity-PID design.
        """
        with self.lock:
            sp = self.setpoint
            sp.header.stamp = rospy.Time.now()
            if not self.takeoff_ramp_active and self.current_pose is None:
                self.pub_sp.publish(sp)
            if self.use_raw:
                # cmd_raw publishes setpoint_raw/local at its own rate;
                # we just keep setpoint_position alive as a fallback.
                return

            # Build a setpoint_raw with position + capped velocity.
            cp = self.current_pose
            if cp is None:
                return
            tgt = PositionTarget()
            tgt.header.stamp = rospy.Time.now()
            tgt.header.frame_id = self.frame_id
            tgt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

            # TAKEOFF PHASE: trust SVO loosely (slam_assist level). Instead of
            # holding an XY POSITION (which makes PX4 chase a temporary SVO
            # divergence and fly away), we hold ZERO HORIZONTAL VELOCITY and
            # ignore XY position. The drone climbs straight up (Z position
            # ramp); if SVO momentarily flies off in XY there is no position
            # setpoint to chase, so the drone stays put and rides through it.
            # We switch to full XY position hold only after the climb ramp
            # completes (estimate has had time to stay solid).
            if self.takeoff_ramp_active:
                # VELOCITY-controlled climb (IGNORE_PZ) -- this gave the smooth
                # liftoff (no position error -> no windup pop). The earlier
                # "infinite climb" was only a bad completion check (timer-based);
                # we now FINISH when the ACTUAL altitude reaches the target (see
                # the ramp block), so it climbs smoothly then stops at 0.70.
                tgt.type_mask = (PositionTarget.IGNORE_PX
                                 | PositionTarget.IGNORE_PY
                                 | PositionTarget.IGNORE_PZ
                                 | PositionTarget.IGNORE_AFX
                                 | PositionTarget.IGNORE_AFY
                                 | PositionTarget.IGNORE_AFZ
                                 | PositionTarget.IGNORE_YAW)
                # Hold the takeoff XY with a gentle P+D position correction on
                # the latched takeoff origin, so the drone doesn't DRIFT FORWARD
                # during the climb. The old "zero xy velocity + ignore xy
                # position" let real drift accumulate -- worse the slower we
                # climb. Capped + velocity-damped so a VIO xy glitch can't yank.
                ex = self.setpoint.pose.position.x - cp.pose.position.x
                ey = self.setpoint.pose.position.y - cp.pose.position.y
                vmx, vmy = (self.vel[0], self.vel[1]) if self.vel else (0.0, 0.0)
                vx = self.kp_xy * ex - self.kd_xy * vmx
                vy = self.kp_xy * ey - self.kd_xy * vmy
                sxy = (vx * vx + vy * vy) ** 0.5
                if sxy > self.max_step_vel:
                    k = self.max_step_vel / sxy
                    vx *= k
                    vy *= k
                tgt.velocity.x = vx
                tgt.velocity.y = vy
                tgt.velocity.z = self.takeoff_climb_rate
                # Z-DIAG during the CLIMB too (was logged only in hold). Shows
                # VIO vs EKF z so we can see whether the EKF actually tracks the
                # altitude during takeoff (or stays ~0 while the drone climbs).
                if (self._last_zdiag_t is None
                        or (rospy.Time.now() - self._last_zdiag_t).to_sec() >= 0.5):
                    self._last_zdiag_t = rospy.Time.now()
                    rospy.loginfo(
                        "[Z-DIAG/TKO] target=%.2f ekf=%.2f vio=%s climb_cmd=%.2f",
                        self.takeoff_z_target, cp.pose.position.z,
                        ("%.2f" % self.vio_z) if self.vio_z is not None else "NA",
                        self.takeoff_climb_rate)
                q = sp.pose.orientation
                tgt.yaw_rate = self._yaw_rate_cmd(
                    _math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
                self.pub_sp_raw.publish(tgt)
                return

            q = sp.pose.orientation
            yaw = _math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            # ---- FULL PID HOLD: velocity on X, Y AND Z ----
            # Close the position loop on all three axes and send a VELOCITY
            # setpoint (position masked) so PX4 only runs its inner
            # velocity->thrust loop. velocity = Kp*err + Ki*∫err - Kd*vel_meas.
            # The D term uses the (good) VIO velocity to damp. Z is controlled
            # here too: VIO altitude is accurate, so an aggressive z-PID holds
            # it directly (Kp_z high so any sag => strong upward velocity cmd;
            # max_step_vel_z must be high enough to actually command recovery).
            if self.use_pid_hold and self.vel is not None:
                now = rospy.Time.now()
                dt = (0.05 if self._last_pid_t is None
                      else (now - self._last_pid_t).to_sec())
                dt = max(1e-3, min(dt, 0.2))
                self._last_pid_t = now
                ex = sp.pose.position.x - cp.pose.position.x
                ey = sp.pose.position.y - cp.pose.position.y
                ez = sp.pose.position.z - cp.pose.position.z
                vmx, vmy, vmz = self.vel
                self.int_x = max(-self.i_limit, min(self.i_limit, self.int_x + ex * dt))
                self.int_y = max(-self.i_limit, min(self.i_limit, self.int_y + ey * dt))
                self.int_z = max(-self.i_limit_z, min(self.i_limit_z, self.int_z + ez * dt))
                # EGO trajectory velocity feedforward (decays to 0 once EGO's
                # cmds go stale, so the same PID just holds the last setpoint as
                # a hover -- course end / replan gap both fall back cleanly).
                ff = self.ego_ff
                if (self.last_raw_t is None
                        or (now - self.last_raw_t).to_sec() > self.raw_timeout):
                    ff = (0.0, 0.0, 0.0)
                vx = self.kp_xy * ex + self.ki_xy * self.int_x - self.kd_xy * vmx + ff[0]
                vy = self.kp_xy * ey + self.ki_xy * self.int_y - self.kd_xy * vmy + ff[1]
                vz = self.kp_z * ez + self.ki_z * self.int_z - self.kd_z * vmz + ff[2]
                sxy = (vx * vx + vy * vy) ** 0.5
                if sxy > self.max_step_vel:
                    k = self.max_step_vel / sxy
                    vx *= k
                    vy *= k
                vz = max(-self.max_step_vel_z, min(self.max_step_vel_z, vz))
                # SLEW-RATE LIMIT: the commanded velocity may only change by
                # accel_lim*dt per cycle. Turns "팍" steps into smooth ramps and
                # breaks the bang-bang oscillation against the velocity clamps.
                dv = self.accel_lim * dt
                dvz = self.accel_lim_z * dt
                vx = max(self.last_vx - dv, min(self.last_vx + dv, vx))
                vy = max(self.last_vy - dv, min(self.last_vy + dv, vy))
                vz = max(self.last_vz - dvz, min(self.last_vz + dvz, vz))
                self.last_vx, self.last_vy, self.last_vz = vx, vy, vz
                # ---- Z DIAGNOSTIC (2 Hz) ----
                # Decides WHY altitude isn't held, without guessing gains:
                #  sp   = our held altitude setpoint
                #  ekf  = EKF z (what PX4 actually controls on)  [cp]
                #  vio  = raw VIO z (what the camera sees)
                #  ekf_vz = EKF vertical velocity (LP)            [vmz]
                #  cmd_vz = velocity we command up(+)/down(-)
                # READ: if cmd_vz>0 (up) yet the drone sinks -> PX4 THRUST issue.
                #       if ekf tracks sp (ez~0) while vio drops/drone sinks
                #       -> EKF z DRIFT (estimate), gains can't help.
                if (self._last_zdiag_t is None
                        or (now - self._last_zdiag_t).to_sec() >= 0.5):
                    self._last_zdiag_t = now
                    rospy.loginfo(
                        "[Z-DIAG] sp=%.2f ekf=%.2f vio=%s ez=%+.2f "
                        "ekf_vz=%+.2f cmd_vz=%+.2f",
                        sp.pose.position.z, cp.pose.position.z,
                        ("%.2f" % self.vio_z) if self.vio_z is not None else "NA",
                        ez, vmz, vz)
                tgt.type_mask = (PositionTarget.IGNORE_PX
                                 | PositionTarget.IGNORE_PY
                                 | PositionTarget.IGNORE_PZ
                                 | PositionTarget.IGNORE_AFX
                                 | PositionTarget.IGNORE_AFY
                                 | PositionTarget.IGNORE_AFZ
                                 | PositionTarget.IGNORE_YAW)
                tgt.velocity.x = vx
                tgt.velocity.y = vy
                tgt.velocity.z = vz
                # yaw as a RATE too (P on error, clamped +-yaw_rate_max)
                tgt.yaw_rate = self._yaw_rate_cmd(yaw)
                self.pub_sp_raw.publish(tgt)
                return

            # ---- FALLBACK (use_pid_hold:=false): position setpoint + capped
            # velocity feedforward when slewing, pure position when hovering. ----
            tgt.type_mask = (PositionTarget.IGNORE_AFX
                             | PositionTarget.IGNORE_AFY
                             | PositionTarget.IGNORE_AFZ
                             | PositionTarget.IGNORE_YAW)
            tgt.position.x = sp.pose.position.x
            tgt.position.y = sp.pose.position.y
            tgt.position.z = sp.pose.position.z
            dx = sp.pose.position.x - cp.pose.position.x
            dy = sp.pose.position.y - cp.pose.position.y
            dz = sp.pose.position.z - cp.pose.position.z
            err = (dx * dx + dy * dy + dz * dz) ** 0.5
            if err > self.hold_radius_m:
                dxy = (dx * dx + dy * dy) ** 0.5
                if dxy > 1e-3:
                    k_xy = min(1.0, self.max_step_vel / dxy)
                    tgt.velocity.x = dx * k_xy
                    tgt.velocity.y = dy * k_xy
                if abs(dz) > 1e-3:
                    tgt.velocity.z = max(-self.max_step_vel_z,
                                         min(self.max_step_vel_z, dz * 2.0))
            tgt.yaw_rate = self._yaw_rate_cmd(yaw)
            self.pub_sp_raw.publish(tgt)


if __name__ == "__main__":
    try:
        OffboardController().spin()
    except rospy.ROSInterruptException:
        pass
