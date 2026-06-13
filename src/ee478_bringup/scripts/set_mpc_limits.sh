#!/usr/bin/env bash
# set_mpc_limits.sh — cap PX4 MPC velocity/accel so the ACTUAL drone motion is
# gentle.
#
# WHY: EGO's max_vel only limits the PLANNED trajectory. The drone's real motion
# (and its CORRECTIONS when the VIO estimate jumps) is governed by PX4's position
# controller limits MPC_*_VEL_MAX / MPC_ACC_* -- whose defaults (~12 m/s horiz,
# ~3 m/s climb) let the drone lurch aggressively and overshoot on takeoff. We cap
# them low so the drone physically creeps even while correcting a VIO jump.
#
# Runs once after mavros connects; PX4 saves the params to FC flash (persistent).
# Args: optional overrides "xy_vel z_vel_up z_vel_dn acc_hor".
set +e
XYV="${1:-0.5}"; ZVU="${2:-0.5}"; ZVD="${3:-0.5}"; ACH="${4:-0.5}"

# wait for the mavros param plugin to be up
for _ in $(seq 1 90); do
  rosservice list 2>/dev/null | grep -q "/mavros/param/set" && break
  sleep 1
done
sleep 5   # let mavros finish its initial param sync before we set

setp(){ rosrun mavros mavparam set "$1" "$2" >/dev/null 2>&1 && echo "[mpc] $1=$2" || echo "[mpc] FAILED $1 (retry once)"; }
retry(){ setp "$1" "$2"; rosrun mavros mavparam get "$1" >/dev/null 2>&1 || { sleep 2; setp "$1" "$2"; }; }

retry MPC_XY_VEL_MAX   "$XYV"   # horizontal real max speed (and correction cap)
retry MPC_Z_VEL_MAX_UP "$ZVU"   # climb cap -> kills takeoff overshoot
retry MPC_Z_VEL_MAX_DN "$ZVD"   # descent cap
retry MPC_ACC_HOR      "$ACH"   # horizontal accel -> gentle
retry MPC_ACC_UP_MAX   0.8
retry MPC_ACC_DOWN_MAX 0.8
retry MPC_JERK_MAX     4.0      # lower jerk -> smoother
echo "[mpc] gentle MPC limits applied (xy=$XYV z_up=$ZVU z_dn=$ZVD acc_hor=$ACH); saved to FC flash"
