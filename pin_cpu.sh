#!/usr/bin/env bash
# pin_cpu.sh — OPTIONAL extra hardening. You normally do NOT need to run this:
# the launch files already auto-pin the heavy nodes (VINS, vio_bridge,
# cloud nodelet, offboard) to cores 2-5 via `launch-prefix="taskset -c 2-5"`,
# which leaves cores 0,1 free so the RealSense camera nodelet keeps them to
# itself (root-cause fix for RGB/IR stream drops = nodelet CPU starvation).
#
# Run this ONLY to additionally pin EXPLICITLY the two processes that live
# inside includes and can't take a launch-prefix:
#   cores 0,1  -> RealSense camera nodelet (force it onto its reserved cores)
#   cores 2-5  -> MAVROS (keep it off the camera cores too)
# No sudo needed (own-process affinity). Re-run after restarting those nodes.
#   bash ~/EE478_final/pin_cpu.sh
set -u
CAM_CORES="0,1"
REST_CORES="2-5"

pin() {  # $1 = pgrep pattern, $2 = core list
  local n=0
  for pid in $(pgrep -f "$1" 2>/dev/null); do
    if taskset -a -p -c "$2" "$pid" >/dev/null 2>&1; then
      echo "    pid $pid -> $2"; n=$((n+1))
    fi
  done
  [ "$n" -eq 0 ] && echo "    (none found for '$1')"
}

echo "=== camera nodelet -> cores $CAM_CORES (PROTECTED) ==="
pin "realsense2_camera_manager" "$CAM_CORES"

echo "=== everything else -> cores $REST_CORES ==="
for p in vins_node cloud_nodelet_manager mavros_node vio_bridge_node \
         offboard_controller direct_goal_follower ego_planner; do
  echo "  $p:"; pin "$p" "$REST_CORES"
done

echo
echo "verify:  ps -eo pid,psr,comm,%cpu | grep -E 'realsense|vins|nodelet|mavros'"
echo "(psr column = core the thread last ran on)"
