#!/usr/bin/env bash
# set_ir_exposure.sh — force a FIXED SHORT IR exposure on the D435i after the
# RealSense node is up.
#
# WHY a script (not <param>): on this realsense2_camera build the static
# stereo_module/* ROS params are IGNORED — the camera boots in AUTO exposure
# (~23 ms), which motor vibration smears into the IR image so SVO loses
# tracked features by the dozen and the VIO diverges on takeoff. The ONLY
# thing that actually takes is the dynamic_reconfigure server
# /camera/stereo_module, and only once the node has finished initialising.
# So we wait for that server, then push a short FIXED exposure with
# auto-exposure OFF. 6.5 ms freezes the motion; the IR projector dots stay
# bright enough to track.
#
# Args: [exposure_us] [gain]   (defaults 6500 32)
EXP="${1:-6500}"
GAIN="${2:-32}"
for _ in $(seq 1 40); do
  rosservice list 2>/dev/null | grep -q "/camera/stereo_module/set_parameters" && break
  sleep 1
done
sleep 2   # let the sensor finish opening before reconfiguring
rosrun dynamic_reconfigure dynparam set /camera/stereo_module \
  "{'enable_auto_exposure': False, 'exposure': ${EXP}, 'gain': ${GAIN}}" \
  && echo "[set_ir_exposure] IR exposure fixed: ${EXP}us gain ${GAIN}, auto OFF" \
  || echo "[set_ir_exposure] FAILED to set IR exposure"
# Signal that the IR exposure is fixed. svo_exposure_gate.sh blocks SVO until
# this is true, so SVO initialises its map on SHARP frames (no blurry-init
# jump / baseline reset / wrong start altitude).
rosparam set /svo/exposure_ready true
