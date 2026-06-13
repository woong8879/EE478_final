#!/usr/bin/env bash
# svo_exposure_gate.sh — launch-prefix that BLOCKS SVO until the IR exposure
# has been fixed, then execs the real node command ("$@").
#
# WHY: the camera boots in AUTO exposure (~23 ms, blurry); set_ir_exposure.sh
# only switches it to the fixed short exposure AFTER the camera's reconfigure
# server is up. If SVO starts at the same time it initialises its map on the
# blurry frames -> big startup jump -> the relay resets its baseline to that
# jumped pose -> the drone's start position/altitude is wrong. Forcing the
# order (exposure first, THEN SVO) makes SVO init on sharp frames = clean.
#
# It waits for the /svo/exposure_ready flag (set by set_ir_exposure.sh) with a
# hard timeout so SVO still starts even if the exposure step fails (degraded,
# but better than never localising).
TIMEOUT_S="${SVO_GATE_TIMEOUT_S:-30}"
t=0
while [ "$(rosparam get /svo/exposure_ready 2>/dev/null)" != "true" ]; do
  sleep 0.5
  t=$(awk "BEGIN{print $t+0.5}")
  if awk "BEGIN{exit !($t >= $TIMEOUT_S)}"; then
    echo "[svo_gate] exposure_ready not seen in ${TIMEOUT_S}s; starting SVO anyway"
    break
  fi
done
[ "$(rosparam get /svo/exposure_ready 2>/dev/null)" = "true" ] && \
  echo "[svo_gate] IR exposure ready -> starting SVO on sharp frames"
exec "$@"
