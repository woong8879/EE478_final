#!/bin/bash
# review_flight.sh [bag]  — play a recorded flight + open the all-in-one RViz.
#
# Run on the GROUND laptop after copying a bag + flight_review.rviz next to this
# script (e.g. into ~/flight_logs/). With no arg it plays the NEWEST core bag.
#
#   ./review_flight.sh                         # newest flight_*.bag
#   ./review_flight.sh ~/flight_logs/flight_2026-06-11-00-56-50_0.bag
#
# Controls during playback:  SPACE = pause/resume,  -> = step (when paused).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BAG="${1:-$(ls -t "$HERE"/flight_2026*.bag ~/flight_logs/flight_2026*.bag 2>/dev/null | head -1)}"
RVIZ="$HERE/flight_review.rviz"
[ -f "$RVIZ" ] || RVIZ="$(ls "$HERE"/*.rviz 2>/dev/null | head -1)"
[ -f "$BAG" ]  || { echo "no bag found (pass one as arg)"; exit 1; }
[ -f "$RVIZ" ] || { echo "no flight_review.rviz next to this script"; exit 1; }

source /opt/ros/noetic/setup.bash 2>/dev/null || true
# Local master for offline review (don't touch the drone's master).
export ROS_MASTER_URI=http://localhost:11311
unset ROS_IP

pgrep -x rosmaster >/dev/null 2>&1 || { roscore >/tmp/roscore_review.log 2>&1 & sleep 3; }
rosparam set use_sim_time true
echo "RViz: $RVIZ"
rviz -d "$RVIZ" >/tmp/rviz_review.log 2>&1 &
# intended course waypoints in absolute/map coords (latched) -> drift comparison
[ -f "$HERE/waypoint_markers.py" ] && python3 "$HERE/waypoint_markers.py" >/tmp/wpmarkers.log 2>&1 &
sleep 2
echo "Playing: $BAG"
echo "  SPACE = pause/resume,  ->  = step frame (while paused)"
rosbag play --clock --pause "$BAG"
