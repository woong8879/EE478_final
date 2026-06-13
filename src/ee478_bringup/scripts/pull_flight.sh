#!/bin/bash
# pull_flight.sh — run on the GROUND laptop. Pulls the newest flight bag(s) +
# the all-in-one RViz config + review script FROM THE DRONE, then opens RViz.
#
#   ./pull_flight.sh        # core bag (pose / nav / obstacle map) -- fast
#   ./pull_flight.sh img    # ALSO pull the image bag (large, slow)
#
# (Asks for the drone's team5 password once.)
set -e
JETSON="${JETSON:-team5@10.249.25.180}"
DST=~/flight_logs
mkdir -p "$DST"
echo "Pulling from $JETSON ..."
scp "$JETSON:~/flight_logs/flight_2026*.bag" \
    "$JETSON:~/flight_logs/flight_review.rviz" \
    "$JETSON:~/flight_logs/review_flight.sh" \
    "$JETSON:~/flight_logs/waypoint_markers.py" "$DST"/
if [ "$1" = "img" ]; then
  echo "Pulling image bag (large) ..."
  scp "$JETSON:~/flight_logs/flight_img_*.bag" "$DST"/
fi
chmod +x "$DST/review_flight.sh"
echo "Opening review ..."
exec "$DST/review_flight.sh"
