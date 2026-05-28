#!/usr/bin/env bash
# Source me at the start of every shell that runs nodes in this workspace:
#   source ~/EE478/final_project_ws/setup_env.sh

# ROS Noetic
source /opt/ros/noetic/setup.bash

# Our catkin workspace
if [ -f "$(dirname "$BASH_SOURCE")/devel/setup.bash" ]; then
  source "$(dirname "$BASH_SOURCE")/devel/setup.bash"
fi

# PX4 SITL + Gazebo Classic (sim only; on the real drone this block is a no-op)
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
if [ -d "$PX4_DIR" ] && [ -d "$PX4_DIR/build/px4_sitl_default" ]; then
  source "$PX4_DIR/Tools/simulation/gazebo-classic/setup_gazebo.bash" \
    "$PX4_DIR" "$PX4_DIR/build/px4_sitl_default" >/dev/null 2>&1
  export ROS_PACKAGE_PATH="$ROS_PACKAGE_PATH:$PX4_DIR:$PX4_DIR/Tools/simulation/gazebo-classic"
fi
export PX4_HOME_LAT=47.397742
export PX4_HOME_LON=8.545594
export PX4_HOME_ALT=488.0

# Gazebo OpenGL context — needed even with gui:=false
export DISPLAY=${DISPLAY:-:0}

# OPENAI_API_KEY: read from external file (NOT committed)
APIKEY_FILE="$HOME/.openai_key"
if [ -f "$APIKEY_FILE" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  export OPENAI_API_KEY="$(cat "$APIKEY_FILE")"
fi
