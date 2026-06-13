#!/usr/bin/env bash
# Source me at the start of every shell that runs nodes in this workspace:
#   source ~/EE478/final_project_ws/setup_env.sh

# ROS Noetic
source /opt/ros/noetic/setup.bash

# ROS networking: advertise the routable IP (NOT the hostname), else a ground
# laptop's RViz can't resolve "team5-desktop" and fails to connect. Picks the
# source IP of the default route (the WiFi/LAN IP), auto-updates on DHCP.
# Override by exporting ROS_IP before sourcing (e.g. USB-tether 192.168.55.1).
if [ -z "${ROS_IP:-}" ]; then
  _ros_ip=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)
  [ -n "$_ros_ip" ] && export ROS_IP="$_ros_ip"
fi

# Vendored realsense-ros (realsense2_camera) lives in ~/catkin_ws. Our
# workspace is built on top of it, so sourcing our devel below chains
# back here; this explicit line is a fallback for fresh/clean trees.
CATKIN_WS="${CATKIN_WS:-$HOME/catkin_ws}"
if [ -f "$CATKIN_WS/devel/setup.bash" ]; then
  source "$CATKIN_WS/devel/setup.bash"
fi

# Our catkin workspace (overlays ~/catkin_ws when built on top of it)
if [ -f "$(dirname "$BASH_SOURCE")/devel/setup.bash" ]; then
  source "$(dirname "$BASH_SOURCE")/devel/setup.bash" --extend
fi

# SVO Pro workspace overlay (svo_ros etc.) so estimator:=svo can find svo_node.
# Present only once svo_ws is built; harmless otherwise.
if [ -f "$HOME/svo_ws/devel/setup.bash" ]; then
  source "$HOME/svo_ws/devel/setup.bash" --extend
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
