cmake_minimum_required(VERSION 3.0.2)
project(ee478_planner)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package(CATKIN_DEPENDS rospy)

catkin_install_python(PROGRAMS
  scripts/offboard_controller.py
  scripts/ego_bridge_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
