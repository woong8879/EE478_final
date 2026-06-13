cmake_minimum_required(VERSION 3.0.2)
project(ee478_agent)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package(CATKIN_DEPENDS rospy)

catkin_install_python(PROGRAMS
  scripts/final_mission_node.py
  scripts/precision_land_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
