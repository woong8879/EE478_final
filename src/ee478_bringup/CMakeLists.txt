cmake_minimum_required(VERSION 3.0.2)
project(ee478_bringup)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package(CATKIN_DEPENDS rospy)

catkin_install_python(PROGRAMS
  scripts/px4_param_setter.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
