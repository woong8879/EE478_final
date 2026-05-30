cmake_minimum_required(VERSION 3.0.2)
project(ee478_perception)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package(CATKIN_DEPENDS rospy)

catkin_install_python(PROGRAMS
  scripts/sim_world_publisher_node.py
  scripts/world_cloud_publisher_node.py
  scripts/store_identifier_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
