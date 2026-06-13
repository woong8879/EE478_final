cmake_minimum_required(VERSION 3.0.2)
project(ee478_localization)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package(CATKIN_DEPENDS rospy)

catkin_install_python(PROGRAMS
  scripts/svo_vision_relay_node.py
  scripts/odom_tf_broadcaster_node.py
  scripts/camera_pose_publisher_node.py
  scripts/apriltag_anchor_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
