cmake_minimum_required(VERSION 3.0.2)
project(ee478_agent)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package(CATKIN_DEPENDS rospy)

catkin_install_python(PROGRAMS
  scripts/command_interpreter_node.py
  scripts/quiz_solver_node.py
  scripts/signature_move_node.py
  scripts/mission_fsm_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
