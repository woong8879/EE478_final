cmake_minimum_required(VERSION 3.0.2)
project(gpt_llm_client)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

add_service_files(
  FILES
  LLMQuery.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs message_runtime
)

catkin_install_python(
  PROGRAMS
    scripts/simple_llm_client.py
    scripts/memory_llm_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
