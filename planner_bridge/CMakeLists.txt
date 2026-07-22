cmake_minimum_required(VERSION 3.0.2)
project(vitfly_planner_bridge)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  cv_bridge
)

catkin_package(CATKIN_DEPENDS geometry_msgs nav_msgs sensor_msgs std_msgs tf2 tf2_ros cv_bridge)

catkin_install_python(PROGRAMS
  scripts/planner_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
