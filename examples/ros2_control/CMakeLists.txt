cmake_minimum_required(VERSION 3.20)
project(ros2_control_example)

find_package(my_robot_lib REQUIRED) # This would usually be installed

add_executable(ros2_example_node example_controller.cpp)
target_link_libraries(ros2_example_node PRIVATE my_robot_lib)
