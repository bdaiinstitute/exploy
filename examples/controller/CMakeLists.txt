# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

cmake_minimum_required(VERSION 3.20)
project(loopback_controller_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXPLOY_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    CACHE PATH "Path to the exploy project root")

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(
    "${EXPLOY_ROOT_DIR}/control"
    "${CMAKE_CURRENT_BINARY_DIR}/exploy_control"
    EXCLUDE_FROM_ALL
)

add_executable(loopback_controller_example
    main.cpp
)

target_include_directories(loopback_controller_example PRIVATE
    # The example's own headers (loopback_state_interface.hpp etc.)
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(loopback_controller_example PRIVATE
    exploy
)

target_compile_options(loopback_controller_example PRIVATE
    -Wall -Wextra -Wunused-variable -Wunused-parameter
)
