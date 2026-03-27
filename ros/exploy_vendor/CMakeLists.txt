# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
cmake_minimum_required(VERSION 3.16)
project(exploy_vendor)

# Load macros for ament packages.
find_package(ament_cmake REQUIRED)

# Find the wrapped package.
find_package(exploy REQUIRED CONFIG)

# Create an interface library that wraps the imported targets.
add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE exploy::exploy)

# Create an alias target for easier usage in downstream packages.
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Install and export.
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets)
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)

# Export exploy as a dependency for downstream packages.
ament_export_dependencies(exploy)

# Generate package configuration files for downstream packages to find this package.
ament_package()
