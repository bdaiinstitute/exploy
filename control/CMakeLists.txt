# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

cmake_minimum_required(VERSION 3.20)
project(exploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Treat warnings as errors
add_compile_options(-Werror -Wpedantic -Wunused-variable -Wunused-parameter -Wreorder)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(GTest REQUIRED)

# C++ Shared Library
add_library(exploy SHARED
    controller.cpp
    context.cpp
    onnx_runtime.cpp
    components.cpp
    matcher.cpp
)

target_include_directories(exploy PUBLIC .)

target_link_libraries(exploy PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    fmt::fmt
    onnxruntime
)

# Tests
set(GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/test/testdata/test_onnx_generator.py")
set(GEN_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/test/testdata/test.onnx")

add_custom_command(
    OUTPUT "${GEN_OUTPUT}"
    COMMAND ${CMAKE_COMMAND} -E env PYTHONWARNINGS="ignore" python3 "${GEN_SCRIPT}" "${GEN_OUTPUT}"
    DEPENDS "${GEN_SCRIPT}"
    COMMENT "Generating test ONNX model..."
)

add_executable(exploy_test
    test/controller_test.cpp
    test/components_test.cpp
    test/context_test.cpp
    test/logging_test.cpp
    test/onnx_runtime_test.cpp
    "${GEN_OUTPUT}"
)
target_compile_definitions(exploy_test PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/testdata"
)
target_link_libraries(exploy_test PRIVATE
    exploy
    GTest::gtest_main
    GTest::gmock
    fmt::fmt
)
add_test(NAME exploy_test COMMAND exploy_test)
