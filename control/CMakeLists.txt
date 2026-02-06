cmake_minimum_required(VERSION 3.20)
project(my_robot_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Treat warnings as errors
add_compile_options(-Werror -Wpedantic -Wunused-variable -Wunused-parameter -Wreorder)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(GTest REQUIRED)

# C++ Shared Library
add_library(onnx_controller SHARED
    onnx_controller.cpp
    onnx_context.cpp
    onnx_runtime.cpp
    onnx_components.cpp
    onnx_matcher.cpp
)

target_include_directories(onnx_controller PUBLIC .)

target_link_libraries(onnx_controller PUBLIC
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
    COMMAND python3 "${GEN_SCRIPT}" "${GEN_OUTPUT}"
    DEPENDS "${GEN_SCRIPT}"
    COMMENT "Generating test ONNX model..."
)

add_executable(control_test
    test/onnx_controller_test.cpp
    test/onnx_components_test.cpp
    test/onnx_context_test.cpp
    test/onnx_runtime_test.cpp
    "${GEN_OUTPUT}"
)
target_compile_definitions(control_test PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/testdata"
)
target_link_libraries(control_test PRIVATE
    onnx_controller
    GTest::gtest_main
    GTest::gmock
    fmt::fmt
)
add_test(NAME control_test COMMAND control_test)
