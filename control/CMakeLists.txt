cmake_minimum_required(VERSION 3.20)
project(my_robot_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
# Find ONNX Runtime manually since it might not provide a CMake config
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    PATH_SUFFIXES onnxruntime/core/session onnxruntime
)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Include: ${ONNXRUNTIME_INCLUDE_DIR}, Library: ${ONNXRUNTIME_LIBRARY}")
endif()

# Create imported target for onnxruntime
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}/../../"
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
)
# Note: INTERFACE_INCLUDE_DIRECTORIES needs to point to the root of headers so #include <onnxruntime/...> works
# Wait, let's check the find_path result. If it finds onnxruntime_c_api.h in include/onnxruntime/core/session, 
# then header root is include/.
find_package(GTest REQUIRED)

# Include directory
include_directories(/usr/include/eigen3) # Fallback if FindEigen3 is weird, but usually Eigen3::Eigen is better

# C++ Shared Library
add_library(onnx_controller SHARED 
    onnx_controller.cpp
    onnx_config_parser.cpp
    onnx_runtime.cpp
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

add_executable(onnx_controller_test test/onnx_controller_test.cpp "${GEN_OUTPUT}")
target_compile_definitions(onnx_controller_test PRIVATE 
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/testdata"
)
target_link_libraries(onnx_controller_test PRIVATE 
    onnx_controller 
    GTest::gtest_main 
    GTest::gmock 
    fmt::fmt
)
add_test(NAME onnx_controller_test COMMAND onnx_controller_test)
