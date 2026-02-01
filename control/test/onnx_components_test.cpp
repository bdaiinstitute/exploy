// Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "onnx_components.hpp"
#include "mock_command_interface.hpp"
#include "mock_state_interface.hpp"
#include "onnx_runtime.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <filesystem>
#include <span>
#include <vector>

namespace rai::cs::control::common::onnx {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

class OnnxComponentsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a minimal test ONNX model for testing
    std::string test_model_path = (std::filesystem::path(TEST_DATA_DIR) / "test.onnx").string();
    ASSERT_TRUE(runtime.initialize(test_model_path));
  }

  NiceMock<MockRobotStateInterface> state_mock_;
  NiceMock<MockCommandInterface> command_mock_;
  OnnxRuntime runtime;
};

TEST_F(OnnxComponentsTest, JointPositionInput_InitAndRead) {
  // Arrange
  std::vector<std::string> joint_names = {"joint1", "joint2", "joint3"};
  JointPositionInput joint_input("articulation.joint.pos", joint_names);

  // Test initialization
  EXPECT_CALL(state_mock_, initJointPosition("joint1")).WillOnce(Return(true));
  EXPECT_CALL(state_mock_, initJointPosition("joint2")).WillOnce(Return(true));
  EXPECT_CALL(state_mock_, initJointPosition("joint3")).WillOnce(Return(true));
  EXPECT_TRUE(joint_input.init(state_mock_, command_mock_));

  // Test read functionality (interface calls, even without ONNX buffer)
  EXPECT_CALL(state_mock_, jointPosition("joint1")).WillOnce(Return(1.5));
  EXPECT_CALL(state_mock_, jointPosition("joint2")).WillOnce(Return(2.0));
  EXPECT_CALL(state_mock_, jointPosition("joint3")).WillOnce(Return(0.5));

  // Read should call the interface methods even if ONNX buffer doesn't exist
  ASSERT_TRUE(joint_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BasePositionInput_InitAndRead) {
  BasePositionInput base_input("articulation.bodies.pelvis.pos_base_in_w");

  // Test successful initialization
  EXPECT_CALL(state_mock_, initBasePosW()).WillOnce(Return(true));
  EXPECT_TRUE(base_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Vector3d expected_pos(1.0, 2.0, 3.0);
  EXPECT_CALL(state_mock_, basePosW()).WillOnce(Return(expected_pos));

  ASSERT_TRUE(base_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BasePositionInput_InitFailure) {
  BasePositionInput base_input("base_pos");
  EXPECT_CALL(state_mock_, initBasePosW()).WillOnce(Return(false));
  EXPECT_FALSE(base_input.init(state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BaseOrientationInput_InitAndRead) {
  BaseOrientationInput base_input("articulation.bodies.pelvis.world_Q_body");

  // Test initialization
  EXPECT_CALL(state_mock_, initBaseQuatW()).WillOnce(Return(true));
  EXPECT_TRUE(base_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Quaterniond expected_quat(1.0, 0.0, 0.0, 0.0);
  EXPECT_CALL(state_mock_, baseQuatW()).WillOnce(Return(expected_quat));

  ASSERT_TRUE(base_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BaseLinearVelocityInput_InitAndRead) {
  BaseLinearVelocityInput base_input("articulation.bodies.pelvis.lin_vel_base_in_base");

  // Test initialization
  EXPECT_CALL(state_mock_, initBaseLinVelB()).WillOnce(Return(true));
  EXPECT_TRUE(base_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Vector3d expected_vel(0.5, 0.0, 0.0);
  EXPECT_CALL(state_mock_, baseLinVelB()).WillOnce(Return(expected_vel));

  ASSERT_TRUE(base_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BaseAngularVelocityInput_InitAndRead) {
  BaseAngularVelocityInput base_input("articulation.bodies.pelvis.ang_vel_base_in_base");

  // Test initialization
  EXPECT_CALL(state_mock_, initBaseAngVelB()).WillOnce(Return(true));
  EXPECT_TRUE(base_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Vector3d expected_ang_vel(0.0, 0.0, 0.1);
  EXPECT_CALL(state_mock_, baseAngVelB()).WillOnce(Return(expected_ang_vel));

  ASSERT_TRUE(base_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, JointVelocityInput_InitAndRead) {
  std::vector<std::string> joint_names = {"joint1", "joint2", "joint3"};
  JointVelocityInput joint_input("articulation.joint.vel", joint_names);

  // Test initialization
  EXPECT_CALL(state_mock_, initJointVelocity("joint1")).WillOnce(Return(true));
  EXPECT_CALL(state_mock_, initJointVelocity("joint2")).WillOnce(Return(true));
  EXPECT_CALL(state_mock_, initJointVelocity("joint3")).WillOnce(Return(true));
  EXPECT_TRUE(joint_input.init(state_mock_, command_mock_));

  // Test read functionality
  EXPECT_CALL(state_mock_, jointVelocity("joint1")).WillOnce(Return(0.5));
  EXPECT_CALL(state_mock_, jointVelocity("joint2")).WillOnce(Return(-0.3));
  EXPECT_CALL(state_mock_, jointVelocity("joint3")).WillOnce(Return(0.1));

  ASSERT_TRUE(joint_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BodyOrientationInput_InitAndRead) {
  BodyOrientationInput body_input("rigid_bodies.box.body_Q_body", "test_body");

  // Test initialization
  EXPECT_CALL(state_mock_, initBodyOrientationW("test_body")).WillOnce(Return(true));
  EXPECT_TRUE(body_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Quaterniond expected_quat(0.707, 0.0, 0.0, 0.707);
  EXPECT_CALL(state_mock_, bodyOrientationW("test_body")).WillOnce(Return(expected_quat));

  EXPECT_TRUE(body_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, StepCountInput_WithRealRuntime) {
  // Arrange
  StepCountInput step_input("step_count");

  // Get initial buffer state
  auto buffer = runtime.inputBuffer<int32_t>("step_count");
  ASSERT_TRUE(buffer.has_value()) << "step_count input should exist in test model";

  int32_t initial_value = buffer.value()[0];

  // Act & Assert: First increment
  ASSERT_TRUE(step_input.read(runtime, state_mock_, command_mock_));
  ASSERT_TRUE(buffer.has_value());
  EXPECT_EQ(buffer.value()[0], initial_value + 1) << "Step count should be incremented by 1";

  // Act & Assert: Second increment
  ASSERT_TRUE(step_input.read(runtime, state_mock_, command_mock_));
  ASSERT_TRUE(buffer.has_value());
  EXPECT_EQ(buffer.value()[0], initial_value + 2) << "Step count should be incremented to 2";

  // Act & Assert: Third increment
  ASSERT_TRUE(step_input.read(runtime, state_mock_, command_mock_));
  ASSERT_TRUE(buffer.has_value());
  EXPECT_EQ(buffer.value()[0], initial_value + 3) << "Step count should be incremented to 3";
}

// Integration test for MemoryOutput
TEST_F(OnnxComponentsTest, MemoryOutput_WithRealRuntime) {
  // Arrange
  MemoryOutput memory_output("output.joint_targets.pos");

  // Set initial memory data
  auto buffer = runtime.inputBuffer<float>("memory.output.joint_targets.pos.in");
  ASSERT_TRUE(buffer.has_value()) << "memory input should exist in test model";

  // Set some test values in the input buffer
  std::span<float> input_span = buffer.value();
  for (size_t i = 0; i < input_span.size(); ++i) {
    input_span[i] = static_cast<float>(i + 100);  // Set to 100, 101, 102, etc.
  }

  // Get initial values for comparison
  std::vector<float> initial_values(input_span.begin(), input_span.end());

  // Act: Run inference to generate output, then copy output to input
  ASSERT_TRUE(runtime.evaluate());
  bool result = memory_output.write(runtime, state_mock_, command_mock_);
  ASSERT_TRUE(result) << "Memory write should succeed";

  // Assert: Verify memory was overwritten
  ASSERT_TRUE(buffer.has_value());
  std::span<float> updated_span = buffer.value();

  // Memory should have been overwritten with output data
  bool memory_was_overwritten = false;
  for (size_t i = 0; i < updated_span.size() && i < initial_values.size(); ++i) {
    if (updated_span[i] != initial_values[i]) {
      memory_was_overwritten = true;
      break;
    }
  }

  EXPECT_TRUE(memory_was_overwritten) << "Memory should be overwritten with new output data";
}

}  // namespace rai::cs::control::common::onnx
