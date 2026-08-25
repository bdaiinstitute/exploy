// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "exploy/components.hpp"
#include "exploy/matcher.hpp"
#include "exploy/onnx_runtime.hpp"
#include "mock_command_interface.hpp"
#include "mock_state_interface.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <filesystem>
#include <span>
#include <vector>

namespace exploy::control {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

class OnnxComponentsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a minimal test ONNX model for testing
    std::string test_model_path =
        (std::filesystem::path(TEST_DATA_DIR) / "test_export.onnx").string();
    ASSERT_TRUE(runtime.initialize(test_model_path));
  }

  NiceMock<MockRobotStateInterface> state_mock_;
  NiceMock<MockCommandInterface> command_mock_;
  OnnxRuntime runtime;
};

TEST_F(OnnxComponentsTest, JointPositionInput_InitAndRead) {
  // Arrange
  std::vector<std::string> joint_names = {"joint1", "joint2", "joint3"};
  JointPositionInput joint_input("obj.robot1.joints.pos", "robot1", joint_names);

  // Test initialization
  EXPECT_CALL(state_mock_, initJointPosition(JointIs<JointPositionInfo>("robot1", "joint1")))
      .WillOnce(Return(true));
  EXPECT_CALL(state_mock_, initJointPosition(JointIs<JointPositionInfo>("robot1", "joint2")))
      .WillOnce(Return(true));
  EXPECT_CALL(state_mock_, initJointPosition(JointIs<JointPositionInfo>("robot1", "joint3")))
      .WillOnce(Return(true));
  EXPECT_TRUE(joint_input.init(state_mock_, command_mock_));

  // Test read functionality (interface calls, even without ONNX buffer)
  EXPECT_CALL(state_mock_, jointPosition(JointIs<JointPositionInfo>("robot1", "joint1")))
      .WillOnce(Return(1.5));
  EXPECT_CALL(state_mock_, jointPosition(JointIs<JointPositionInfo>("robot1", "joint2")))
      .WillOnce(Return(2.0));
  EXPECT_CALL(state_mock_, jointPosition(JointIs<JointPositionInfo>("robot1", "joint3")))
      .WillOnce(Return(0.5));

  // Read should call the interface methods even if ONNX buffer doesn't exist
  ASSERT_TRUE(joint_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BasePositionInput_InitAndRead) {
  BasePositionInput base_input("obj.robot1.base_name.pos_b_rt_w_in_w", "robot1");

  // Test successful initialization
  EXPECT_CALL(state_mock_, initBasePosW(ArticulationIs<BasePosWInfo>("robot1")))
      .WillOnce(Return(true));
  EXPECT_TRUE(base_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Vector3d expected_pos(1.0, 2.0, 3.0);
  EXPECT_CALL(state_mock_, basePosW(ArticulationIs<BasePosWInfo>("robot1")))
      .WillOnce(Return(expected_pos));

  ASSERT_TRUE(base_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BasePositionInput_InitFailure) {
  BasePositionInput base_input("obj.robot1.base_name.pos_b_rt_w_in_w", "robot1");
  EXPECT_CALL(state_mock_, initBasePosW(ArticulationIs<BasePosWInfo>("robot1")))
      .WillOnce(Return(false));
  EXPECT_FALSE(base_input.init(state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BaseOrientationInput_InitAndRead) {
  BaseOrientationInput base_input("obj.robot1.base_name.w_Q_b", "robot1");

  // Test initialization
  EXPECT_CALL(state_mock_, initBaseQuatW(ArticulationIs<BaseQuatWInfo>("robot1")))
      .WillOnce(Return(true));
  EXPECT_TRUE(base_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Quaterniond expected_quat(1.0, 0.0, 0.0, 0.0);
  EXPECT_CALL(state_mock_, baseQuatW(ArticulationIs<BaseQuatWInfo>("robot1")))
      .WillOnce(Return(expected_quat));

  ASSERT_TRUE(base_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BaseLinearVelocityInput_InitAndRead) {
  BaseLinearVelocityInput base_input("obj.robot1.base_name.lin_vel_b_rt_w_in_b", "robot1");

  // Test initialization
  EXPECT_CALL(state_mock_, initBaseLinVelB(ArticulationIs<BaseLinVelBInfo>("robot1")))
      .WillOnce(Return(true));
  EXPECT_TRUE(base_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Vector3d expected_vel(0.5, 0.0, 0.0);
  EXPECT_CALL(state_mock_, baseLinVelB(ArticulationIs<BaseLinVelBInfo>("robot1")))
      .WillOnce(Return(expected_vel));

  ASSERT_TRUE(base_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BaseAngularVelocityInput_InitAndRead) {
  BaseAngularVelocityInput base_input("obj.robot1.base_name.ang_vel_b_rt_w_in_b", "robot1");

  // Test initialization
  EXPECT_CALL(state_mock_, initBaseAngVelB(ArticulationIs<BaseAngVelBInfo>("robot1")))
      .WillOnce(Return(true));
  EXPECT_TRUE(base_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Vector3d expected_ang_vel(0.0, 0.0, 0.1);
  EXPECT_CALL(state_mock_, baseAngVelB(ArticulationIs<BaseAngVelBInfo>("robot1")))
      .WillOnce(Return(expected_ang_vel));

  ASSERT_TRUE(base_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, JointVelocityInput_InitAndRead) {
  std::vector<std::string> joint_names = {"joint1", "joint2", "joint3"};
  JointVelocityInput joint_input("obj.robot1.joints.vel", "robot1", joint_names);

  // Test initialization
  EXPECT_CALL(state_mock_, initJointVelocity(JointIs<JointVelocityInfo>("robot1", "joint1")))
      .WillOnce(Return(true));
  EXPECT_CALL(state_mock_, initJointVelocity(JointIs<JointVelocityInfo>("robot1", "joint2")))
      .WillOnce(Return(true));
  EXPECT_CALL(state_mock_, initJointVelocity(JointIs<JointVelocityInfo>("robot1", "joint3")))
      .WillOnce(Return(true));
  EXPECT_TRUE(joint_input.init(state_mock_, command_mock_));

  // Test read functionality
  EXPECT_CALL(state_mock_, jointVelocity(JointIs<JointVelocityInfo>("robot1", "joint1")))
      .WillOnce(Return(0.5));
  EXPECT_CALL(state_mock_, jointVelocity(JointIs<JointVelocityInfo>("robot1", "joint2")))
      .WillOnce(Return(-0.3));
  EXPECT_CALL(state_mock_, jointVelocity(JointIs<JointVelocityInfo>("robot1", "joint3")))
      .WillOnce(Return(0.1));

  ASSERT_TRUE(joint_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BodyOrientationInput_InitAndRead) {
  BodyOrientationInput body_input("obj.box1.box.w_Q_b", "box1", "test_body");

  // Test initialization
  EXPECT_CALL(state_mock_, initBodyOrientationW(BodyIs<BodyOrientationWInfo>("box1", "test_body")))
      .WillOnce(Return(true));
  EXPECT_TRUE(body_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Quaterniond expected_quat(0.707, 0.0, 0.0, 0.707);
  EXPECT_CALL(state_mock_, bodyOrientationW(BodyIs<BodyOrientationWInfo>("box1", "test_body")))
      .WillOnce(Return(expected_quat));

  EXPECT_TRUE(body_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BodyLinearVelocityInput_InitAndRead) {
  BodyLinearVelocityInput body_input("obj.box1.box.lin_vel_b_rt_w_in_b", "box1", "box");

  // Test initialization
  EXPECT_CALL(state_mock_, initBodyLinearVelocityB(BodyIs<BodyLinearVelocityBInfo>("box1", "box")))
      .WillOnce(Return(true));
  EXPECT_TRUE(body_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Vector3d expected_vel(1.0, 0.5, 0.0);
  EXPECT_CALL(state_mock_, bodyLinearVelocityB(BodyIs<BodyLinearVelocityBInfo>("box1", "box")))
      .WillOnce(Return(expected_vel));

  EXPECT_TRUE(body_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BodyLinearVelocityInput_InitFailure) {
  BodyLinearVelocityInput body_input("obj.box1.box.lin_vel_b_rt_w_in_b", "box1", "box");
  EXPECT_CALL(state_mock_, initBodyLinearVelocityB(BodyIs<BodyLinearVelocityBInfo>("box1", "box")))
      .WillOnce(Return(false));
  EXPECT_FALSE(body_input.init(state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BodyLinearVelocityInput_ReadFailsWhenStateReturnsNullopt) {
  BodyLinearVelocityInput body_input("obj.box1.box.lin_vel_b_rt_w_in_b", "box1", "box");
  EXPECT_CALL(state_mock_, bodyLinearVelocityB(BodyIs<BodyLinearVelocityBInfo>("box1", "box")))
      .WillOnce(Return(std::nullopt));
  EXPECT_FALSE(body_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BodyAngularVelocityInput_InitAndRead) {
  BodyAngularVelocityInput body_input("obj.box1.box.ang_vel_b_rt_w_in_b", "box1", "box");

  // Test initialization
  EXPECT_CALL(state_mock_,
              initBodyAngularVelocityB(BodyIs<BodyAngularVelocityBInfo>("box1", "box")))
      .WillOnce(Return(true));
  EXPECT_TRUE(body_input.init(state_mock_, command_mock_));

  // Test read functionality
  Eigen::Vector3d expected_ang_vel(0.0, 0.1, 0.2);
  EXPECT_CALL(state_mock_, bodyAngularVelocityB(BodyIs<BodyAngularVelocityBInfo>("box1", "box")))
      .WillOnce(Return(expected_ang_vel));

  EXPECT_TRUE(body_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BodyAngularVelocityInput_InitFailure) {
  BodyAngularVelocityInput body_input("obj.box1.box.ang_vel_b_rt_w_in_b", "box1", "box");
  EXPECT_CALL(state_mock_,
              initBodyAngularVelocityB(BodyIs<BodyAngularVelocityBInfo>("box1", "box")))
      .WillOnce(Return(false));
  EXPECT_FALSE(body_input.init(state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, BodyAngularVelocityInput_ReadFailsWhenStateReturnsNullopt) {
  BodyAngularVelocityInput body_input("obj.box1.box.ang_vel_b_rt_w_in_b", "box1", "box");
  EXPECT_CALL(state_mock_, bodyAngularVelocityB(BodyIs<BodyAngularVelocityBInfo>("box1", "box")))
      .WillOnce(Return(std::nullopt));
  EXPECT_FALSE(body_input.read(runtime, state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, StepCountInput_WithRealRuntime) {
  // Arrange
  StepCountInput step_input("ctx.step_count");

  // Get initial buffer state
  auto buffer = runtime.inputBuffer<int32_t>("ctx.step_count");
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
  MemoryOutput memory_output("output.joint_targets.robot1.pos");

  // Set initial memory data
  auto buffer = runtime.inputBuffer<float>("memory.output.joint_targets.robot1.pos.in");
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

TEST_F(OnnxComponentsTest, CommandJointPositionInput_InitAndRead) {
  metadata::JointPositionCommandMetadata meta;
  meta.joint_names = {"j1", "j2"};
  CommandJointPositionInput input("cmd.joint_pos.arm", "arm", meta);

  EXPECT_CALL(command_mock_, initJointPosition(JointPositionCommandIs("arm", "j1")))
      .WillOnce(Return(true));
  EXPECT_CALL(command_mock_, initJointPosition(JointPositionCommandIs("arm", "j2")))
      .WillOnce(Return(true));
  EXPECT_TRUE(input.init(state_mock_, command_mock_));

  EXPECT_CALL(command_mock_, jointPosition(JointPositionCommandIs("arm", "j1")))
      .WillOnce(Return(std::make_optional(0.1f)));
  EXPECT_CALL(command_mock_, jointPosition(JointPositionCommandIs("arm", "j2")))
      .WillOnce(Return(std::make_optional(0.2f)));
  EXPECT_TRUE(input.read(runtime, state_mock_, command_mock_));

  auto buffer = runtime.inputBuffer<float>("cmd.joint_pos.arm");
  ASSERT_TRUE(buffer.has_value());
  ASSERT_EQ(buffer->size(), 2u);
  EXPECT_FLOAT_EQ((*buffer)[0], 0.1f);
  EXPECT_FLOAT_EQ((*buffer)[1], 0.2f);
}

TEST_F(OnnxComponentsTest, CommandJointPositionInput_InitFailsWhenJointNamesEmpty) {
  metadata::JointPositionCommandMetadata meta;  // joint_names left empty
  CommandJointPositionInput input("cmd.joint_pos.arm", "arm", meta);
  EXPECT_FALSE(input.init(state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, CommandJointPositionInput_InitFailsWhenOneJointFails) {
  metadata::JointPositionCommandMetadata meta;
  meta.joint_names = {"j1", "j2"};
  CommandJointPositionInput input("cmd.joint_pos.arm", "arm", meta);

  EXPECT_CALL(command_mock_, initJointPosition(JointPositionCommandIs("arm", "j1")))
      .WillOnce(Return(false));
  EXPECT_FALSE(input.init(state_mock_, command_mock_));
}

TEST_F(OnnxComponentsTest, CommandJointPositionInput_ReadFailsWhenJointUnavailable) {
  metadata::JointPositionCommandMetadata meta;
  meta.joint_names = {"j1", "j2"};
  CommandJointPositionInput input("cmd.joint_pos.arm", "arm", meta);

  EXPECT_CALL(command_mock_, jointPosition(JointPositionCommandIs("arm", "j1")))
      .WillOnce(Return(std::make_optional(0.1f)));
  EXPECT_CALL(command_mock_, jointPosition(JointPositionCommandIs("arm", "j2")))
      .WillOnce(Return(std::nullopt));
  EXPECT_FALSE(input.read(runtime, state_mock_, command_mock_));
}

// ---------------  Matcher tests for default metadata --------------------------------

TEST(CommandFloatMatcherTest, CreatesInputWithoutMetadata) {
  CommandFloatMatcher matcher;
  Match match_without_metadata{.name = "cmd.float.gain"};
  ASSERT_TRUE(matcher.matches(match_without_metadata));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 1u) << "Should create input even without metadata";
}

TEST(CommandFloatMatcherTest, CreatesInputWithMetadata) {
  CommandFloatMatcher matcher;
  Match match_with_metadata{
      .name = "cmd.float.scale",
      .metadata = R"({"range": [-1.0, 1.0]})",
  };
  ASSERT_TRUE(matcher.matches(match_with_metadata));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 1u);
}

TEST(CommandFloatMatcherTest, MatchesDottedCommandName) {
  // Composite command terms register one float per field, e.g. the wholebody command's
  // `cmd.float.<command>.height` / `.pitch`.
  CommandFloatMatcher matcher;
  EXPECT_TRUE(matcher.matches({.name = "cmd.float.wholebody_command.height"}));
  EXPECT_TRUE(matcher.matches({.name = "cmd.float.wholebody_command.pitch"}));
}

TEST(CommandFloatMatcherTest, MatchesArbitrarilyNestedCommandName) {
  CommandFloatMatcher matcher;
  EXPECT_TRUE(matcher.matches({.name = "cmd.float.wholebody_command.height.extra"}));
}

TEST(CommandFloatMatcherTest, DoesNotMatchMalformedNames) {
  CommandFloatMatcher matcher;
  EXPECT_FALSE(matcher.matches({.name = "cmd.float"}));          // no segment
  EXPECT_FALSE(matcher.matches({.name = "cmd.float."}));         // trailing dot
  EXPECT_FALSE(matcher.matches({.name = "cmd.float..height"}));  // empty segment
  EXPECT_FALSE(matcher.matches({.name = "cmd.float.a.b."}));     // trailing dot after nesting
}

TEST(CommandFloatMatcherTest, KeepsFieldsOfSameCommandDistinct) {
  // The whole `<command>.<field>` string is the storage key. Keying on `<command>` alone
  // would make the two fields collide and silently drop one input.
  CommandFloatMatcher matcher;
  ASSERT_TRUE(matcher.matches({.name = "cmd.float.wholebody_command.height"}));
  ASSERT_TRUE(matcher.matches({.name = "cmd.float.wholebody_command.pitch"}));

  auto inputs = matcher.createInputs();
  EXPECT_EQ(inputs.size(), 2u);
}

TEST(CommandSE2VelocityMatcherTest, CreatesInputWithoutMetadata) {
  CommandSE2VelocityMatcher matcher;
  Match match_without_metadata{.name = "cmd.se2_velocity.vel"};
  ASSERT_TRUE(matcher.matches(match_without_metadata));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 1u) << "Should create input even without metadata";
}

TEST(CommandSE2VelocityMatcherTest, CreatesInputWithMetadata) {
  CommandSE2VelocityMatcher matcher;
  Match match_with_metadata{
      .name = "cmd.se2_velocity.vel",
      .metadata =
          R"({"ranges": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-0.5, 0.5], "ang_vel_z": [-2.0, 2.0]}})",
  };
  ASSERT_TRUE(matcher.matches(match_with_metadata));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 1u);
}

TEST(CommandJointPositionMatcherTest, DoesNotMatchOtherPatterns) {
  CommandJointPositionMatcher matcher;
  EXPECT_FALSE(matcher.matches({.name = "cmd.float.gain"}));
  EXPECT_FALSE(matcher.matches({.name = "cmd.joint_vel.arm"}));
  EXPECT_FALSE(matcher.matches({.name = "cmd.joint_pos"}));
}

TEST(CommandJointPositionMatcherTest, SkipsInputWithoutMetadata) {
  CommandJointPositionMatcher matcher;
  ASSERT_TRUE(matcher.matches({.name = "cmd.joint_pos.arm"}));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 0u);
}

TEST(CommandJointPositionMatcherTest, SkipsInputWithEmptyJointNames) {
  CommandJointPositionMatcher matcher;
  ASSERT_TRUE(matcher.matches({.name = "cmd.joint_pos.arm", .metadata = R"({"joint_names": []})"}));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 0u);
}

TEST(CommandJointPositionMatcherTest, CreatesInputWithMetadata) {
  CommandJointPositionMatcher matcher;
  Match m{
      .name = "cmd.joint_pos.arm",
      .metadata = R"({"joint_names": ["j1", "j2"]})",
  };
  ASSERT_TRUE(matcher.matches(m));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 1u);
}

TEST(CommandJointPositionMatcherTest, DoesNotMatchSameNameTwice) {
  CommandJointPositionMatcher matcher;
  ASSERT_TRUE(matcher.matches({.name = "cmd.joint_pos.arm"}));
  // Second match with the same name overwrites, still produces one entry (but 0 inputs without
  // metadata)
  ASSERT_TRUE(matcher.matches({.name = "cmd.joint_pos.arm"}));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 0u);
}

TEST(CommandJointPositionMatcherTest, MatchesMultipleCommands) {
  CommandJointPositionMatcher matcher;
  ASSERT_TRUE(matcher.matches(
      {.name = "cmd.joint_pos.arm", .metadata = R"({"joint_names": ["j1", "j2"]})"}));
  ASSERT_TRUE(matcher.matches(
      {.name = "cmd.joint_pos.leg", .metadata = R"({"joint_names": ["j3", "j4"]})"}));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 2u);
}

// ---------------  JointTargetMatcher tests --------------------------------

namespace {
// Metadata shape emitted by the exporter's joint-targets group.
constexpr const char* kJointTargetMetadata =
    R"({"type": "joint_targets", "names": ["j1", "j2"], "stiffness": [10.0, 10.0], "damping": [1.0, 1.0]})";
}  // namespace

TEST(JointTargetMatcherTest, MatchesTwoSegmentName) {
  // Legacy form: `output.joint_targets.<articulation>.<field>`, still emitted by the
  // mjlab frontend and by previously exported artifacts.
  JointTargetMatcher matcher;
  EXPECT_TRUE(matcher.matches({.name = "output.joint_targets.robot1.pos"}));
  EXPECT_TRUE(matcher.matches({.name = "output.joint_targets.robot1.vel"}));
  EXPECT_TRUE(matcher.matches({.name = "output.joint_targets.robot1.effort"}));
}

TEST(JointTargetMatcherTest, MatchesThreeSegmentName) {
  // Current form: `output.joint_targets.<articulation>.<action_term>.<field>`.
  JointTargetMatcher matcher;
  EXPECT_TRUE(matcher.matches({.name = "output.joint_targets.robot1.joint_pos.pos"}));
  EXPECT_TRUE(matcher.matches({.name = "output.joint_targets.robot1.joint_pos.vel"}));
  EXPECT_TRUE(matcher.matches({.name = "output.joint_targets.robot1.joint_pos.effort"}));
}

TEST(JointTargetMatcherTest, DoesNotMatchUnrelatedPatterns) {
  JointTargetMatcher matcher;
  // Missing field suffix.
  EXPECT_FALSE(matcher.matches({.name = "output.joint_targets.robot1"}));
  // Unsupported field.
  EXPECT_FALSE(matcher.matches({.name = "output.joint_targets.robot1.acceleration"}));
  // Wrong prefix.
  EXPECT_FALSE(matcher.matches({.name = "output.se2_velocity.robot1.pos"}));
  // One segment too many: kAlphanumeric excludes '.', so this must not match.
  EXPECT_FALSE(matcher.matches({.name = "output.joint_targets.robot1.joint_pos.extra.pos"}));
}

TEST(JointTargetMatcherTest, GroupsFieldsOfOneTermIntoSingleOutput) {
  JointTargetMatcher matcher;
  ASSERT_TRUE(matcher.matches(
      {.name = "output.joint_targets.robot1.joint_pos.pos", .metadata = kJointTargetMetadata}));
  ASSERT_TRUE(matcher.matches({.name = "output.joint_targets.robot1.joint_pos.vel"}));
  ASSERT_TRUE(matcher.matches({.name = "output.joint_targets.robot1.joint_pos.effort"}));

  auto outputs = matcher.createOutputs();
  EXPECT_EQ(outputs.size(), 1u) << "pos/vel/effort of one term belong to one group";
}

TEST(JointTargetMatcherTest, SeparatesActionTermsOfSameArticulation) {
  // Two joint action terms on one robot (e.g. policy-driven lower body plus arms driven
  // from a command term) must yield two distinct groups.
  JointTargetMatcher matcher;
  ASSERT_TRUE(matcher.matches(
      {.name = "output.joint_targets.robot1.joint_pos.pos", .metadata = kJointTargetMetadata}));
  ASSERT_TRUE(matcher.matches({.name = "output.joint_targets.robot1.arms_joint_pos.pos",
                               .metadata = kJointTargetMetadata}));

  auto outputs = matcher.createOutputs();
  EXPECT_EQ(outputs.size(), 2u);
}

TEST(JointTargetMatcherTest, SkipsOutputWithoutMetadata) {
  JointTargetMatcher matcher;
  ASSERT_TRUE(matcher.matches({.name = "output.joint_targets.robot1.joint_pos.pos"}));

  auto outputs = matcher.createOutputs();
  EXPECT_EQ(outputs.size(), 0u);
}

TEST(JointTargetMatcherTest, ResolvesArticulationFromFirstSegmentInBothForms) {
  // The regression this guards: the articulation must come from the segment directly
  // after `joint_targets.` in both forms. A group-index slip would resolve the action
  // term name ("joint_pos") as the articulation and route targets to a nonexistent robot.
  JointTargetMatcher matcher;
  ASSERT_TRUE(matcher.matches(
      {.name = "output.joint_targets.robot1.joint_pos.pos", .metadata = kJointTargetMetadata}));
  ASSERT_TRUE(matcher.matches(
      {.name = "output.joint_targets.robot2.pos", .metadata = kJointTargetMetadata}));

  auto outputs = matcher.createOutputs();
  ASSERT_EQ(outputs.size(), 2u);

  // JointTargetOutput::init calls initJointOutput once per joint in the metadata,
  // carrying the articulation the matcher resolved.
  StrictMock<MockRobotStateInterface> state;
  MockCommandInterface command;
  EXPECT_CALL(state, initJointOutput(JointIs<JointOutputInfo>("robot1", "j1")))
      .WillOnce(Return(true));
  EXPECT_CALL(state, initJointOutput(JointIs<JointOutputInfo>("robot1", "j2")))
      .WillOnce(Return(true));
  EXPECT_CALL(state, initJointOutput(JointIs<JointOutputInfo>("robot2", "j1")))
      .WillOnce(Return(true));
  EXPECT_CALL(state, initJointOutput(JointIs<JointOutputInfo>("robot2", "j2")))
      .WillOnce(Return(true));
  for (auto& output : outputs) EXPECT_TRUE(output->init(state, command));
}

// ---------------  Base*Matcher tests --------------------------------

TEST(BasePositionMatcherTest, MatchesPairFromBaseNames) {
  BasePositionMatcher matcher;
  EXPECT_TRUE(matcher.matches({
      .name = "obj.robot1.base_link.pos_b_rt_w_in_w",
      .base_names = {{"robot1", "base_link"}},
  }));
}

TEST(BasePositionMatcherTest, DoesNotMatchWhenBaseNamesEmpty) {
  BasePositionMatcher matcher;
  EXPECT_FALSE(matcher.matches({.name = "obj.robot1.base_link.pos_b_rt_w_in_w"}));
}

TEST(BasePositionMatcherTest, DoesNotMatchWhenPairNotInBaseNames) {
  BasePositionMatcher matcher;
  // Articulation name in tensor doesn't match any registered articulation.
  EXPECT_FALSE(matcher.matches({
      .name = "obj.other.base_link.pos_b_rt_w_in_w",
      .base_names = {{"robot1", "base_link"}},
  }));
  // Base name in tensor doesn't match the articulation's registered base.
  EXPECT_FALSE(matcher.matches({
      .name = "obj.robot1.torso.pos_b_rt_w_in_w",
      .base_names = {{"robot1", "base_link"}},
  }));
  // Wrong field suffix.
  EXPECT_FALSE(matcher.matches({
      .name = "obj.robot1.base_link.w_Q_b",
      .base_names = {{"robot1", "base_link"}},
  }));
}

TEST(BasePositionMatcherTest, PropagatesCorrectArticulationWithMultipleArticulations) {
  // Two articulations registered, each with its own base name.
  std::unordered_map<std::string, std::string> base_names = {{"robot1", "base_link"},
                                                             {"robot2", "torso"}};
  BasePositionMatcher matcher;
  ASSERT_TRUE(matcher.matches({
      .name = "obj.robot1.base_link.pos_b_rt_w_in_w",
      .base_names = base_names,
  }));
  ASSERT_TRUE(matcher.matches({
      .name = "obj.robot2.torso.pos_b_rt_w_in_w",
      .base_names = base_names,
  }));
  // Cross-pair tensors must be rejected (robot1 is not paired with torso).
  ASSERT_FALSE(matcher.matches({
      .name = "obj.robot1.torso.pos_b_rt_w_in_w",
      .base_names = base_names,
  }));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 2u);

  // Verify each input was constructed with the correct articulation name by
  // observing the init() call on a mock state interface.
  StrictMock<MockRobotStateInterface> state;
  MockCommandInterface command;
  EXPECT_CALL(state, initBasePosW(ArticulationIs<BasePosWInfo>("robot1"))).WillOnce(Return(true));
  EXPECT_CALL(state, initBasePosW(ArticulationIs<BasePosWInfo>("robot2"))).WillOnce(Return(true));
  for (auto& input : inputs) EXPECT_TRUE(input->init(state, command));
}

TEST(BaseOrientationMatcherTest, MatchesPairFromBaseNames) {
  BaseOrientationMatcher matcher;
  EXPECT_TRUE(matcher.matches({
      .name = "obj.robot1.base_link.w_Q_b",
      .base_names = {{"robot1", "base_link"}},
  }));
}

TEST(BaseOrientationMatcherTest, DoesNotMatchWhenBaseNamesEmpty) {
  BaseOrientationMatcher matcher;
  EXPECT_FALSE(matcher.matches({.name = "obj.robot1.base_link.w_Q_b"}));
}

TEST(BaseOrientationMatcherTest, PropagatesCorrectArticulationWithMultipleArticulations) {
  std::unordered_map<std::string, std::string> base_names = {{"robot1", "base_link"},
                                                             {"robot2", "torso"}};
  BaseOrientationMatcher matcher;
  ASSERT_TRUE(matcher.matches({.name = "obj.robot1.base_link.w_Q_b", .base_names = base_names}));
  ASSERT_TRUE(matcher.matches({.name = "obj.robot2.torso.w_Q_b", .base_names = base_names}));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 2u);

  StrictMock<MockRobotStateInterface> state;
  MockCommandInterface command;
  EXPECT_CALL(state, initBaseQuatW(ArticulationIs<BaseQuatWInfo>("robot1"))).WillOnce(Return(true));
  EXPECT_CALL(state, initBaseQuatW(ArticulationIs<BaseQuatWInfo>("robot2"))).WillOnce(Return(true));
  for (auto& input : inputs) EXPECT_TRUE(input->init(state, command));
}

TEST(BaseLinearVelocityMatcherTest, MatchesPairFromBaseNames) {
  BaseLinearVelocityMatcher matcher;
  EXPECT_TRUE(matcher.matches({
      .name = "obj.robot1.base_link.lin_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));
}

TEST(BaseLinearVelocityMatcherTest, DoesNotMatchWhenBaseNamesEmpty) {
  BaseLinearVelocityMatcher matcher;
  EXPECT_FALSE(matcher.matches({.name = "obj.robot1.base_link.lin_vel_b_rt_w_in_b"}));
}

TEST(BaseLinearVelocityMatcherTest, PropagatesCorrectArticulationWithMultipleArticulations) {
  std::unordered_map<std::string, std::string> base_names = {{"robot1", "base_link"},
                                                             {"robot2", "torso"}};
  BaseLinearVelocityMatcher matcher;
  ASSERT_TRUE(matcher.matches(
      {.name = "obj.robot1.base_link.lin_vel_b_rt_w_in_b", .base_names = base_names}));
  ASSERT_TRUE(
      matcher.matches({.name = "obj.robot2.torso.lin_vel_b_rt_w_in_b", .base_names = base_names}));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 2u);

  StrictMock<MockRobotStateInterface> state;
  MockCommandInterface command;
  EXPECT_CALL(state, initBaseLinVelB(ArticulationIs<BaseLinVelBInfo>("robot1")))
      .WillOnce(Return(true));
  EXPECT_CALL(state, initBaseLinVelB(ArticulationIs<BaseLinVelBInfo>("robot2")))
      .WillOnce(Return(true));
  for (auto& input : inputs) EXPECT_TRUE(input->init(state, command));
}

TEST(BaseAngularVelocityMatcherTest, MatchesPairFromBaseNames) {
  BaseAngularVelocityMatcher matcher;
  EXPECT_TRUE(matcher.matches({
      .name = "obj.robot1.base_link.ang_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));
}

TEST(BaseAngularVelocityMatcherTest, DoesNotMatchWhenBaseNamesEmpty) {
  BaseAngularVelocityMatcher matcher;
  EXPECT_FALSE(matcher.matches({.name = "obj.robot1.base_link.ang_vel_b_rt_w_in_b"}));
}

TEST(BaseAngularVelocityMatcherTest, PropagatesCorrectArticulationWithMultipleArticulations) {
  std::unordered_map<std::string, std::string> base_names = {{"robot1", "base_link"},
                                                             {"robot2", "torso"}};
  BaseAngularVelocityMatcher matcher;
  ASSERT_TRUE(matcher.matches(
      {.name = "obj.robot1.base_link.ang_vel_b_rt_w_in_b", .base_names = base_names}));
  ASSERT_TRUE(
      matcher.matches({.name = "obj.robot2.torso.ang_vel_b_rt_w_in_b", .base_names = base_names}));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 2u);

  StrictMock<MockRobotStateInterface> state;
  MockCommandInterface command;
  EXPECT_CALL(state, initBaseAngVelB(ArticulationIs<BaseAngVelBInfo>("robot1")))
      .WillOnce(Return(true));
  EXPECT_CALL(state, initBaseAngVelB(ArticulationIs<BaseAngVelBInfo>("robot2")))
      .WillOnce(Return(true));
  for (auto& input : inputs) EXPECT_TRUE(input->init(state, command));
}

// ---------------  BodyLinearVelocityMatcher tests --------------------------------

TEST(BodyLinearVelocityMatcherTest, MatchesBodyLinearVelocityPattern) {
  BodyLinearVelocityMatcher matcher;
  Match match{
      .name = "obj.box1.box.lin_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  };
  EXPECT_TRUE(matcher.matches(match));
}

TEST(BodyLinearVelocityMatcherTest, MatchesBodyLinearVelocityPatternWithoutBaseNames) {
  BodyLinearVelocityMatcher matcher;
  EXPECT_TRUE(matcher.matches({.name = "obj.box1.box.lin_vel_b_rt_w_in_b"}));
}

TEST(BodyLinearVelocityMatcherTest, DoesNotMatchUnrelatedPatterns) {
  BodyLinearVelocityMatcher matcher;
  // Wrong prefix (not obj.)
  EXPECT_FALSE(matcher.matches({
      .name = "sensor.imu.pelvis.lin_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));
  // Wrong suffix
  EXPECT_FALSE(matcher.matches({
      .name = "obj.box1.bodies.box.lin_vel_b_rt_w_in_w",
      .base_names = {{"robot1", "base_link"}},
  }));
  // Missing bodies segment (only one alphanumeric after obj.)
  EXPECT_FALSE(matcher.matches({
      .name = "obj.box1.lin_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));
}

TEST(BodyLinearVelocityMatcherTest, CreatesInputForMatchedBody) {
  BodyLinearVelocityMatcher matcher;
  ASSERT_TRUE(matcher.matches({
      .name = "obj.box1.sphere.lin_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 1u);
}

TEST(BodyLinearVelocityMatcherTest, CreatesInputsForMultipleBodies) {
  BodyLinearVelocityMatcher matcher;
  ASSERT_TRUE(matcher.matches({
      .name = "obj.box1.square.lin_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));
  ASSERT_TRUE(matcher.matches({
      .name = "obj.box2.circle.lin_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));
  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 2u);
}

TEST(BodyLinearVelocityMatcherTest, DoesNotMatchBodyLinearVelocityWhenBaseNameMatches) {
  BodyLinearVelocityMatcher matcher;
  // When the tensor name matches the base_names pattern it should be rejected.
  Match match{
      .name = "obj.robot1.base_link.lin_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  };
  EXPECT_FALSE(matcher.matches(match));
}

// ---------------  BodyAngularVelocityMatcher tests --------------------------------

TEST(BodyAngularVelocityMatcherTest, MatchesBodyAngularVelocityPattern) {
  BodyAngularVelocityMatcher matcher;
  Match match{
      .name = "obj.box1.box.ang_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  };
  EXPECT_TRUE(matcher.matches(match));
}

TEST(BodyAngularVelocityMatcherTest, MatchesBodyAngularVelocityPatternWithoutBaseNames) {
  BodyAngularVelocityMatcher matcher;
  EXPECT_TRUE(matcher.matches({.name = "obj.box1.box.ang_vel_b_rt_w_in_b"}));
}

TEST(BodyAngularVelocityMatcherTest, DoesNotMatchUnrelatedPatterns) {
  BodyAngularVelocityMatcher matcher;
  // Wrong prefix (not obj.)
  EXPECT_FALSE(matcher.matches({
      .name = "sensor.imu.pelvis.ang_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));
  // Wrong suffix
  EXPECT_FALSE(matcher.matches({
      .name = "obj.box1.bodies.box.ang_vel_b_rt_w_in_w",
      .base_names = {{"robot1", "base_link"}},
  }));
  // Missing bodies segment (only one alphanumeric after obj.)
  EXPECT_FALSE(matcher.matches({
      .name = "obj.box1.ang_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));
}

TEST(BodyAngularVelocityMatcherTest, CreatesInputForMatchedBody) {
  BodyAngularVelocityMatcher matcher;
  ASSERT_TRUE(matcher.matches({
      .name = "obj.box1.sphere.ang_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));

  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 1u);
}

TEST(BodyAngularVelocityMatcherTest, CreatesInputsForMultipleBodies) {
  BodyAngularVelocityMatcher matcher;
  ASSERT_TRUE(matcher.matches({
      .name = "obj.box1.square.ang_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));
  ASSERT_TRUE(matcher.matches({
      .name = "obj.box2.circle.ang_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  }));
  auto inputs = matcher.createInputs();
  ASSERT_EQ(inputs.size(), 2u);
}

TEST(BodyAngularVelocityMatcherTest, DoesNotMatchBodyAngularVelocityWhenBaseNameMatches) {
  BodyAngularVelocityMatcher matcher;
  Match match{
      .name = "obj.robot1.base_link.ang_vel_b_rt_w_in_b",
      .base_names = {{"robot1", "base_link"}},
  };
  EXPECT_FALSE(matcher.matches(match));
}

}  // namespace exploy::control
