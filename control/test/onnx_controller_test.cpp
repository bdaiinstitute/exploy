// Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "onnx_controller.hpp"
#include "mock_command_interface.hpp"
#include "mock_data_collection_interface.hpp"
#include "mock_state_interface.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <filesystem>

namespace rai::cs::control::common::onnx {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Field;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

using operation::common::data_collection::MockDataCollectionInterface;

MATCHER_P2(DoubleRangeIs, min_val, max_val, "") {
  return arg.min == min_val && arg.max == max_val;
}

MATCHER_P(HasRanges, expected_ranges, "") {
  if (!arg.ranges) {
    *result_listener << "config ranges are not set";
    return false;
  }

  const auto ranges_matcher =
      AllOf(Field(&SE2VelocityRanges::lin_vel_x,
                  DoubleRangeIs(expected_ranges.lin_vel_x.min, expected_ranges.lin_vel_x.max)),
            Field(&SE2VelocityRanges::lin_vel_y,
                  DoubleRangeIs(expected_ranges.lin_vel_y.min, expected_ranges.lin_vel_y.max)),
            Field(&SE2VelocityRanges::ang_vel_z,
                  DoubleRangeIs(expected_ranges.ang_vel_z.min, expected_ranges.ang_vel_z.max)));

  if (!Matches(ranges_matcher)(arg.ranges.value())) {
    *result_listener << "ranges do not match";
    return false;
  }

  return true;
}

const std::string model_path = (std::filesystem::path(TEST_DATA_DIR) / "test.onnx").string();

HeightScan createTestHeightScan(int size) {
  return HeightScan{.height = std::vector<double>(size),
                    .color = HeightScan::ColorScan{
                        .r = std::vector<double>(size),
                        .g = std::vector<double>(size),
                        .b = std::vector<double>(size),
                    }};
}
auto kHeightScanData = std::vector<HeightScan>{createTestHeightScan(4), createTestHeightScan(4),
                                               createTestHeightScan(8)};
auto kRangeImageData = std::vector<double>{0, 0, 0, 0};
auto kDepthImageData = std::vector<double>{0, 0, 0, 0};
const auto kPositionData = std::make_optional(Position{0, 0, 0});
const auto kQuaternionData = std::make_optional(Quaternion{1, 0, 0, 0});  // w x y z
const auto kLinearVelocityData = std::make_optional(LinearVelocity{0, 0, 0});
const auto kAngularVelocityData = std::make_optional(AngularVelocity{0, 0, 0});
const auto kSE3PoseData = std::make_optional(SE3Pose{
    .position{0, 0, 0},
    .orientation{1, 0, 0, 0},
});

const SE2VelocityRanges kRanges{
    .lin_vel_x = {.min = -1.5, .max = 1.5},
    .lin_vel_y = {.min = -0.75, .max = 0.75},
    .ang_vel_z = {.min = -2.5, .max = 2.5},
};

class OnnxControllerTest : public ::testing::Test {
 protected:
  NiceMock<MockRobotStateInterface> state_mock_;
  NiceMock<MockCommandInterface> command_mock_;
  NiceMock<MockDataCollectionInterface> data_collection_mock_;
  OnnxRLController oc_{state_mock_, command_mock_, data_collection_mock_};

  void SetUp() override {}

  void ExpectInitBase() {
    EXPECT_CALL(state_mock_, initBasePosW()).WillOnce(Return(true));  // called once for base pos
    EXPECT_CALL(state_mock_, initBaseQuatW())
        .Times(2)
        .WillRepeatedly(Return(true));  // called twice for pelvis and torso
    EXPECT_CALL(state_mock_, initBaseLinVelB()).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initBaseAngVelB()).WillOnce(Return(true));
  }

  void ExpectInitOutput() {
    // Joint outputs from the output joint targets
    EXPECT_CALL(state_mock_, initJointOutput("j1")).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointOutput("j2")).WillOnce(Return(true));
    // Joint position and velocity inputs
    EXPECT_CALL(state_mock_, initJointPosition("j1")).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointPosition("j2")).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointPosition("j3")).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointVelocity("j1")).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointVelocity("j2")).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointVelocity("j3")).WillOnce(Return(true));

    EXPECT_CALL(state_mock_, initSe2Velocity("base_frame")).WillOnce(Return(true));
  }

  void ExpectInitSensors() {
    EXPECT_CALL(state_mock_, initHeightScan(_)).Times(3).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, initRangeImage(_)).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initDepthImage(_)).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initImuAngularVelocityImu("torso")).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initBodyOrientationW("box")).WillOnce(Return(true));
  }

  void ExpectInitCommands() {
    // SE2 velocity commands from inputs - check range configuration
    EXPECT_CALL(command_mock_,
                initSe2Velocity("vel", Field(&SE2VelocityConfig::ranges, Eq(std::nullopt))))
        .WillOnce(Return(true));
    EXPECT_CALL(command_mock_, initSe2Velocity("vel_with_range", HasRanges(kRanges)))
        .WillOnce(Return(true));
    EXPECT_CALL(command_mock_, initSe3Pose("pose")).WillOnce(Return(true));
  }

  void ExpectSetOutput() {
    EXPECT_CALL(state_mock_, setJointPosition("j1", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, setJointPosition("j2", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, setJointVelocity("j1", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, setJointVelocity("j2", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, setJointEffort("j1", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, setJointEffort("j2", _)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, setJointPGain("j1", 1)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, setJointPGain("j2", 2)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, setJointDGain("j1", 0.1)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, setJointDGain("j2", 0.2)).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, setSe2Velocity("base_frame", _)).WillRepeatedly(Return(true));
  }

  void ExpectReadJointState() {
    EXPECT_CALL(state_mock_, jointPosition("j1")).WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock_, jointPosition("j2")).WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock_, jointPosition("j3")).WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock_, jointVelocity("j1")).WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock_, jointVelocity("j2")).WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock_, jointVelocity("j3")).WillRepeatedly(Return(std::make_optional(0)));
  }

  void ExpectReadState() {
    EXPECT_CALL(state_mock_, imuAngularVelocityImu("pelvis")).WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(state_mock_, imuAngularVelocityImu("torso")).WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(state_mock_, imuOrientationW("pelvis")).WillRepeatedly(Return(kQuaternionData));
    EXPECT_CALL(state_mock_, bodyOrientationW("box")).WillRepeatedly(Return(kQuaternionData));
    EXPECT_CALL(state_mock_, heightScan(_, _))
        .WillRepeatedly(Return(std::make_optional(&kHeightScanData)));
    EXPECT_CALL(state_mock_, basePosW()).WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(state_mock_, baseQuatW()).WillRepeatedly(Return(kQuaternionData));
    EXPECT_CALL(state_mock_, baseLinVelB()).WillRepeatedly(Return(kLinearVelocityData));
    EXPECT_CALL(state_mock_, baseAngVelB()).WillRepeatedly(Return(kAngularVelocityData));
    EXPECT_CALL(state_mock_, rangeImage())
        .WillRepeatedly(Return(std::make_optional(&kRangeImageData)));
    EXPECT_CALL(state_mock_, depthImage())
        .WillRepeatedly(Return(std::make_optional(&kDepthImageData)));
  }

  void ExpectReadCommands() {
    EXPECT_CALL(command_mock_, se2Velocity("vel")).WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(command_mock_, se2Velocity("vel_with_range")).WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(command_mock_, se3Pose("pose")).WillRepeatedly(Return(kSE3PoseData));
  }
};

TEST_F(OnnxControllerTest, LoadFails) {
  StrictMock<MockRobotStateInterface> strict_state_mock;
  StrictMock<MockCommandInterface> strict_command_mock;
  NiceMock<MockDataCollectionInterface> nice_data_collection_mock;
  OnnxRLController oc(strict_state_mock, strict_command_mock, nice_data_collection_mock);

  ASSERT_FALSE(oc.load(""));
}

TEST_F(OnnxControllerTest, InitWithoutLoad) {
  StrictMock<MockRobotStateInterface> strict_state_mock;
  StrictMock<MockCommandInterface> strict_command_mock;
  NiceMock<MockDataCollectionInterface> nice_data_collection_mock;
  OnnxRLController oc(strict_state_mock, strict_command_mock, nice_data_collection_mock);

  EXPECT_FALSE(oc.init(false));
}

TEST_F(OnnxControllerTest, Update) {
  ExpectInitBase();
  ExpectInitOutput();
  ExpectInitSensors();
  ExpectInitCommands();

  // Initialize
  ASSERT_TRUE(oc_.load(model_path));
  ASSERT_TRUE(oc_.init(false));
  EXPECT_EQ(oc_.updateRate(), 10);

  ExpectReadJointState();
  ExpectReadState();
  ExpectReadCommands();

  ExpectSetOutput();

  const uint64_t t0_us = 0;
  ASSERT_TRUE(oc_.update(t0_us));
}

TEST_F(OnnxControllerTest, ReadJointFailure) {
  ExpectInitBase();
  ExpectInitOutput();
  ExpectInitSensors();
  ExpectInitCommands();

  ASSERT_TRUE(oc_.load(model_path));
  ASSERT_TRUE(oc_.init(false));

  EXPECT_CALL(state_mock_, jointPosition("j1")).WillRepeatedly(Return(std::nullopt));
  EXPECT_CALL(state_mock_, jointPosition("j2")).WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointPosition("j3")).WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointVelocity("j1")).WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointVelocity("j2")).WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointVelocity("j3")).WillRepeatedly(Return(std::make_optional(0)));

  ExpectReadState();
  ExpectReadCommands();

  ExpectSetOutput();

  EXPECT_FALSE(oc_.update(0));
}

TEST_F(OnnxControllerTest, WriteJointFailure) {
  ExpectInitBase();
  ExpectInitOutput();
  ExpectInitSensors();
  ExpectInitCommands();

  ASSERT_TRUE(oc_.load(model_path));
  ASSERT_TRUE(oc_.init(false));

  ExpectReadJointState();
  ExpectReadState();
  ExpectReadCommands();

  // j1 write fails
  EXPECT_CALL(state_mock_, setJointPosition("j1", _)).WillOnce(Return(false));
  EXPECT_CALL(state_mock_, setJointPosition("j2", _)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointVelocity(_, _)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointEffort(_, _)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointPGain(_, _)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointDGain(_, _)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setSe2Velocity(_, _)).WillRepeatedly(Return(true));

  EXPECT_FALSE(oc_.update(0));
}

}  // namespace rai::cs::control::common::onnx
