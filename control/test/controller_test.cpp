// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "exploy/controller.hpp"
#include "exploy/components.hpp"
#include "exploy/context.hpp"
#include "exploy/matcher.hpp"
#include "mock_command_interface.hpp"
#include "mock_data_collection_interface.hpp"
#include "mock_state_interface.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <regex>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <filesystem>

namespace exploy::control {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Field;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrictMock;

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

const std::string model_path = (std::filesystem::path(TEST_DATA_DIR) / "test_export.onnx").string();

std::unique_ptr<HeightScan> createTestHeightScan(int size) {
  const std::vector<double> data(size, 0.0);
  auto scan = std::make_unique<HeightScan>();
  scan->float_layers["height"] = std::vector<float>(data.begin(), data.end());
  scan->float_layers["r"] = std::vector<float>(data.begin(), data.end());
  scan->float_layers["g"] = std::vector<float>(data.begin(), data.end());
  scan->float_layers["b"] = std::vector<float>(data.begin(), data.end());
  return scan;
};

auto kHeightScanData = createTestHeightScan(4);
auto kTrailScanData = createTestHeightScan(8);

std::unique_ptr<MultiChannelImage> createTestSphericalImage(int size) {
  std::vector<float> data(size, 0.0f);
  auto image = std::make_unique<MultiChannelImage>();
  image->float_channels["range"] = std::vector<float>(data.begin(), data.end());
  image->float_channels["risk"] = std::vector<float>(data.begin(), data.end());
  return image;
}

std::unique_ptr<MultiChannelImage> createTestPinholeImage(int size) {
  std::vector<float> data(size, 0.0f);
  auto image = std::make_unique<MultiChannelImage>();
  image->float_channels["depth"] = std::vector<float>(data.begin(), data.end());
  return image;
}

auto kSphericalImageData = createTestSphericalImage(4);
auto kPinholeImageData = createTestPinholeImage(4);
const auto kPositionData = std::make_optional(Position{0, 0, 0});
const auto kQuaternionData = std::make_optional(Quaternion{1, 0, 0, 0});  // w x y z
const auto kLinearVelocityData = std::make_optional(LinearVelocity{0, 0, 0});
const auto kAngularVelocityData = std::make_optional(AngularVelocity{0, 0, 0});
const auto kSE3PoseData = std::make_optional(SE3Pose{
    .position{0, 0, 0},
    .orientation{1, 0, 0, 0},
});
const auto kExtensibleData = std::vector<double>{0.1, 0.2, 0.3};

const SE2VelocityRanges kRanges{
    .lin_vel_x = {.min = -1.5, .max = 1.5},
    .lin_vel_y = {.min = -0.75, .max = 0.75},
    .ang_vel_z = {.min = -2.5, .max = 2.5},
};

// Simple mock interface for testing extensibility pattern
class CustomInterface {
 public:
  MOCK_METHOD(bool, initExtensibleCommand, (const std::string& command_name), ());
  MOCK_METHOD(std::optional<std::vector<double>>, extensibleCommand,
              (const std::string& command_name), (const));
};

// Custom input component that calls the extended command interface
class CustomInput : public Input {
 public:
  CustomInput(const std::string& key, const std::string& command_name,
              CustomInterface* custom_interface)
      : Input("CustomInput", {key}),
        key_(key),
        command_name_(command_name),
        custom_interface_(*custom_interface) {}

  bool init(RobotStateInterface& /*state*/, CommandInterface& /*command*/) override {
    return custom_interface_.initExtensibleCommand(command_name_);
  }

  bool read(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
            CommandInterface& /*command*/) override {
    auto maybe_data = custom_interface_.extensibleCommand(command_name_);
    if (!maybe_data.has_value()) return false;
    auto maybe_buffer = runtime.inputBuffer<float>(key_);
    if (!maybe_buffer.has_value()) return false;
    std::copy(maybe_data->begin(), maybe_data->end(), maybe_buffer->begin());
    return true;
  }

 private:
  std::string key_;
  std::string command_name_;
  CustomInterface& custom_interface_;
};

// Custom matcher that recognizes and handles custom.* inputs
class CustomMatcher : public Matcher {
 public:
  explicit CustomMatcher(CustomInterface* custom_interface)
      : Matcher("CustomMatcher"), custom_interface_(*custom_interface) {}

  bool matches(const Match& maybe_match) override {
    // Use regex pattern like other matchers to extract name
    std::regex pattern = std::regex("custom\\.([a-zA-Z0-9_]+)");
    std::smatch match;
    if (std::regex_match(maybe_match.name, match, pattern) && match.size() > 1) {
      found_matches_[match[1].str()] = maybe_match;
      return true;
    }
    return false;
  }

  std::vector<std::unique_ptr<Input>> createInputs() const override {
    std::vector<std::unique_ptr<Input>> inputs;
    for (const auto& [command_name, found_match] : found_matches_) {
      // Use the extracted command name directly from regex match
      inputs.push_back(
          std::make_unique<CustomInput>(found_match.name, command_name, &custom_interface_));
    }
    return inputs;
  }

 private:
  CustomInterface& custom_interface_;
};

class OnnxControllerTest : public ::testing::Test {
 protected:
  NiceMock<MockRobotStateInterface> state_mock_;
  NiceMock<MockCommandInterface> command_mock_;
  NiceMock<MockDataCollectionInterface> data_collection_mock_;
  NiceMock<CustomInterface> custom_mock_;
  OnnxRLController oc_{state_mock_, command_mock_, data_collection_mock_};

  void SetUp() override {
    oc_.context().registerMatcher(std::make_unique<CustomMatcher>(&custom_mock_));
  }

  void ExpectInitBase() {
    // base position and three height scans
    EXPECT_CALL(state_mock_, initBasePosW(ArticulationIs<BasePosWInfo>("robot1")))
        .Times(4)
        .WillRepeatedly(Return(true));
    // base quat and three height scans
    EXPECT_CALL(state_mock_, initBaseQuatW(ArticulationIs<BaseQuatWInfo>("robot1")))
        .Times(4)
        .WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, initBaseLinVelB(ArticulationIs<BaseLinVelBInfo>("robot1")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initBaseAngVelB(ArticulationIs<BaseAngVelBInfo>("robot1")))
        .WillOnce(Return(true));
  }

  void ExpectInitOutput() {
    // Joint outputs from the output joint targets
    EXPECT_CALL(state_mock_, initJointOutput(JointIs<JointOutputInfo>("robot1", "j1")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointOutput(JointIs<JointOutputInfo>("robot1", "j2")))
        .WillOnce(Return(true));
    // Joint position and velocity inputs
    EXPECT_CALL(state_mock_, initJointPosition(JointIs<JointPositionInfo>("robot1", "j1")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointPosition(JointIs<JointPositionInfo>("robot1", "j2")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointPosition(JointIs<JointPositionInfo>("robot1", "j3")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointVelocity(JointIs<JointVelocityInfo>("robot1", "j1")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointVelocity(JointIs<JointVelocityInfo>("robot1", "j2")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initJointVelocity(JointIs<JointVelocityInfo>("robot1", "j3")))
        .WillOnce(Return(true));

    EXPECT_CALL(state_mock_, initSe2Velocity(FrameIs<Se2VelocityInfo>("base_frame")))
        .WillOnce(Return(true));
  }

  void ExpectInitSensors() {
    EXPECT_CALL(state_mock_, initHeightScan(_)).Times(3).WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, initSphericalImage(_)).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initPinholeImage(_)).WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initImuLinearVelocityImu(ImuIs<ImuLinearVelocityImuInfo>("pelvis")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initImuAngularVelocityImu(ImuIs<ImuAngularVelocityImuInfo>("pelvis")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initImuOrientationW(ImuIs<ImuOrientationWInfo>("torso")))
        .WillOnce(Return(true));
  }

  void ExpectInitBody() {
    EXPECT_CALL(state_mock_, initBodyPositionW(BodyIs<BodyPositionWInfo>("box1", "box")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initBodyOrientationW(BodyIs<BodyOrientationWInfo>("box1", "box")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_,
                initBodyLinearVelocityB(BodyIs<BodyLinearVelocityBInfo>("box1", "box")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_,
                initBodyAngularVelocityB(BodyIs<BodyAngularVelocityBInfo>("box1", "box")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initBodyPositionW(BodyIs<BodyPositionWInfo>("box2", "box")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_, initBodyOrientationW(BodyIs<BodyOrientationWInfo>("box2", "box")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_,
                initBodyLinearVelocityB(BodyIs<BodyLinearVelocityBInfo>("box2", "box")))
        .WillOnce(Return(true));
    EXPECT_CALL(state_mock_,
                initBodyAngularVelocityB(BodyIs<BodyAngularVelocityBInfo>("box2", "box")))
        .WillOnce(Return(true));
  }

  void ExpectInitCommands() {
    // SE2 velocity commands from inputs - check range configuration
    EXPECT_CALL(command_mock_,
                initSe2Velocity(AllOf(Field(&Se2VelocityCommandInfo::command_name, "vel"),
                                      Field(&Se2VelocityCommandInfo::ranges, Eq(std::nullopt)))))
        .WillOnce(Return(true));
    EXPECT_CALL(command_mock_, initSe2Velocity(AllOf(
                                   Field(&Se2VelocityCommandInfo::command_name, "vel_with_range"),
                                   HasRanges(kRanges))))
        .WillOnce(Return(true));
    EXPECT_CALL(command_mock_, initSe3Pose(Field(&Se3PoseCommandInfo::command_name, "pose")))
        .WillOnce(Return(true));
    EXPECT_CALL(command_mock_,
                initBooleanSelector(Field(&BooleanSelectorCommandInfo::command_name, "selector")))
        .WillOnce(Return(true));
    EXPECT_CALL(command_mock_, initFloatValue(Field(&FloatValueCommandInfo::command_name, "value")))
        .WillOnce(Return(true));
    EXPECT_CALL(command_mock_, initJointPosition(JointPositionCommandIs("arm", "j1")))
        .WillOnce(Return(true));
    EXPECT_CALL(command_mock_, initJointPosition(JointPositionCommandIs("arm", "j2")))
        .WillOnce(Return(true));
  }

  void ExpectInitCustom() {
    EXPECT_CALL(custom_mock_, initExtensibleCommand("extensible_data")).WillOnce(Return(true));
  }

  void ExpectSetOutput() {
    EXPECT_CALL(state_mock_,
                setJointPosition(AllOf(Field(&SetJointPositionInfo::articulation_name, "robot1"),
                                       Field(&SetJointPositionInfo::joint_name, "j1"))))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_,
                setJointPosition(AllOf(Field(&SetJointPositionInfo::articulation_name, "robot1"),
                                       Field(&SetJointPositionInfo::joint_name, "j2"))))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_,
                setJointVelocity(AllOf(Field(&SetJointVelocityInfo::articulation_name, "robot1"),
                                       Field(&SetJointVelocityInfo::joint_name, "j1"))))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_,
                setJointVelocity(AllOf(Field(&SetJointVelocityInfo::articulation_name, "robot1"),
                                       Field(&SetJointVelocityInfo::joint_name, "j2"))))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_,
                setJointEffort(AllOf(Field(&SetJointEffortInfo::articulation_name, "robot1"),
                                     Field(&SetJointEffortInfo::joint_name, "j1"))))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_,
                setJointEffort(AllOf(Field(&SetJointEffortInfo::articulation_name, "robot1"),
                                     Field(&SetJointEffortInfo::joint_name, "j2"))))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_,
                setJointPGain(AllOf(Field(&SetJointPGainInfo::articulation_name, "robot1"),
                                    Field(&SetJointPGainInfo::joint_name, "j1"),
                                    Field(&SetJointPGainInfo::p_gain, 1))))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_,
                setJointPGain(AllOf(Field(&SetJointPGainInfo::articulation_name, "robot1"),
                                    Field(&SetJointPGainInfo::joint_name, "j2"),
                                    Field(&SetJointPGainInfo::p_gain, 2))))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_,
                setJointDGain(AllOf(Field(&SetJointDGainInfo::articulation_name, "robot1"),
                                    Field(&SetJointDGainInfo::joint_name, "j1"),
                                    Field(&SetJointDGainInfo::d_gain, 0.1))))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_,
                setJointDGain(AllOf(Field(&SetJointDGainInfo::articulation_name, "robot1"),
                                    Field(&SetJointDGainInfo::joint_name, "j2"),
                                    Field(&SetJointDGainInfo::d_gain, 0.2))))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(state_mock_, setSe2Velocity(Field(&Se2VelocityInfo::frame_name, "base_frame")))
        .WillRepeatedly(Return(true));
  }

  void ExpectReadJointState() {
    EXPECT_CALL(state_mock_, jointPosition(JointIs<JointPositionInfo>("robot1", "j1")))
        .WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock_, jointPosition(JointIs<JointPositionInfo>("robot1", "j2")))
        .WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock_, jointPosition(JointIs<JointPositionInfo>("robot1", "j3")))
        .WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock_, jointVelocity(JointIs<JointVelocityInfo>("robot1", "j1")))
        .WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock_, jointVelocity(JointIs<JointVelocityInfo>("robot1", "j2")))
        .WillRepeatedly(Return(std::make_optional(0)));
    EXPECT_CALL(state_mock_, jointVelocity(JointIs<JointVelocityInfo>("robot1", "j3")))
        .WillRepeatedly(Return(std::make_optional(0)));
  }

  void ExpectReadState() {
    EXPECT_CALL(state_mock_, imuLinearVelocityImu(ImuIs<ImuLinearVelocityImuInfo>("pelvis")))
        .WillRepeatedly(Return(kLinearVelocityData));
    EXPECT_CALL(state_mock_, imuAngularVelocityImu(ImuIs<ImuAngularVelocityImuInfo>("pelvis")))
        .WillRepeatedly(Return(kAngularVelocityData));
    EXPECT_CALL(state_mock_, imuOrientationW(ImuIs<ImuOrientationWInfo>("torso")))
        .WillRepeatedly(Return(kQuaternionData));
    EXPECT_CALL(state_mock_, bodyPositionW(BodyIs<BodyPositionWInfo>("box1", "box")))
        .WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(state_mock_, bodyOrientationW(BodyIs<BodyOrientationWInfo>("box1", "box")))
        .WillRepeatedly(Return(kQuaternionData));
    EXPECT_CALL(state_mock_, bodyLinearVelocityB(BodyIs<BodyLinearVelocityBInfo>("box1", "box")))
        .WillRepeatedly(Return(kLinearVelocityData));
    EXPECT_CALL(state_mock_, bodyAngularVelocityB(BodyIs<BodyAngularVelocityBInfo>("box1", "box")))
        .WillRepeatedly(Return(kAngularVelocityData));
    EXPECT_CALL(state_mock_, bodyPositionW(BodyIs<BodyPositionWInfo>("box2", "box")))
        .WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(state_mock_, bodyOrientationW(BodyIs<BodyOrientationWInfo>("box2", "box")))
        .WillRepeatedly(Return(kQuaternionData));
    EXPECT_CALL(state_mock_, bodyLinearVelocityB(BodyIs<BodyLinearVelocityBInfo>("box2", "box")))
        .WillRepeatedly(Return(kLinearVelocityData));
    EXPECT_CALL(state_mock_, bodyAngularVelocityB(BodyIs<BodyAngularVelocityBInfo>("box2", "box")))
        .WillRepeatedly(Return(kAngularVelocityData));
    EXPECT_CALL(state_mock_, heightScan(Field(&HeightScanInfo::sensor_name, "trail"), _, _))
        .WillRepeatedly(Return(std::make_optional(kTrailScanData.get())));
    EXPECT_CALL(state_mock_, heightScan(Field(&HeightScanInfo::sensor_name, "one"), _, _))
        .WillRepeatedly(Return(std::make_optional(kHeightScanData.get())));
    EXPECT_CALL(state_mock_, heightScan(Field(&HeightScanInfo::sensor_name, "two"), _, _))
        .WillRepeatedly(Return(std::make_optional(kHeightScanData.get())));
    EXPECT_CALL(state_mock_, basePosW(ArticulationIs<BasePosWInfo>("robot1")))
        .WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(state_mock_, baseQuatW(ArticulationIs<BaseQuatWInfo>("robot1")))
        .WillRepeatedly(Return(kQuaternionData));
    EXPECT_CALL(state_mock_, baseLinVelB(ArticulationIs<BaseLinVelBInfo>("robot1")))
        .WillRepeatedly(Return(kLinearVelocityData));
    EXPECT_CALL(state_mock_, baseAngVelB(ArticulationIs<BaseAngVelBInfo>("robot1")))
        .WillRepeatedly(Return(kAngularVelocityData));
    EXPECT_CALL(state_mock_, sphericalImage(Field(&SphericalImageInfo::sensor_name, "one")))
        .WillRepeatedly(Return(std::make_optional(kSphericalImageData.get())));
    EXPECT_CALL(state_mock_, pinholeImage(Field(&PinholeImageInfo::sensor_name, "one")))
        .WillRepeatedly(Return(std::make_optional(kPinholeImageData.get())));
  }

  void ExpectReadCommands() {
    EXPECT_CALL(command_mock_, se2Velocity(Field(&Se2VelocityCommandInfo::command_name, "vel")))
        .WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(command_mock_,
                se2Velocity(Field(&Se2VelocityCommandInfo::command_name, "vel_with_range")))
        .WillRepeatedly(Return(kPositionData));
    EXPECT_CALL(command_mock_, se3Pose(Field(&Se3PoseCommandInfo::command_name, "pose")))
        .WillRepeatedly(Return(kSE3PoseData));
    EXPECT_CALL(command_mock_,
                booleanSelector(Field(&BooleanSelectorCommandInfo::command_name, "selector")))
        .WillRepeatedly(Return(std::make_optional(true)));
    EXPECT_CALL(command_mock_, floatValue(Field(&FloatValueCommandInfo::command_name, "value")))
        .WillRepeatedly(Return(std::make_optional(1.23f)));
    EXPECT_CALL(command_mock_, jointPosition(JointPositionCommandIs("arm", "j1")))
        .WillRepeatedly(Return(std::make_optional(0.0f)));
    EXPECT_CALL(command_mock_, jointPosition(JointPositionCommandIs("arm", "j2")))
        .WillRepeatedly(Return(std::make_optional(0.0f)));
  }

  void ExpectReadCustom() {
    EXPECT_CALL(custom_mock_, extensibleCommand("extensible_data"))
        .WillOnce(Return(std::make_optional(kExtensibleData)));
  }
};

TEST_F(OnnxControllerTest, LoadFails) {
  StrictMock<MockRobotStateInterface> strict_state_mock;
  StrictMock<MockCommandInterface> strict_command_mock;
  NiceMock<MockDataCollectionInterface> nice_data_collection_mock;
  OnnxRLController oc(strict_state_mock, strict_command_mock, nice_data_collection_mock);

  ASSERT_FALSE(oc.create(""));
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
  ExpectInitBody();
  ExpectInitCustom();

  // Initialize
  ASSERT_TRUE(oc_.create(model_path));
  ASSERT_TRUE(oc_.init(false));
  EXPECT_EQ(oc_.context().updateRate(), 10);

  ExpectReadJointState();
  ExpectReadState();
  ExpectReadCommands();
  ExpectReadCustom();

  ExpectSetOutput();

  const uint64_t t0_us = 0;
  ASSERT_TRUE(oc_.update(t0_us));
}

TEST_F(OnnxControllerTest, ReadJointFailure) {
  ExpectInitBase();
  ExpectInitOutput();
  ExpectInitSensors();
  ExpectInitCommands();
  ExpectInitBody();
  ExpectInitCustom();

  ASSERT_TRUE(oc_.create(model_path));
  ASSERT_TRUE(oc_.init(false));

  EXPECT_CALL(state_mock_, jointPosition(JointIs<JointPositionInfo>("robot1", "j1")))
      .WillRepeatedly(Return(std::nullopt));
  EXPECT_CALL(state_mock_, jointPosition(JointIs<JointPositionInfo>("robot1", "j2")))
      .WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointPosition(JointIs<JointPositionInfo>("robot1", "j3")))
      .WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointVelocity(JointIs<JointVelocityInfo>("robot1", "j1")))
      .WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointVelocity(JointIs<JointVelocityInfo>("robot1", "j2")))
      .WillRepeatedly(Return(std::make_optional(0)));
  EXPECT_CALL(state_mock_, jointVelocity(JointIs<JointVelocityInfo>("robot1", "j3")))
      .WillRepeatedly(Return(std::make_optional(0)));

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
  ExpectInitBody();
  ExpectInitCustom();

  ASSERT_TRUE(oc_.create(model_path));
  ASSERT_TRUE(oc_.init(false));

  ExpectReadJointState();
  ExpectReadState();
  ExpectReadCommands();
  ExpectReadCustom();

  // j1 write fails
  EXPECT_CALL(state_mock_,
              setJointPosition(AllOf(Field(&SetJointPositionInfo::articulation_name, "robot1"),
                                     Field(&SetJointPositionInfo::joint_name, "j1"))))
      .WillOnce(Return(false));
  EXPECT_CALL(state_mock_,
              setJointPosition(AllOf(Field(&SetJointPositionInfo::articulation_name, "robot1"),
                                     Field(&SetJointPositionInfo::joint_name, "j2"))))
      .WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointVelocity(_)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointEffort(_)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointPGain(_)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setJointDGain(_)).WillRepeatedly(Return(true));
  EXPECT_CALL(state_mock_, setSe2Velocity(_)).WillRepeatedly(Return(true));

  EXPECT_FALSE(oc_.update(0));
}

}  // namespace exploy::control
