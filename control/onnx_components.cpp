#include <optional>
#include <regex>
#include <string>

#include "onnx_components.hpp"

namespace rai::cs::control::common::onnx {

namespace {

template <typename T>
void copyToBuffer(std::span<const T> from, std::span<T> to) {
  assert(to.size() == from.size() && "Buffer size must match input size.");
  std::copy(from.begin(), from.end(), to.begin());
}

template <typename T, typename U>
void copyToBuffer(std::span<const T> from, std::span<U> to) {
  assert(to.size() == from.size() && "Buffer size must match input size.");
  std::transform(from.begin(), from.end(), to.begin(), [](T val) {
    return static_cast<U>(val);
  });
}

void copyToBuffer(const Eigen::VectorXd& from, std::span<float> to) {
  assert(to.size() == static_cast<std::size_t>(from.size()) &&
         "Buffer size must match input size.");
  std::transform(from.data(), from.data() + from.size(), to.begin(), [](double val) {
    return static_cast<float>(val);
  });
}

void copyToBuffer(const SE3Pose& from, std::span<float> to) {
  assert(to.size() == 7 && "Buffer size must be 7 to hold SE3Pose data.");
  for (int i = 0; i < 3; ++i) {
    to[i] = static_cast<float>(from.position[i]);
  }
  to[3] = static_cast<float>(from.orientation.w());
  to[4] = static_cast<float>(from.orientation.x());
  to[5] = static_cast<float>(from.orientation.y());
  to[6] = static_cast<float>(from.orientation.z());
}

void copyToBuffer(const Quaternion& from, std::span<float> to) {
  assert(to.size() == 4 && "Buffer size must be 4 to hold Quaternion data.");
  to[0] = static_cast<float>(from.w());
  to[1] = static_cast<float>(from.x());
  to[2] = static_cast<float>(from.y());
  to[3] = static_cast<float>(from.z());
}

void copyToBuffer(const std::vector<double>& from, std::span<float> to) {
  assert(to.size() == from.size() && "Buffer size must match input size.");
  std::transform(from.begin(), from.end(), to.begin(), [](double val) {
    return static_cast<float>(val);
  });
}

// void copyToBuffer(std::span<double> from, std::span<float> to) {
//   assert(to.size() == from.size() && "Buffer size must match input size.");
//   std::transform(from.begin(), from.end(), to.begin(), [](double val) {
//     return static_cast<float>(val);
//   });
// }

// void copyToBuffer(std::span<float> from, std::span<float> to) {
//   assert(to.size() == from.size() && "Buffer size must match input size.");
//   std::copy(from.begin(), from.end(), to.begin());
// }

}  // namespace

// Implementation of IMUAngularVelocityInput methods
IMUAngularVelocityInput::IMUAngularVelocityInput(const std::string& key,
                                                 const std::string& imu_name)
    : key_(key), imu_name_(imu_name) {}

bool IMUAngularVelocityInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initImuAngularVelocityImu(imu_name_);
}

bool IMUAngularVelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                   CommandInterface&) {
  auto maybe_angvel = state.imuAngularVelocityImu(imu_name_);
  if (!maybe_angvel.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_angvel.value(), maybe_buffer.value());
  return true;
}

IMUOrientationInput::IMUOrientationInput(const std::string& key, const std::string& imu_name)
    : key_(key), imu_name_(imu_name) {}

bool IMUOrientationInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initImuOrientationW(imu_name_);
}

bool IMUOrientationInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                               CommandInterface&) {
  auto maybe_quaternion = state.imuOrientationW(imu_name_);
  if (!maybe_quaternion.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_quaternion.value(), maybe_buffer.value());
  return true;
}

// Implementation of JointPositionInput methods
JointPositionInput::JointPositionInput(const std::string& key,
                                       const std::vector<std::string>& joint_names)
    : key_(key), joint_names_(joint_names) {}

bool JointPositionInput::init(RobotStateInterface& state, CommandInterface&) {
  for (const auto& joint_name : joint_names_) {
    state.initJointPosition(joint_name);
  }
  return true;
}

bool JointPositionInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                              CommandInterface&) {
  std::vector<double> positions;
  for (const auto& joint_name : joint_names_) {
    auto maybe_pos = state.jointPosition(joint_name);
    if (!maybe_pos.has_value()) return false;
    positions.push_back(maybe_pos.value());
  }
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(positions, maybe_buffer.value());
  return true;
}

// Implementation of JointVelocityInput methods
JointVelocityInput::JointVelocityInput(const std::string& key,
                                       const std::vector<std::string>& joint_names)
    : key_(key), joint_names_(joint_names) {}

bool JointVelocityInput::init(RobotStateInterface& state, CommandInterface&) {
  for (const auto& joint_name : joint_names_) {
    state.initJointVelocity(joint_name);
  }
  return true;
}

bool JointVelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                              CommandInterface&) {
  std::vector<double> velocities;
  for (const auto& joint_name : joint_names_) {
    auto maybe_vel = state.jointVelocity(joint_name);
    if (!maybe_vel.has_value()) return false;
    velocities.push_back(maybe_vel.value());
  }
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(velocities, maybe_buffer.value());
  return true;
}

// Implementation of BasePositionInput methods
BasePositionInput::BasePositionInput(const std::string& key) : key_(key) {}

bool BasePositionInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBasePosW();
}

bool BasePositionInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                             CommandInterface&) {
  auto maybe_pos = state.basePosW();
  if (!maybe_pos.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_pos.value(), maybe_buffer.value());
  return true;
}

// Implementation of BaseOrientationInput methods
BaseOrientationInput::BaseOrientationInput(const std::string& key) : key_(key) {}

bool BaseOrientationInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBaseQuatW();
}

bool BaseOrientationInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                CommandInterface&) {
  auto maybe_quat = state.baseQuatW();
  if (!maybe_quat.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_quat.value(), maybe_buffer.value());
  return true;
}

// Implementation of BaseLinearVelocityInput methods
BaseLinearVelocityInput::BaseLinearVelocityInput(const std::string& key) : key_(key) {}

bool BaseLinearVelocityInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBaseLinVelB();
}

bool BaseLinearVelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                   CommandInterface&) {
  auto maybe_vel = state.baseLinVelB();
  if (!maybe_vel.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_vel.value(), maybe_buffer.value());
  return true;
}

// Implementation of BaseAngularVelocityInput methods
BaseAngularVelocityInput::BaseAngularVelocityInput(const std::string& key) : key_(key) {}

bool BaseAngularVelocityInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBaseAngVelB();
}

bool BaseAngularVelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                    CommandInterface&) {
  auto maybe_vel = state.baseAngVelB();
  if (!maybe_vel.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_vel.value(), maybe_buffer.value());
  return true;
}

// Implementation of JointTargetOutput methods
JointTargetOutput::JointTargetOutput(const std::string& pos_key, const std::string& vel_key,
                                     const std::string& eff_key,
                                     const metadata::JointOutputMetadata& metadata)
    : pos_key_(pos_key), vel_key_(vel_key), eff_key_(eff_key), metadata_(metadata) {}

bool JointTargetOutput::init(RobotStateInterface& state, CommandInterface&) {
  for (const auto& joint_name : metadata_.names) {
    if (!state.initJointOutput(joint_name)) return false;
  }
  return true;
}

bool JointTargetOutput::write(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface&) {
  auto maybe_pos_buffer = runtime.outputBuffer<float>(pos_key_);
  if (!maybe_pos_buffer.has_value()) return false;

  auto maybe_vel_buffer = runtime.outputBuffer<float>(vel_key_);
  if (!maybe_vel_buffer.has_value()) return false;

  auto maybe_eff_buffer = runtime.outputBuffer<float>(eff_key_);
  if (!maybe_eff_buffer.has_value()) return false;

  for (size_t i = 0; i < metadata_.names.size(); ++i) {
    const auto& joint_name = metadata_.names.at(i);

    if (!state.setJointPosition(joint_name, maybe_pos_buffer.value()[i])) {
      GENERIC_LOG_STREAM(ERROR, fmt::format("Failed to set position of joint '{}'", joint_name));
      return false;
    }

    if (!state.setJointVelocity(joint_name, maybe_vel_buffer.value()[i])) {
      GENERIC_LOG_STREAM(ERROR, fmt::format("Failed to set velocity of joint '{}'", joint_name));
      return false;
    }

    if (!state.setJointEffort(joint_name, maybe_eff_buffer.value()[i])) {
      GENERIC_LOG_STREAM(ERROR, fmt::format("Failed to set effort of joint '{}'", joint_name));
      return false;
    }

    if (!state.setJointPGain(joint_name, metadata_.stiffness.at(i))) {
      GENERIC_LOG_STREAM(ERROR, fmt::format("Failed to set p-gain of joint '{}'", joint_name));
      return false;
    }

    if (!state.setJointDGain(joint_name, metadata_.damping.at(i))) {
      GENERIC_LOG_STREAM(ERROR, fmt::format("Failed to set d-gain of joint '{}'", joint_name));
      return false;
    }
  }
  return true;
}

SE2VelocityOutput::SE2VelocityOutput(const std::string& key,
                                     const metadata::Se2VelocityOutputMetadata& metadata)
    : key_(key), metadata_(metadata) {}

bool SE2VelocityOutput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initSe2Velocity(metadata_.target_frame);
}

bool SE2VelocityOutput::write(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface&) {
  auto maybe_buffer = runtime.outputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;

  auto buffer = maybe_buffer.value();
  if (buffer.size() != 3) return false;  // x, y, yaw velocities

  const double vx = buffer[0];
  const double vy = buffer[1];
  const double wz = buffer[2];

  if (!state.setSe2Velocity(metadata_.target_frame, {vx, vy, wz})) {
    constexpr auto msg = "Failed to set se(2) target velocity of frame '{}'";
    GENERIC_LOG_STREAM(ERROR, fmt::format(msg, metadata_.target_frame));
    return false;
  }

  return true;
}

// Implementation of HeightScanInput methods
HeightScanInput::HeightScanInput(const std::string& key, const std::string& sensor_name,
                                 const std::unordered_set<std::string>& layer_names,
                                 const metadata::HeightScanMetadata& metadata)
    : key_(key), sensor_name_(sensor_name), layer_names_(layer_names), metadata_(metadata) {}

bool HeightScanInput::init(RobotStateInterface& state, CommandInterface&) {
  HeightScanConfig config = HeightScanConfig{
      .pattern =
          HeightScanConfig::Pattern{
              .size = Eigen::Vector2d(metadata_.size_x, metadata_.size_y),
              .resolution = metadata_.resolution,
              .offset = Eigen::Vector2d(metadata_.offset_x, metadata_.offset_y),
          },
      .layer_names = layer_names_,
  };
  if (!state.initBasePosW()) {
    GENERIC_LOG_STREAM(ERROR, "Failed to initialize base position for HeightScanInput");
    return false;
  };
  if (!state.initBaseQuatW()) {
    GENERIC_LOG_STREAM(ERROR, "Failed to initialize base orientation for HeightScanInput");
    return false;
  };
  return state.initHeightScan(sensor_name_, config);
}

bool HeightScanInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                           CommandInterface&) {
  auto maybe_base_pos = state.basePosW();
  if (!maybe_base_pos.has_value()) {
    GENERIC_LOG_STREAM(ERROR, "Failed to get base position for HeightScanInput");
    return false;
  }
  auto maybe_base_quat = state.baseQuatW();
  if (!maybe_base_quat.has_value()) {
    GENERIC_LOG_STREAM(ERROR, "Failed to get base orientation for HeightScanInput");
    return false;
  }
  auto maybe_scan =
      state.heightScan(sensor_name_, layer_names_, maybe_base_pos.value(), maybe_base_quat.value());
  if (!maybe_scan.has_value()) {
    GENERIC_LOG_STREAM(ERROR, "Failed to get height scan data for HeightScanInput");
    return false;
  }
  for (const auto& layer_name : layer_names_) {
    auto maybe_buffer = runtime.inputBuffer<float>(fmt::format("{}.{}", key_, layer_name));
    if (!maybe_buffer.has_value()) {
      GENERIC_LOG_STREAM(ERROR, fmt::format("Failed to get input buffer {}.{}", key_, layer_name));
      return false;
    }
    copyToBuffer(maybe_scan.value()->layers.at(layer_name), maybe_buffer.value());
  }
  return true;
}

// Implementation of RangeImageInput methods
RangeImageInput::RangeImageInput(const std::string& key,
                                 const metadata::RangeImageMetadata& metadata)
    : key_(key), metadata_(metadata) {}

bool RangeImageInput::init(RobotStateInterface& state, CommandInterface&) {
  RangeImageConfig config;
  config.v_res = static_cast<int>(metadata_.v_res);
  config.h_res = static_cast<int>(metadata_.h_res);
  config.v_fov_min_deg = metadata_.v_fov_min_deg;
  config.v_fov_max_deg = metadata_.v_fov_max_deg;
  config.unobserved_value = metadata_.unobserved_value;
  return state.initRangeImage(config);
}

bool RangeImageInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                           CommandInterface&) {
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  auto maybe_image = state.rangeImage();
  if (!maybe_image.has_value()) return false;
  copyToBuffer(maybe_image.value(), maybe_buffer.value());
  return true;
}

// Implementation of DepthImageInput methods
DepthImageInput::DepthImageInput(const std::string& key,
                                 const metadata::DepthImageMetadata& metadata)
    : key_(key), metadata_(metadata) {}

bool DepthImageInput::init(RobotStateInterface& state, CommandInterface&) {
  DepthImageConfig config;
  config.width = metadata_.width;
  config.height = metadata_.height;
  config.fx = metadata_.fx;
  config.fy = metadata_.fy;
  config.cx = metadata_.cx;
  config.cy = metadata_.cy;
  return state.initDepthImage(config);
}

bool DepthImageInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                           CommandInterface&) {
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  auto maybe_image = state.depthImage();
  if (!maybe_image.has_value()) return false;
  copyToBuffer(maybe_image.value(), maybe_buffer.value());
  return true;
}

// Implementation of CommandSE3PoseInput methods
CommandSE3PoseInput::CommandSE3PoseInput(const std::string& key, const std::string& command_name)
    : key_(key), command_name_(command_name) {}

bool CommandSE3PoseInput::init(RobotStateInterface& /*state*/, CommandInterface& command) {
  return command.initSe3Pose(command_name_);
}

bool CommandSE3PoseInput::read(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                               CommandInterface& command) {
  auto maybe_pose = command.se3Pose(command_name_);
  if (!maybe_pose.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_pose.value(), maybe_buffer.value());
  return true;
}

// Implementation of CommandSE2VelocityInput methods
CommandSE2VelocityInput::CommandSE2VelocityInput(
    const std::string& key, const std::string& command_name,
    const metadata::SE2VelocityCommandMetadata& metadata)
    : key_(key), command_name_(command_name), metadata_(metadata) {}

bool CommandSE2VelocityInput::init(RobotStateInterface& /*state*/, CommandInterface& command) {
  SE2VelocityConfig config;
  config.ranges = metadata_.ranges;
  return command.initSe2Velocity(command_name_, config);
}

bool CommandSE2VelocityInput::read(OnnxRuntime& /*runtime*/, RobotStateInterface& /*state*/,
                                   CommandInterface& /*command*/) {
  return true;
}

// Implementation of CommandBooleanInput methods
CommandBooleanInput::CommandBooleanInput(const std::string& key, const std::string& command_name)
    : key_(key), command_name_(command_name) {}

bool CommandBooleanInput::init(RobotStateInterface& /*state*/, CommandInterface& command) {
  return command.initBooleanSelector(command_name_);
}

bool CommandBooleanInput::read(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                               CommandInterface& command) {
  auto maybe_bool = command.booleanSelector(command_name_);
  if (!maybe_bool.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<bool>(key_);
  if (!maybe_buffer.has_value()) return false;
  maybe_buffer.value()[0] = maybe_bool.value();
  return true;
}

// Implementation of CommandFloatInput methods
CommandFloatInput::CommandFloatInput(const std::string& key, const std::string& command_name)
    : key_(key), command_name_(command_name) {}

bool CommandFloatInput::init(RobotStateInterface& /*state*/, CommandInterface& command) {
  return command.initFloatValue(command_name_);
}

bool CommandFloatInput::read(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                             CommandInterface& command) {
  auto maybe_float = command.floatValue(command_name_);
  if (!maybe_float.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  maybe_buffer.value()[0] = maybe_float.value();
  return true;
}

BodyPositionInput::BodyPositionInput(const std::string& key, const std::string& body_name)
    : key_(key), body_name_(body_name) {}

bool BodyPositionInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBodyPositionW(body_name_);
}
bool BodyPositionInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                             CommandInterface&) {
  auto maybe_pos = state.bodyPositionW(body_name_);
  if (!maybe_pos.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_pos.value(), maybe_buffer.value());
  return true;
}

BodyOrientationInput::BodyOrientationInput(const std::string& key, const std::string& body_name)
    : key_(key), body_name_(body_name) {}

bool BodyOrientationInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBodyOrientationW(body_name_);
}

bool BodyOrientationInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                CommandInterface&) {
  auto maybe_quaternion = state.bodyOrientationW(body_name_);
  if (!maybe_quaternion.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_quaternion.value(), maybe_buffer.value());
  return true;
}

// Implementation of StepCountInput methods
StepCountInput::StepCountInput(const std::string& key) : key_(key) {}

bool StepCountInput::read(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                          CommandInterface& /*command*/) {
  auto maybe_step_count_buffer = runtime.inputBuffer<int32_t>(key_);
  if (!maybe_step_count_buffer.has_value()) return false;
  ++maybe_step_count_buffer.value()[0];
  return true;
}

MemoryOutput::MemoryOutput(const std::string& key) : key_(key) {}

bool MemoryOutput::init(RobotStateInterface& /*state*/, CommandInterface& /*command*/) {
  return true;
}

bool MemoryOutput::write(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                         CommandInterface& /*command*/) {
  return runtime.copyOutputToInput(fmt::format("memory.{}.out", key_),
                                   fmt::format("memory.{}.in", key_));
}

}  // namespace rai::cs::control::common::onnx
