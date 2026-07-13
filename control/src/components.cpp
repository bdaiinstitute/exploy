// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include <optional>
#include <string>

#include "exploy/components.hpp"

namespace exploy::control {

namespace {

template <typename T>
void copyToBuffer(std::span<const T> from, std::span<T> to) {
  assert(to.size() == from.size() && "Buffer size must match input size.");
  std::copy(from.begin(), from.end(), to.begin());
}

template <typename T>
void copyToBuffer(std::span<T> from, std::span<T> to) {
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

}  // namespace

// Implementation of IMULinearVelocityInput methods
IMULinearVelocityInput::IMULinearVelocityInput(const std::string& key, const std::string& imu_name)
    : Input("IMULinearVelocityInput"), key_(key), imu_name_(imu_name) {}

bool IMULinearVelocityInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initImuLinearVelocityImu({.imu_name = imu_name_});
}

bool IMULinearVelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                  CommandInterface&) {
  auto maybe_linvel = state.imuLinearVelocityImu({.imu_name = imu_name_});
  if (!maybe_linvel.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_linvel.value(), maybe_buffer.value());
  return true;
}

// Implementation of IMUAngularVelocityInput methods
IMUAngularVelocityInput::IMUAngularVelocityInput(const std::string& key,
                                                 const std::string& imu_name)
    : Input("IMUAngularVelocityInput"), key_(key), imu_name_(imu_name) {}

bool IMUAngularVelocityInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initImuAngularVelocityImu({.imu_name = imu_name_});
}

bool IMUAngularVelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                   CommandInterface&) {
  auto maybe_angvel = state.imuAngularVelocityImu({.imu_name = imu_name_});
  if (!maybe_angvel.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_angvel.value(), maybe_buffer.value());
  return true;
}

IMUOrientationInput::IMUOrientationInput(const std::string& key, const std::string& imu_name)
    : Input("IMUOrientationInput"), key_(key), imu_name_(imu_name) {}

bool IMUOrientationInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initImuOrientationW({.imu_name = imu_name_});
}

bool IMUOrientationInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                               CommandInterface&) {
  auto maybe_quaternion = state.imuOrientationW({.imu_name = imu_name_});
  if (!maybe_quaternion.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_quaternion.value(), maybe_buffer.value());
  return true;
}

// Implementation of JointPositionInput methods
JointPositionInput::JointPositionInput(const std::string& key, const std::string& articulation_name,
                                       const std::vector<std::string>& joint_names)
    : Input("JointPositionInput"),
      key_(key),
      articulation_name_(articulation_name),
      joint_names_(joint_names) {}

bool JointPositionInput::init(RobotStateInterface& state, CommandInterface&) {
  for (const auto& joint_name : joint_names_) {
    if (!state.initJointPosition(
            {.articulation_name = articulation_name_, .joint_name = joint_name}))
      return false;
  }
  return true;
}

bool JointPositionInput::read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface&) {
  std::vector<double> positions;
  if (joint_names_.empty()) return false;
  for (const auto& joint_name : joint_names_) {
    auto maybe_pos =
        state.jointPosition({.articulation_name = articulation_name_, .joint_name = joint_name});
    if (!maybe_pos.has_value()) return false;
    positions.push_back(maybe_pos.value());
  }
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(positions, maybe_buffer.value());
  return true;
}

// Implementation of JointVelocityInput methods
JointVelocityInput::JointVelocityInput(const std::string& key, const std::string& articulation_name,
                                       const std::vector<std::string>& joint_names)
    : Input("JointVelocityInput"),
      key_(key),
      articulation_name_(articulation_name),
      joint_names_(joint_names) {}

bool JointVelocityInput::init(RobotStateInterface& state, CommandInterface&) {
  for (const auto& joint_name : joint_names_) {
    if (!state.initJointVelocity(
            {.articulation_name = articulation_name_, .joint_name = joint_name}))
      return false;
  }
  return true;
}

bool JointVelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface&) {
  std::vector<double> velocities;
  for (const auto& joint_name : joint_names_) {
    auto maybe_vel =
        state.jointVelocity({.articulation_name = articulation_name_, .joint_name = joint_name});
    if (!maybe_vel.has_value()) return false;
    velocities.push_back(maybe_vel.value());
  }
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(velocities, maybe_buffer.value());
  return true;
}

// Implementation of BasePositionInput methods
BasePositionInput::BasePositionInput(const std::string& key, const std::string& articulation_name)
    : Input("BasePositionInput"), key_(key), articulation_name_(articulation_name) {}

bool BasePositionInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBasePosW({.articulation_name = articulation_name_});
}

bool BasePositionInput::read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface&) {
  auto maybe_pos = state.basePosW({.articulation_name = articulation_name_});
  if (!maybe_pos.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_pos.value(), maybe_buffer.value());
  return true;
}

// Implementation of BaseOrientationInput methods
BaseOrientationInput::BaseOrientationInput(const std::string& key,
                                           const std::string& articulation_name)
    : Input("BaseOrientationInput"), key_(key), articulation_name_(articulation_name) {}

bool BaseOrientationInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBaseQuatW({.articulation_name = articulation_name_});
}

bool BaseOrientationInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                CommandInterface&) {
  auto maybe_quat = state.baseQuatW({.articulation_name = articulation_name_});
  if (!maybe_quat.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_quat.value(), maybe_buffer.value());
  return true;
}

// Implementation of BaseLinearVelocityInput methods
BaseLinearVelocityInput::BaseLinearVelocityInput(const std::string& key,
                                                 const std::string& articulation_name)
    : Input("BaseLinearVelocityInput"), key_(key), articulation_name_(articulation_name) {}

bool BaseLinearVelocityInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBaseLinVelB({.articulation_name = articulation_name_});
}

bool BaseLinearVelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                   CommandInterface&) {
  auto maybe_vel = state.baseLinVelB({.articulation_name = articulation_name_});
  if (!maybe_vel.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_vel.value(), maybe_buffer.value());
  return true;
}

// Implementation of BaseAngularVelocityInput methods
BaseAngularVelocityInput::BaseAngularVelocityInput(const std::string& key,
                                                   const std::string& articulation_name)
    : Input("BaseAngularVelocityInput"), key_(key), articulation_name_(articulation_name) {}

bool BaseAngularVelocityInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBaseAngVelB({.articulation_name = articulation_name_});
}

bool BaseAngularVelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                    CommandInterface&) {
  auto maybe_vel = state.baseAngVelB({.articulation_name = articulation_name_});
  if (!maybe_vel.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_vel.value(), maybe_buffer.value());
  return true;
}

// Implementation of JointTargetOutput methods
JointTargetOutput::JointTargetOutput(const std::string& pos_key, const std::string& vel_key,
                                     const std::string& eff_key,
                                     const std::string& articulation_name,
                                     const metadata::JointOutputMetadata& metadata)
    : Output("JointTargetOutput"),
      pos_key_(pos_key),
      vel_key_(vel_key),
      eff_key_(eff_key),
      articulation_name_(articulation_name),
      metadata_(metadata) {}

bool JointTargetOutput::init(RobotStateInterface& state, CommandInterface&) {
  for (const auto& joint_name : metadata_.names) {
    if (!state.initJointOutput({.articulation_name = articulation_name_, .joint_name = joint_name}))
      return false;
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

    if (!state.setJointPosition({.articulation_name = articulation_name_,
                                 .joint_name = joint_name,
                                 .position = maybe_pos_buffer.value()[i]})) {
      LOG_STREAM(ERROR, fmt::format("Failed to set position of joint '{}'", joint_name));
      return false;
    }

    if (!state.setJointVelocity({.articulation_name = articulation_name_,
                                 .joint_name = joint_name,
                                 .velocity = maybe_vel_buffer.value()[i]})) {
      LOG_STREAM(ERROR, fmt::format("Failed to set velocity of joint '{}'", joint_name));
      return false;
    }

    if (!state.setJointEffort({.articulation_name = articulation_name_,
                               .joint_name = joint_name,
                               .effort = maybe_eff_buffer.value()[i]})) {
      LOG_STREAM(ERROR, fmt::format("Failed to set effort of joint '{}'", joint_name));
      return false;
    }

    if (!state.setJointPGain({.articulation_name = articulation_name_,
                              .joint_name = joint_name,
                              .p_gain = metadata_.stiffness.at(i)})) {
      LOG_STREAM(ERROR, fmt::format("Failed to set p-gain of joint '{}'", joint_name));
      return false;
    }

    if (!state.setJointDGain({.articulation_name = articulation_name_,
                              .joint_name = joint_name,
                              .d_gain = metadata_.damping.at(i)})) {
      LOG_STREAM(ERROR, fmt::format("Failed to set d-gain of joint '{}'", joint_name));
      return false;
    }
  }
  return true;
}

SE2VelocityOutput::SE2VelocityOutput(const std::string& key,
                                     const metadata::Se2VelocityOutputMetadata& metadata)
    : Output("SE2VelocityOutput"), key_(key), metadata_(metadata) {}

bool SE2VelocityOutput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initSe2Velocity({.frame_name = metadata_.target_frame});
}

bool SE2VelocityOutput::write(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface&) {
  auto maybe_buffer = runtime.outputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;

  auto buffer = maybe_buffer.value();
  if (buffer.size() != 3) return false;  // x, y, yaw velocities

  const double vx = buffer[0];
  const double vy = buffer[1];
  const double wz = buffer[2];

  if (!state.setSe2Velocity({.frame_name = metadata_.target_frame, .velocity = {vx, vy, wz}})) {
    constexpr auto msg = "Failed to set se(2) target velocity of frame '{}'";
    LOG_STREAM(ERROR, fmt::format(msg, metadata_.target_frame));
    return false;
  }

  return true;
}

// Implementation of HeightScanInput methods
HeightScanInput::HeightScanInput(const std::string& key, const std::string& sensor_name,
                                 const std::unordered_set<std::string>& layer_names,
                                 const metadata::HeightScanMetadata& metadata)
    : Input("HeightScanInput"),
      key_(key),
      articulation_name_(metadata.articulation_name),
      scan_info_{
          .sensor_name = sensor_name,
          .pattern =
              HeightScanPattern{
                  .size = Eigen::Vector2d(metadata.size_x, metadata.size_y),
                  .resolution = metadata.resolution,
                  .offset = Eigen::Vector2d(metadata.offset_x, metadata.offset_y),
              },
          .layer_names = layer_names,
      } {}

bool HeightScanInput::init(RobotStateInterface& state, CommandInterface&) {
  if (!state.initBasePosW({.articulation_name = articulation_name_})) {
    LOG_STREAM(ERROR, "Failed to initialize base position for HeightScanInput");
    return false;
  };
  if (!state.initBaseQuatW({.articulation_name = articulation_name_})) {
    LOG_STREAM(ERROR, "Failed to initialize base orientation for HeightScanInput");
    return false;
  };
  return state.initHeightScan(scan_info_);
}

bool HeightScanInput::read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface&) {
  auto maybe_base_pos = state.basePosW({.articulation_name = articulation_name_});
  if (!maybe_base_pos.has_value()) {
    LOG_STREAM(ERROR, "Failed to get base position for HeightScanInput");
    return false;
  }
  auto maybe_base_quat = state.baseQuatW({.articulation_name = articulation_name_});
  if (!maybe_base_quat.has_value()) {
    LOG_STREAM(ERROR, "Failed to get base orientation for HeightScanInput");
    return false;
  }
  auto maybe_scan = state.heightScan(scan_info_, maybe_base_pos.value(), maybe_base_quat.value());
  if (!maybe_scan.has_value()) {
    LOG_STREAM(ERROR, "Failed to get height scan data for HeightScanInput");
    return false;
  }
  for (const auto& layer_name : scan_info_.layer_names) {
    auto maybe_buffer = runtime.inputBuffer<float>(fmt::format("{}.{}", key_, layer_name));
    if (!maybe_buffer.has_value()) {
      LOG_STREAM(ERROR, fmt::format("Failed to get input buffer {}.{}", key_, layer_name));
      return false;
    }
    copyToBuffer(maybe_scan.value()->float_layers.at(layer_name), maybe_buffer.value());
  }
  return true;
}

// Implementation of SphericalImageInput methods
SphericalImageInput::SphericalImageInput(const std::string& key, const std::string& sensor_name,
                                         const std::unordered_set<std::string>& channel_names,
                                         const metadata::SphericalImageMetadata& metadata)
    : Input("SphericalImageInput"),
      key_(key),
      info_{
          .sensor_name = sensor_name,
          .v_res = static_cast<int>(metadata.v_res),
          .h_res = static_cast<int>(metadata.h_res),
          .v_fov_min_deg = metadata.v_fov_min_deg,
          .v_fov_max_deg = metadata.v_fov_max_deg,
          .channel_names = channel_names,
      } {}

bool SphericalImageInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initSphericalImage(info_);
}

bool SphericalImageInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                               CommandInterface&) {
  auto maybe_image = state.sphericalImage(info_);
  if (!maybe_image.has_value()) {
    LOG_STREAM(ERROR, "Failed to get spherical image data for SphericalImageInput");
    return false;
  }
  for (const auto& channel_name : info_.channel_names) {
    auto maybe_buffer = runtime.inputBuffer<float>(fmt::format("{}.{}", key_, channel_name));
    if (!maybe_buffer.has_value()) {
      LOG_STREAM(ERROR, fmt::format("Failed to get input buffer {}.{}", key_, channel_name));
      return false;
    }
    copyToBuffer(maybe_image.value()->float_channels.at(channel_name), maybe_buffer.value());
  }
  return true;
}

// Implementation of PinholeImageInput methods
PinholeImageInput::PinholeImageInput(const std::string& key, const std::string& sensor_name,
                                     const std::unordered_set<std::string>& channel_names,
                                     const metadata::PinholeImageMetadata& metadata)
    : Input("PinholeImageInput"),
      key_(key),
      info_{
          .sensor_name = sensor_name,
          .width = metadata.width,
          .height = metadata.height,
          .fx = metadata.fx,
          .fy = metadata.fy,
          .cx = metadata.cx,
          .cy = metadata.cy,
          .channel_names = channel_names,
      } {}

bool PinholeImageInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initPinholeImage(info_);
}

bool PinholeImageInput::read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface&) {
  auto maybe_image = state.pinholeImage(info_);
  if (!maybe_image.has_value()) {
    LOG_STREAM(ERROR, "Failed to get pinhole image data for PinholeImageInput");
    return false;
  }
  for (const auto& channel_name : info_.channel_names) {
    auto maybe_buffer = runtime.inputBuffer<float>(fmt::format("{}.{}", key_, channel_name));
    if (!maybe_buffer.has_value()) {
      LOG_STREAM(ERROR, fmt::format("Failed to get input buffer {}.{}", key_, channel_name));
      return false;
    }
    copyToBuffer(maybe_image.value()->float_channels.at(channel_name), maybe_buffer.value());
  }
  return true;
}

// Implementation of CommandSE3PoseInput methods
CommandSE3PoseInput::CommandSE3PoseInput(const std::string& key, const std::string& command_name)
    : Input("CommandSE3PoseInput"), key_(key), command_name_(command_name) {}

bool CommandSE3PoseInput::init(RobotStateInterface& /*state*/, CommandInterface& command) {
  return command.initSe3Pose({.command_name = command_name_});
}

bool CommandSE3PoseInput::read(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                               CommandInterface& command) {
  auto maybe_pose = command.se3Pose({.command_name = command_name_});
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
    : Input("CommandSE2VelocityInput"),
      key_(key),
      command_name_(command_name),
      metadata_(metadata) {}

bool CommandSE2VelocityInput::init(RobotStateInterface& /*state*/, CommandInterface& command) {
  return command.initSe2Velocity({.command_name = command_name_, .ranges = metadata_.ranges});
}

bool CommandSE2VelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                                   CommandInterface& command) {
  auto maybe_pose = command.se2Velocity({.command_name = command_name_});
  if (!maybe_pose.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_pose.value(), maybe_buffer.value());
  return true;
}

// Implementation of CommandBooleanInput methods
CommandBooleanInput::CommandBooleanInput(const std::string& key, const std::string& command_name)
    : Input("CommandBooleanInput"), key_(key), command_name_(command_name) {}

bool CommandBooleanInput::init(RobotStateInterface& /*state*/, CommandInterface& command) {
  return command.initBooleanSelector({.command_name = command_name_});
}

bool CommandBooleanInput::read(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                               CommandInterface& command) {
  auto maybe_bool = command.booleanSelector({.command_name = command_name_});
  if (!maybe_bool.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<bool>(key_);
  if (!maybe_buffer.has_value()) return false;
  maybe_buffer.value()[0] = maybe_bool.value();
  return true;
}

// Implementation of CommandJointPositionInput methods
CommandJointPositionInput::CommandJointPositionInput(
    const std::string& key, const std::string& command_name,
    const metadata::JointPositionCommandMetadata& metadata)
    : Input("CommandJointPositionInput"),
      key_(key),
      command_name_(command_name),
      metadata_(metadata) {}

bool CommandJointPositionInput::init(RobotStateInterface& /*state*/, CommandInterface& command) {
  if (metadata_.joint_names.empty()) {
    LOG_STREAM(ERROR,
               "initJointPosition() called with empty joint_names for command: " << command_name_);
    return false;
  }
  for (const auto& joint_name : metadata_.joint_names) {
    if (!command.initJointPosition({.command_name = command_name_, .joint_name = joint_name}))
      return false;
  }
  return true;
}

bool CommandJointPositionInput::read(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                                     CommandInterface& command) {
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  auto buffer = maybe_buffer.value();
  if (buffer.size() != metadata_.joint_names.size()) {
    LOG_STREAM(ERROR, "Buffer size " << buffer.size() << " does not match joint_names size "
                                     << metadata_.joint_names.size()
                                     << " for command: " << command_name_);
    return false;
  }
  for (std::size_t i = 0; i < metadata_.joint_names.size(); ++i) {
    auto maybe_pos = command.jointPosition(
        {.command_name = command_name_, .joint_name = metadata_.joint_names[i]});
    if (!maybe_pos.has_value()) return false;
    buffer[i] = maybe_pos.value();
  }
  return true;
}

// Implementation of CommandFloatInput methods
CommandFloatInput::CommandFloatInput(const std::string& key, const std::string& command_name,
                                     const metadata::FloatCommandMetadata& metadata)
    : Input("CommandFloatInput"), key_(key), command_name_(command_name), metadata_(metadata) {}

bool CommandFloatInput::init(RobotStateInterface& /*state*/, CommandInterface& command) {
  return command.initFloatValue({.command_name = command_name_, .range = metadata_.range});
}

bool CommandFloatInput::read(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                             CommandInterface& command) {
  auto maybe_float = command.floatValue({.command_name = command_name_});
  if (!maybe_float.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  maybe_buffer.value()[0] = maybe_float.value();
  return true;
}

// Implementation of methods for body components.
BodyPositionInput::BodyPositionInput(const std::string& key, const std::string& articulation_name,
                                     const std::string& body_name)
    : Input("BodyPositionInput"),
      key_(key),
      articulation_name_(articulation_name),
      body_name_(body_name) {}

bool BodyPositionInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBodyPositionW(
      {.articulation_name = articulation_name_, .body_name = body_name_});
}
bool BodyPositionInput::read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface&) {
  auto maybe_pos =
      state.bodyPositionW({.articulation_name = articulation_name_, .body_name = body_name_});
  if (!maybe_pos.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_pos.value(), maybe_buffer.value());
  return true;
}

BodyOrientationInput::BodyOrientationInput(const std::string& key,
                                           const std::string& articulation_name,
                                           const std::string& body_name)
    : Input("BodyOrientationInput"),
      key_(key),
      articulation_name_(articulation_name),
      body_name_(body_name) {}

bool BodyOrientationInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBodyOrientationW(
      {.articulation_name = articulation_name_, .body_name = body_name_});
}

bool BodyOrientationInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                CommandInterface&) {
  auto maybe_quaternion =
      state.bodyOrientationW({.articulation_name = articulation_name_, .body_name = body_name_});
  if (!maybe_quaternion.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_quaternion.value(), maybe_buffer.value());
  return true;
}

BodyLinearVelocityInput::BodyLinearVelocityInput(const std::string& key,
                                                 const std::string& articulation_name,
                                                 const std::string& body_name)
    : Input("BodyLinearVelocityInput"),
      key_(key),
      articulation_name_(articulation_name),
      body_name_(body_name) {}

bool BodyLinearVelocityInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBodyLinearVelocityB(
      {.articulation_name = articulation_name_, .body_name = body_name_});
}

bool BodyLinearVelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                   CommandInterface&) {
  auto maybe_vel =
      state.bodyLinearVelocityB({.articulation_name = articulation_name_, .body_name = body_name_});
  if (!maybe_vel.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_vel.value(), maybe_buffer.value());
  return true;
}

BodyAngularVelocityInput::BodyAngularVelocityInput(const std::string& key,
                                                   const std::string& articulation_name,
                                                   const std::string& body_name)
    : Input("BodyAngularVelocityInput"),
      key_(key),
      articulation_name_(articulation_name),
      body_name_(body_name) {}

bool BodyAngularVelocityInput::init(RobotStateInterface& state, CommandInterface&) {
  return state.initBodyAngularVelocityB(
      {.articulation_name = articulation_name_, .body_name = body_name_});
}

bool BodyAngularVelocityInput::read(OnnxRuntime& runtime, RobotStateInterface& state,
                                    CommandInterface&) {
  auto maybe_vel = state.bodyAngularVelocityB(
      {.articulation_name = articulation_name_, .body_name = body_name_});
  if (!maybe_vel.has_value()) return false;
  auto maybe_buffer = runtime.inputBuffer<float>(key_);
  if (!maybe_buffer.has_value()) return false;
  copyToBuffer(maybe_vel.value(), maybe_buffer.value());
  return true;
}

// Implementation of StepCountInput methods
StepCountInput::StepCountInput(const std::string& key) : Input("StepCountInput"), key_(key) {}

bool StepCountInput::read(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                          CommandInterface& /*command*/) {
  auto maybe_step_count_buffer = runtime.inputBuffer<int32_t>(key_);
  if (!maybe_step_count_buffer.has_value()) return false;
  ++maybe_step_count_buffer.value()[0];
  return true;
}

MemoryOutput::MemoryOutput(const std::string& key) : Output("MemoryOutput"), key_(key) {}

bool MemoryOutput::init(RobotStateInterface& /*state*/, CommandInterface& /*command*/) {
  return true;
}

bool MemoryOutput::write(OnnxRuntime& runtime, RobotStateInterface& /*state*/,
                         CommandInterface& /*command*/) {
  return runtime.copyOutputToInput(fmt::format("memory.{}.out", key_),
                                   fmt::format("memory.{}.in", key_));
}

}  // namespace exploy::control
