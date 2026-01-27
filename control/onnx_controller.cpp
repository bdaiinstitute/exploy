// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#include "onnx_controller.hpp"
#include "logging_utils.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <span>
#include <unordered_set>
#include <vector>

namespace rai::cs::control::common::onnx {

namespace {

template <typename T>
void copyToBuffer(const std::span<T>& from, std::span<T> to) {
  assert(to.size() == from.size() && "Buffer size must match input size.");
  std::copy(from.begin(), from.end(), to.begin());
}

void copyToBuffer(const std::vector<double>& from, std::span<float> to) {
  assert(to.size() == from.size() && "Buffer size must match input size.");
  std::transform(from.begin(), from.end(), to.begin(), [](double val) {
    return static_cast<float>(val);
  });
}

void copyToBuffer(const Eigen::VectorXd& from, std::span<float> to) {
  assert(to.size() == static_cast<std::size_t>(from.size()) && "Buffer size must match input size.");
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

}  // namespace

OnnxRLController::OnnxRLController(RobotStateInterface& state, CommandInterface& command,
                                   operation::common::data_collection::DataCollectionInterface& data_collection)
    : state_(state), command_(command), data_collection_(data_collection) {}

bool OnnxRLController::load(const std::string& onnx_model_path) {
  if (!onnx_model_.initialize(onnx_model_path)) {
    GENERIC_LOG_STREAM(ERROR, "Error creating OnnxEvaluator from policy: " << onnx_model_path);
    return false;
  }

  auto maybe_config = parseOnnxControllerConfig(onnx_model_);
  if (!maybe_config.has_value()) {
    GENERIC_LOG(ERROR, "Error parsing configuration");
    return false;
  }
  config_ = maybe_config.value();
  return true;
}

bool OnnxRLController::init(bool enable_data_collection) {
  if (!initCommands()) {
    GENERIC_LOG(ERROR, "Error initializing commands.");
    return false;
  }

  if (!initSensors()) {
    GENERIC_LOG(ERROR, "Error initializing sensors.");
    return false;
  }

  for (const auto& [key, data] : config_.imu_keys_to_data) {
    if (data.interface == keys::kAngularVelocity) {
      if (!state_.initImuAngularVelocityImu(data.name)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of angular velocity failed for IMU " << data.name);
        return false;
      }
    } else if (data.interface == keys::kOrientation) {
      if (!state_.initImuOrientationW(data.name)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of orientation failed for IMU " << data.name);
        return false;
      }
    } else {
      GENERIC_LOG_STREAM(ERROR, "Unknown IMU interface " << data.interface);
      return false;
    }
  }

  for (const auto& [key, data] : config_.body_keys_to_data) {
    if (data.interface == keys::kPosition) {
      if (!state_.initBodyPositionW(data.name)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of position failed for body " << data.name);
        return false;
      }
    } else if (data.interface == keys::kOrientation) {
      if (!state_.initBodyOrientationW(data.name)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of orientation failed for body " << data.name);
        return false;
      }
    } else if (data.interface == keys::kLinearVelocity) {
      if (!state_.initBodyLinearVelocityB(data.name)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of linear velocity failed for body " << data.name);
        return false;
      }
    } else if (data.interface == keys::kAngularVelocity) {
      if (!state_.initBodyAngularVelocityB(data.name)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of angular velocity failed for body " << data.name);
        return false;
      }
    } else {
      GENERIC_LOG_STREAM(ERROR, "Unknown body interface " << data.interface);
      return false;
    }
  }

  for (const auto& [_, joint_targets_data] : config_.joint_target_keys_to_data) {
    for (const auto& joint_name : joint_targets_data.names) {
      if (!state_.initJointOutput(joint_name)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of joint '" << joint_name << "' failed.");
        return false;
      }
    }
  }

  for (const auto& [command_name, se2_velocity_data] : config_.se2_velocity_keys_to_data) {
    if (se2_velocity_data.target_frame.empty()) continue;
    if (!state_.initSe2Velocity(se2_velocity_data.target_frame)) {
      GENERIC_LOG_STREAM(ERROR, "Initialization of se(2) velocity '" << command_name << "' failed.");
      return false;
    }
  }

  auto maybe_joint_pos_buffer = onnx_model_.inputBuffer<float>(keys::kJointPos);
  if (maybe_joint_pos_buffer.has_value()) {
    if (config_.joint_names.size() != maybe_joint_pos_buffer.value().size()) {
      GENERIC_LOG(ERROR, "Size mismatch for joint position input.");
      return false;
    }
    for (auto& joint_name : config_.joint_names) {
      if (!state_.initJointPosition(joint_name)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of joint position '" << joint_name << "' failed.");
        return false;
      }
    }
  }

  auto maybe_joint_vel_buffer = onnx_model_.inputBuffer<float>(keys::kJointVel);
  if (maybe_joint_vel_buffer.has_value()) {
    if (config_.joint_names.size() != maybe_joint_vel_buffer.value().size()) {
      GENERIC_LOG(ERROR, "Size mismatch for joint velocity input.");
      return false;
    }
    for (auto& joint_name : config_.joint_names) {
      if (!state_.initJointVelocity(joint_name)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of joint velocity '" << joint_name << "' failed.");
        return false;
      }
    }
  }

  if (onnx_model_.inputBuffer<float>(keys::kPosBaseInWorld).has_value()) {
    if (!state_.initBasePosW()) {
      GENERIC_LOG(ERROR, "Initialization of base position in world failed.");
      return false;
    }
  }

  if (onnx_model_.inputBuffer<float>(keys::kQuatBaseInWorld).has_value()) {
    if (!state_.initBaseQuatW()) {
      GENERIC_LOG(ERROR, "Initialization of base quaternion in world failed.");
      return false;
    }
  }

  if (onnx_model_.inputBuffer<float>(keys::kLinVelBaseInBase).has_value()) {
    if (!state_.initBaseLinVelB()) {
      GENERIC_LOG(ERROR, "Initialization of linear velocity in base failed.");
      return false;
    }
  }

  if (onnx_model_.inputBuffer<float>(keys::kAngVelBaseInBase).has_value()) {
    if (!state_.initBaseAngVelB()) {
      GENERIC_LOG(ERROR, "Initialization of angular velocity in base failed.");
      return false;
    }
  }

  if (enable_data_collection) {
    for (const auto& name : onnx_model_.inputNames()) {
      auto maybe_buffer = onnx_model_.inputBuffer<float>(name);
      if (!maybe_buffer.has_value()) continue;
      if (!data_collection_.registerDataSource("model/input/" + name, maybe_buffer.value())) {
        GENERIC_LOG_STREAM(ERROR, "Registering data source for model/input/" << name << " failed.");
        return false;
      }
    }
    for (const auto& name : onnx_model_.outputNames()) {
      auto maybe_buffer = onnx_model_.outputBuffer<float>(name);
      if (!maybe_buffer.has_value()) continue;
      if (!data_collection_.registerDataSource("model/output/" + name, maybe_buffer.value())) {
        GENERIC_LOG_STREAM(ERROR, "Registering data source for model/output/" << name << " failed.");
        return false;
      }
    }
    if (!data_collection_.registerDataSource("model/inference/duration_s", inference_duration_s_)) {
      GENERIC_LOG(ERROR, "Registering data source for inference duration failed.");
      return false;
    }
  }

  reset();

  return true;
}

void OnnxRLController::reset() {
  onnx_model_.resetBuffers();
}

bool OnnxRLController::initCommands() {
  for (const auto& [command_name, type] : config_.command_name_to_type) {
    if (type == keys::kSE2Velocity) {
      if (config_.se2_velocity_keys_to_data.contains(command_name) == false) {
        GENERIC_LOG_STREAM(ERROR, "No se(2) velocity data found for command " << command_name);
        return false;
      }
      SE2VelocityConfig cfg;
      cfg.ranges = config_.se2_velocity_keys_to_data[command_name].ranges;
      if (!command_.initSe2Velocity(command_name, cfg)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of se(2) velocity command " << command_name << " failed.");
        return false;
      }
    } else if (type == keys::kSE3Pose) {
      if (!command_.initSe3Pose(command_name)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of SE(3) pose command " << command_name << " failed.");
        return false;
      }
    } else if (type == keys::kBooleanSelector) {
      if (!command_.initBooleanSelector(command_name)) {
        GENERIC_LOG_STREAM(ERROR, "Initialization of boolean selector command " << command_name << " failed.");
        return false;
      }
    } else {
      GENERIC_LOG_STREAM(ERROR, "Unknown command type " << type);
      return false;
    }
  }
  return true;
}

bool OnnxRLController::readCommands() {
  for (const auto& [command_name, type] : config_.command_name_to_type) {
    if (type == keys::kSE2Velocity) {
      auto maybe_se2_velocity = command_.se2Velocity(command_name);
      if (!maybe_se2_velocity.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not read se(2) velocity for " << command_name);
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(command_name);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << command_name);
        return false;
      }
      copyToBuffer(maybe_se2_velocity.value(), maybe_buffer.value());
    } else if (type == keys::kSE3Pose) {
      auto maybe_se3_pose = command_.se3Pose(command_name);
      if (!maybe_se3_pose.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not read SE(3) pose for " << command_name);
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(command_name);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << command_name);
        return false;
      }
      copyToBuffer(maybe_se3_pose.value(), maybe_buffer.value());
    } else if (type == keys::kBooleanSelector) {
      auto maybe_boolean_selector = command_.booleanSelector(command_name);
      if (!maybe_boolean_selector.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not read boolean selector for " << command_name);
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(command_name);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << command_name);
        return false;
      }
      maybe_buffer.value()[0] = maybe_boolean_selector.value() ? 1.0 : 0.0;
    } else {
      GENERIC_LOG_STREAM(ERROR, "Unknown command type " << type);
      return false;
    }
  }
  return true;
}

bool OnnxRLController::readIMU() {
  for (const auto& [key, data] : config_.imu_keys_to_data) {
    if (data.interface == keys::kAngularVelocity) {
      auto maybe_imu_angular_velocity = state_.imuAngularVelocityImu(data.name);
      if (!maybe_imu_angular_velocity.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not read angular velocity of IMU " << data.name);
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(key);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << key);
        return false;
      }
      copyToBuffer(maybe_imu_angular_velocity.value(), maybe_buffer.value());
    } else if (data.interface == keys::kOrientation) {
      auto maybe_imu_orientation = state_.imuOrientationW(data.name);
      if (!maybe_imu_orientation.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not read orientation of IMU " << data.name);
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(key);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << key);
        return false;
      }
      copyToBuffer(maybe_imu_orientation.value(), maybe_buffer.value());
    } else {
      GENERIC_LOG_STREAM(ERROR, "Unknown IMU interface " << data.interface);
      return false;
    }
  }
  return true;
}

bool OnnxRLController::readBody() {
  for (const auto& [key, data] : config_.body_keys_to_data) {
    if (data.interface == keys::kPosition) {
      auto maybe_body_position = state_.bodyPositionW(data.name);
      if (!maybe_body_position.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could position of body " << data.name);
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(key);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << key);
        return false;
      }
      copyToBuffer(maybe_body_position.value(), maybe_buffer.value());
    } else if (data.interface == keys::kOrientation) {
      auto maybe_body_orientation = state_.bodyOrientationW(data.name);
      if (!maybe_body_orientation.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not read orientation of body " << data.name);
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(key);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << key);
        return false;
      }
      copyToBuffer(maybe_body_orientation.value(), maybe_buffer.value());
    } else if (data.interface == keys::kLinearVelocity) {
      auto maybe_body_linear_velocity = state_.bodyLinearVelocityB(data.name);
      if (!maybe_body_linear_velocity.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not read linear velocity of body " << data.name);
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(key);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << key);
        return false;
      }
      copyToBuffer(maybe_body_linear_velocity.value(), maybe_buffer.value());
    } else if (data.interface == keys::kAngularVelocity) {
      auto maybe_body_angular_velocity = state_.bodyAngularVelocityB(data.name);
      if (!maybe_body_angular_velocity.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not read angular velocity of body " << data.name);
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(key);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << key);
        return false;
      }
      copyToBuffer(maybe_body_angular_velocity.value(), maybe_buffer.value());
    } else {
      GENERIC_LOG_STREAM(ERROR, "Unknown body interface " << data.interface);
      return false;
    }
  }
  return true;
}

bool OnnxRLController::readJointState() {
  auto maybe_joint_pos_buffer_model = onnx_model_.inputBuffer<float>(keys::kJointPos);
  auto maybe_joint_vel_buffer = onnx_model_.inputBuffer<float>(keys::kJointVel);

  for (size_t i = 0; i < config_.joint_names.size(); ++i) {
    if (maybe_joint_pos_buffer_model.has_value()) {
      const auto maybe_joint_pos = state_.jointPosition(config_.joint_names[i]);
      if (!maybe_joint_pos.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not read joint position of joint '" << config_.joint_names[i] << "'");
        return false;
      }
      maybe_joint_pos_buffer_model.value()[i] = maybe_joint_pos.value();
    }

    if (maybe_joint_vel_buffer.has_value()) {
      const auto maybe_joint_vel = state_.jointVelocity(config_.joint_names[i]);
      if (!maybe_joint_vel.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not read joint velocity of joint '" << config_.joint_names[i] << "'");
        return false;
      }
      maybe_joint_vel_buffer.value()[i] = maybe_joint_vel.value();
    }
  }
  return true;
}

bool OnnxRLController::writeActions() {
  auto maybe_actions_buffer = onnx_model_.outputBuffer<float>(keys::kActions);
  if (!maybe_actions_buffer.has_value()) return true;
  return true;
}

bool OnnxRLController::writeMemory() {
  for (const auto& [in_key, out_key] : config_.memorized_outputs_to_key) {
    if (!onnx_model_.copyOutputToInput(out_key, in_key)) return false;
  }
  return true;
}

bool OnnxRLController::writeOutputs() {
  for (const auto& [output_name, output_type] : config_.output_name_to_type) {
    if (output_type == keys::kJointTargets) {
      const auto& joint_target_data = config_.joint_target_keys_to_data[output_name];

      auto maybe_pos_buffer = onnx_model_.outputBuffer<float>(joint_target_data.pos_name);
      auto maybe_vel_buffer = onnx_model_.outputBuffer<float>(joint_target_data.vel_name);
      auto maybe_eff_buffer = onnx_model_.outputBuffer<float>(joint_target_data.eff_name);
      if (!maybe_pos_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get position buffer for '" << joint_target_data.pos_name << "'");
        return false;
      }
      if (!maybe_vel_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get velocity buffer for '" << joint_target_data.vel_name << "'");
        return false;
      }
      if (!maybe_eff_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get effort buffer for '" << joint_target_data.eff_name << "'");
        return false;
      }

      for (size_t i = 0; i < joint_target_data.names.size(); ++i) {
        const auto& joint_name = joint_target_data.names.at(i);

        if (!state_.setJointPosition(joint_name, maybe_pos_buffer.value()[i])) {
          GENERIC_LOG_STREAM(ERROR, "Failed to set position of joint '" << joint_name << "'");
          return false;
        }

        if (!state_.setJointVelocity(joint_name, maybe_vel_buffer.value()[i])) {
          GENERIC_LOG_STREAM(ERROR, "Failed to set velocity of joint '" << joint_name << "'");
          return false;
        }

        if (!state_.setJointEffort(joint_name, maybe_eff_buffer.value()[i])) {
          GENERIC_LOG_STREAM(ERROR, "Failed to set effort of joint '" << joint_name << "'");
          return false;
        }

        if (!state_.setJointPGain(joint_name, joint_target_data.stiffness.at(i))) {
          GENERIC_LOG_STREAM(ERROR, "Failed to set p-gain of joint '" << joint_name << "'");
          return false;
        }

        if (!state_.setJointDGain(joint_name, joint_target_data.damping.at(i))) {
          GENERIC_LOG_STREAM(ERROR, "Failed to set d-gain of joint '" << joint_name << "'");
          return false;
        }
      }
    } else if (output_type == keys::kSE2Velocity) {
      auto maybe_buffer = onnx_model_.outputBuffer<float>(output_name);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get output buffer for '" << output_name << "'");
        return false;
      }
      const auto& se2_base_velocity_data = config_.se2_velocity_keys_to_data.at(output_name);
      const double vx = maybe_buffer.value()[0];
      const double vy = maybe_buffer.value()[1];
      const double ωz = maybe_buffer.value()[2];
      if (!state_.setSe2Velocity(se2_base_velocity_data.target_frame, {vx, vy, ωz})) {
        GENERIC_LOG_STREAM(ERROR, "Failed to set se(2) target velocity of frame '" << se2_base_velocity_data.target_frame << "'");
        return false;
      }
    } else {
      GENERIC_LOG_STREAM(ERROR, "Unknown output type '" << output_type << "' found for '" << output_name << "'");
      return false;
    }
  }

  return true;
}

bool OnnxRLController::readBasePosInWorld() {
  auto maybe_buffer = onnx_model_.inputBuffer<float>(keys::kPosBaseInWorld);
  if (!maybe_buffer.has_value()) return true;
  auto maybe_base_pos_w = state_.basePosW();
  if (!maybe_base_pos_w.has_value()) {
    GENERIC_LOG(ERROR, "Could not read base position");
    return false;
  }
  copyToBuffer(maybe_base_pos_w.value(), maybe_buffer.value());
  return true;
}

bool OnnxRLController::readBaseQuatInWorld() {
  auto maybe_buffer = onnx_model_.inputBuffer<float>(keys::kQuatBaseInWorld);
  if (!maybe_buffer.has_value()) return true;
  auto maybe_base_quat_w = state_.baseQuatW();
  if (!maybe_base_quat_w.has_value()) {
    GENERIC_LOG(ERROR, "Could not read base quaternion");
    return false;
  }
  copyToBuffer(maybe_base_quat_w.value(), maybe_buffer.value());
  return true;
}

bool OnnxRLController::readBaseLinVelInBase() {
  auto maybe_buffer = onnx_model_.inputBuffer<float>(keys::kLinVelBaseInBase);
  if (!maybe_buffer.has_value()) return true;
  auto maybe_value = state_.baseLinVelB();
  if (!maybe_value.has_value()) {
    GENERIC_LOG(ERROR, "Could not read base lin vel");
    return false;
  }
  copyToBuffer(maybe_value.value(), maybe_buffer.value());
  return true;
}

bool OnnxRLController::readBaseAngVelInBase() {
  auto maybe_buffer = onnx_model_.inputBuffer<float>(keys::kAngVelBaseInBase);
  if (!maybe_buffer.has_value()) return true;
  auto maybe_value = state_.baseAngVelB();
  if (!maybe_value.has_value()) {
    GENERIC_LOG(ERROR, "Could not read base ang vel");
    return false;
  }
  copyToBuffer(maybe_value.value(), maybe_buffer.value());
  return true;
}

bool OnnxRLController::readSensors() {
  std::vector<HeightScan>* heightscan = nullptr;
  if (!config_.sensor_key_to_pattern_index.empty()) {
    auto maybe_base_pos_w = state_.basePosW();
    auto maybe_base_quat_w = state_.baseQuatW();
    if (!maybe_base_pos_w.has_value() || !maybe_base_quat_w.has_value()) {
      GENERIC_LOG(ERROR, "Could not read base position or base orientation");
      return false;
    }
    auto maybe_heightscan = state_.heightScan(maybe_base_pos_w.value(), maybe_base_quat_w.value());
    if (!maybe_heightscan.has_value()) {
      GENERIC_LOG(ERROR, "Could not read heightscan");
      return false;
    }
    heightscan = maybe_heightscan.value();
  }

  for (const auto& [sensor_name, type] : config_.sensor_name_to_type) {
    if (type == keys::kRayCaster) {
      size_t pattern_index = config_.sensor_key_to_pattern_index[sensor_name];
      if (heightscan == nullptr) {
        GENERIC_LOG_STREAM(ERROR,
                    "Heightscan pattern is defined but heightscan data is not available. This is likely caused due to "
                    "missing or incompatible metadata for sensor " << sensor_name);
        return false;
      }
      if (pattern_index >= heightscan->size()) {
        GENERIC_LOG(
            ERROR,
            "The heightscan was initialized with an incompatible number of patterns. This is likely caused due to "
            "missing or incompatible metadata for the sensors.");
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(sensor_name);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << sensor_name);
        return false;
      }
      copyToBuffer((*heightscan)[pattern_index].height, maybe_buffer.value());
    } else if (type == keys::kLidarRangeImage) {
      auto maybe_range_image = state_.rangeImage();
      if (!maybe_range_image.has_value()) {
        GENERIC_LOG(ERROR, "Could not read range image");
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(sensor_name);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << sensor_name);
        return false;
      }
      copyToBuffer(*maybe_range_image.value(), maybe_buffer.value());
    } else if (type == keys::kTrailRayCaster) {
      size_t pattern_index = config_.sensor_key_to_pattern_index[sensor_name];
      if (heightscan == nullptr || pattern_index >= heightscan->size()) {
        GENERIC_LOG(ERROR, "Heightscan pattern is defined but heightscan data is not available.");
        return false;
      }
      auto scan = (*heightscan)[pattern_index];
      auto height_key = sensor_name + ".height";
      auto maybe_buffer = onnx_model_.inputBuffer<float>(height_key);
      if (maybe_buffer.has_value()) copyToBuffer(scan.height, maybe_buffer.value());
      if (!scan.color.has_value()) {
        GENERIC_LOG(ERROR, "Could not read color from heightscan");
        return false;
      }
      auto maybe_buffer_r = onnx_model_.inputBuffer<float>(sensor_name + ".r");
      auto maybe_buffer_g = onnx_model_.inputBuffer<float>(sensor_name + ".g");
      auto maybe_buffer_b = onnx_model_.inputBuffer<float>(sensor_name + ".b");
      if (!maybe_buffer_g.has_value() || !maybe_buffer_b.has_value() || !maybe_buffer_r.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << sensor_name << " color channels");
        return false;
      }
      copyToBuffer(scan.color.value().r, maybe_buffer_r.value());
      copyToBuffer(scan.color.value().g, maybe_buffer_g.value());
      copyToBuffer(scan.color.value().b, maybe_buffer_b.value());
    } else if (type == keys::kDepthImage) {
      auto maybe_depth_image = state_.depthImage();
      if (!maybe_depth_image.has_value()) {
        GENERIC_LOG(ERROR, "Could not read depth image");
        return false;
      }
      auto maybe_buffer = onnx_model_.inputBuffer<float>(sensor_name);
      if (!maybe_buffer.has_value()) {
        GENERIC_LOG_STREAM(ERROR, "Could not get input buffer for " << sensor_name);
        return false;
      }
      copyToBuffer(*maybe_depth_image.value(), maybe_buffer.value());
    } else {
      GENERIC_LOG_STREAM(ERROR, "Unknown sensor type " << type);
      return false;
    }
  }
  return true;
}

bool OnnxRLController::initSensors() {
  if (config_.heightscan_config.has_value()) {
    if (!state_.initHeightScan(config_.heightscan_config.value())) {
      GENERIC_LOG(ERROR, "Initialization of heightscan failed");
      return false;
    }
    // Initialize base position and orientation as they are needed for heightscan
    if (!state_.initBasePosW()) {
      GENERIC_LOG(ERROR, "Initialization of base position failed.");
      return false;
    }
    if (!state_.initBaseQuatW()) {
      GENERIC_LOG(ERROR, "Initialization of base orientation failed.");
      return false;
    }
  }

  if (config_.range_image_config.has_value()) {
    if (!state_.initRangeImage(config_.range_image_config.value())) {
      GENERIC_LOG(ERROR, "Initialization of range image failed");
      return false;
    }
  }

  if (config_.depth_image_config.has_value()) {
    if (!state_.initDepthImage(config_.depth_image_config.value())) {
      GENERIC_LOG(ERROR, "Initialization of depth image failed");
      return false;
    }
  }

  return true;
}

void OnnxRLController::increaseStepCount() {
  auto maybe_step_count_buffer = onnx_model_.inputBuffer<int32_t>(keys::kStepCount);
  if (!maybe_step_count_buffer.has_value()) return;
  ++maybe_step_count_buffer.value()[0];
}

bool OnnxRLController::update(uint64_t time_us) {
  CS_TRACE_SCOPED_ZONE;
  if (!readJointState()) return false;
  if (!readIMU()) return false;
  if (!readBody()) return false;
  if (!readCommands()) return false;
  if (!readBasePosInWorld()) return false;
  if (!readBaseQuatInWorld()) return false;
  if (!readBaseLinVelInBase()) return false;
  if (!readBaseAngVelInBase()) return false;
  if (!readSensors()) return false;

  auto start_time = std::chrono::high_resolution_clock::now();
  if (!onnx_model_.evaluate()) {
    GENERIC_LOG(ERROR, "Policy evaluation failed.");
    return false;
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  inference_duration_s_ = std::chrono::duration<double>(end_time - start_time).count();

  if (!writeActions()) return false;
  if (!writeOutputs()) return false;
  if (!writeMemory()) return false;
  increaseStepCount();

  if (!data_collection_.collectData(time_us)) {
    GENERIC_LOG(WARN, "Data collection failed.");
  }

  return true;
}

}  // namespace rai::cs::control::common::onnx
