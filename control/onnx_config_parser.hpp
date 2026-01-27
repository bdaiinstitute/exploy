// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "command_interface.hpp"
#include "state_interface.hpp"
#include "onnx_runtime.hpp"

namespace rai::cs::control::common::onnx {

namespace keys {

constexpr char kActions[] = "actions";
constexpr char kAngularVelocity[] = "ang_vel";
constexpr char kAngVelBaseInBase[] = "ang_vel_base_in_base";
constexpr char kBody[] = "body";
constexpr char kBooleanSelector[] = "boolean_selector";
constexpr char kCommands[] = "commands";
constexpr char kDecimation[] = "decimation";
constexpr char kDepthImage[] = "depth_image";
constexpr char kGridPattern[] = "grid_pattern";
constexpr char kIMUData[] = "imu_data";
constexpr char kJointNames[] = "joint_names";
constexpr char kJointPos[] = "joint_pos";
constexpr char kJointTargets[] = "joint_targets";
constexpr char kJointVel[] = "joint_vel";
constexpr char kLidarPattern[] = "lidar_pattern";
constexpr char kLidarRangeImage[] = "lidar_range_image";
constexpr char kLinearVelocity[] = "lin_vel";
constexpr char kLinVelBaseInBase[] = "lin_vel_base_in_base";
constexpr char kMemory[] = "memory";
constexpr char kObservation[] = "obs";
constexpr char kOrientation[] = "quat";
constexpr char kOutputs[] = "outputs";
constexpr char kPolicyDt[] = "policy_dt";
constexpr char kPosBaseInWorld[] = "pos_base_in_w";
constexpr char kPosition[] = "pos";
constexpr char kQuatBaseInWorld[] = "world_Q_base";
constexpr char kRayCaster[] = "ray_caster";
constexpr char kSE2Velocity[] = "se2_velocity";
constexpr char kSE3Pose[] = "se3_pose";
constexpr char kSensors[] = "sensors";
constexpr char kStepCount[] = "step_count";
constexpr char kTrailRayCaster[] = "trail_ray_caster";
constexpr char kType[] = "type";

}  // namespace keys

// Configuration to access body data from state.
struct BodyData {
  // The name of the body.
  std::string name{};
  // The interface of the body data (e.g., "angular_velocity", "orientation", etc.).
  std::string interface {};
};

// Configuration to write joint target data to commands.
struct JointTargetData {
  // The key of the position target output.
  std::string pos_name{};
  // The key of the velocity target output.
  std::string vel_name{};
  // The key of the effort target output.
  std::string eff_name{};
  // The names of the joints.
  std::vector<std::string> names{};
  // The stiffness values for the joints.
  std::vector<double> stiffness{};
  // The damping values for the joints.
  std::vector<double> damping{};
};

// Configuration to write se(2) velocity commands.
struct Se2VelocityData {
  // The target frame for the se(2) velocity command.
  std::string target_frame{};
  // The ranges for the se(2) velocity command.
  std::optional<SE2VelocityRanges> ranges{};
};

// Configuration for the ONNX RL controller parsed from model metadata and keys.
struct OnnxControllerConfig {
  // Mapping from ONNX input keys to body data.
  std::unordered_map<std::string, BodyData> imu_keys_to_data{};
  // Mapping from ONNX input keys to body data.
  std::unordered_map<std::string, BodyData> body_keys_to_data{};
  // Mapping from ONNX output keys to joint target data.
  std::unordered_map<std::string, JointTargetData> joint_target_keys_to_data{};
  // Mapping from ONNX output keys to se(2) velocity data.
  std::unordered_map<std::string, Se2VelocityData> se2_velocity_keys_to_data{};

  // Mapping from sensor keys to heightscan pattern indices.
  std::unordered_map<std::string, size_t> sensor_key_to_pattern_index{};
  // Mapping from memorized output keys to input keys.
  std::unordered_map<std::string, std::string> memorized_outputs_to_key{};

  // Mapping from command names to their types.
  std::unordered_map<std::string, std::string> command_name_to_type{};
  // Mapping from sensor names to their types.
  std::unordered_map<std::string, std::string> sensor_name_to_type{};
  // Mapping from ONNX output names to their types.
  std::unordered_map<std::string, std::string> output_name_to_type{};

  // Set of known input keys. This is used to validate that the model does not contain unexpected inputs.
  std::unordered_set<std::string> known_input_keys{};
  // Set of known output keys. This is used to validate that the model does not contain unexpected outputs.
  std::unordered_set<std::string> known_output_keys{};

  // Names of the robot joints used to read joint states.
  std::vector<std::string> joint_names{};
  // Controller update rate in Hz.
  int update_rate{};

  // Configuration for heightscan sensor.
  std::optional<HeightScanConfig> heightscan_config{};
  // Configuration for range image sensor.
  std::optional<RangeImageConfig> range_image_config{};
  // Configuration for depth image sensor.
  std::optional<DepthImageConfig> depth_image_config{};
};

/**
 * @brief Parses the ONNX model metadata to populate the controller configuration.
 *
 * @param onnx_model The initialized ONNX runtime model.
 * @return The populated OnnxControllerConfig or std::nullopt on failure.
 */
std::optional<OnnxControllerConfig> parseOnnxControllerConfig(OnnxRuntime& onnx_model);

}  // namespace rai::cs::control::common::onnx
