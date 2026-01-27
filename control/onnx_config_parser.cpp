// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "onnx_config_parser.hpp"
#include "metadata.hpp"
#include "logging_utils.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cmath>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace rai::cs::control::common::onnx {

namespace {

// Helper function to split a string by a delimiter
std::vector<std::string> splitString(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  for (auto&& part : str | std::views::split(delimiter)) {
    auto common = part | std::views::common;
    tokens.emplace_back(common.begin(), common.end());
  }
  return tokens;
}

bool parsePolicyMetadata(OnnxRuntime& onnx_model, OnnxControllerConfig& config) {
  const auto maybe_joint_names = onnx_model.getCustomMetadata(keys::kJointNames);
  if (maybe_joint_names.has_value()) {
    config.joint_names = splitString(maybe_joint_names.value(), ',');
  }

  const auto maybe_decimation = onnx_model.getCustomMetadata(keys::kDecimation);
  if (!maybe_decimation.has_value()) {
    GENERIC_LOG(ERROR, "Failed to get decimation metadata");
    return false;
  }
  int decimation = std::stoi(maybe_decimation.value());

  const auto maybe_policy_dt = onnx_model.getCustomMetadata(keys::kPolicyDt);
  if (!maybe_policy_dt.has_value()) {
    GENERIC_LOG(ERROR, "Failed to get policy_dt metadata");
    return false;
  }
  int policy_update_rate = static_cast<int>(std::round(1 / std::stod(maybe_policy_dt.value())));

  bool has_step_count = onnx_model.inputNames().contains(keys::kStepCount);
  config.update_rate = has_step_count ? decimation * policy_update_rate : policy_update_rate;

  return true;
}

bool parseSensorMetadata(OnnxRuntime& onnx_model, OnnxControllerConfig& config) {
  auto maybe_sensors = onnx_model.getCustomMetadata(keys::kSensors);
  if (!maybe_sensors.has_value()) return true;

  const auto maybe_sensors_json = metadata::safe_json_parse(maybe_sensors.value());
  if (!maybe_sensors_json) {
    GENERIC_LOG_STREAM(ERROR, "Failed to parse sensor metadata: " << maybe_sensors_json.error());
    return false;
  }
  const auto& sensor_json = maybe_sensors_json.value();

  HeightScanConfig heightscan_config{
      .use_colors = false,
  };
  bool has_heightscan = false;

  for (const auto& [name, value] : sensor_json.items()) {
    auto maybe_type = metadata::safe_json_get<std::string>(value, keys::kType);
    if (!maybe_type) {
      GENERIC_LOG_STREAM(ERROR, "Failed to get type for sensor " << name << ": " << maybe_type.error());
      return false;
    }
    const std::string type = maybe_type.value();
    if (type == keys::kRayCaster) {
      if (!onnx_model.inputNames().contains(name)) continue;
      auto maybe_meta = metadata::safe_json_get<metadata::RayCasterMetadata>(value);
      if (!maybe_meta) {
        GENERIC_LOG_STREAM(ERROR, "Failed to parse metadata for raycaster " << name << ": " << maybe_meta.error());
        return false;
      }
      metadata::RayCasterMetadata meta = maybe_meta.value();

      if (meta.pattern_type != keys::kGridPattern) {
        GENERIC_LOG_STREAM(ERROR, meta.pattern_type << " pattern type not supported for " << name);
        return false;
      }

      heightscan_config.patterns.emplace_back(HeightScanConfig::Pattern{
          .size = Eigen::Vector2d{meta.size_x, meta.size_y},
          .resolution = meta.resolution,
          .offset = Eigen::Vector2d{meta.offset_x, meta.offset_y},
      });

      config.sensor_key_to_pattern_index[name] = heightscan_config.patterns.size() - 1;
      config.sensor_name_to_type[name] = type;
      config.known_input_keys.insert(name);
      has_heightscan = true;
    } else if (type == keys::kTrailRayCaster) {
      auto maybe_meta = metadata::safe_json_get<metadata::RayCasterMetadata>(value);
      if (!maybe_meta) {
        GENERIC_LOG_STREAM(ERROR, "Failed to parse metadata for trail raycaster " << name << ": " << maybe_meta.error());
        return false;
      }
      metadata::RayCasterMetadata meta = maybe_meta.value();

      if (meta.pattern_type != keys::kGridPattern) {
        GENERIC_LOG_STREAM(ERROR, meta.pattern_type << " pattern type not supported for " << name);
        return false;
      }

      heightscan_config.patterns.emplace_back(HeightScanConfig::Pattern{
          .size = Eigen::Vector2d{meta.size_x, meta.size_y},
          .resolution = meta.resolution,
          .offset = Eigen::Vector2d{meta.offset_x, meta.offset_y},
      });

      heightscan_config.use_colors = true;

      config.sensor_key_to_pattern_index[name] = heightscan_config.patterns.size() - 1;
      config.sensor_name_to_type[name] = type;
      config.known_input_keys.insert(name + ".height");
      config.known_input_keys.insert(name + ".r");
      config.known_input_keys.insert(name + ".g");
      config.known_input_keys.insert(name + ".b");
      has_heightscan = true;
    } else if (type == keys::kLidarRangeImage) {
      auto maybe_meta = metadata::safe_json_get<metadata::RangeImageMetadata>(value);
      if (!maybe_meta) {
        GENERIC_LOG_STREAM(ERROR, "Failed to parse metadata for lidar range image " << name << ": " << maybe_meta.error());
        return false;
      }
      metadata::RangeImageMetadata meta = maybe_meta.value();

      if (meta.pattern_type != keys::kLidarPattern) {
        GENERIC_LOG_STREAM(ERROR, meta.pattern_type << " pattern type not supported for " << name);
        return false;
      }

      if (config.range_image_config.has_value()) {
        GENERIC_LOG(ERROR,
                    "The ONNX file contains multiple range image sensor configurations, which are not yet supported.");
        return false;
      }

      config.range_image_config = RangeImageConfig{
          .v_res = static_cast<int>(meta.v_res),
          .h_res = static_cast<int>(meta.h_res),
          .v_fov_min_deg = meta.v_fov_min_deg,
          .v_fov_max_deg = meta.v_fov_max_deg,
          .unobserved_value = meta.unobserved_value,
      };

      config.sensor_name_to_type[name] = type;
      config.known_input_keys.insert(name);
    } else if (type == keys::kDepthImage) {
      auto maybe_meta = metadata::safe_json_get<metadata::DepthImageMetadata>(value);
      if (!maybe_meta) {
        GENERIC_LOG_STREAM(ERROR, "Failed to parse metadata for depth image " << name << ": " << maybe_meta.error());
        return false;
      }
      metadata::DepthImageMetadata meta = maybe_meta.value();

      if (config.depth_image_config.has_value()) {
        GENERIC_LOG(ERROR,
                    "The ONNX file contains multiple depth image sensor configurations, which are not yet supported.");
        return false;
      }

      config.depth_image_config = DepthImageConfig{
          .width = meta.width,
          .height = meta.height,
          .fx = meta.fx,
          .fy = meta.fy,
          .cx = meta.cx,
          .cy = meta.cy,
      };

      config.sensor_name_to_type[name] = type;
      config.known_input_keys.insert(name);
    } else {
      GENERIC_LOG_STREAM(ERROR, "Unknown sensor type " << type << " found for " << name);
      return false;
    }
  }

  if (has_heightscan) {
    config.heightscan_config = heightscan_config;
  }

  return true;
}

bool parseCommandMetadata(OnnxRuntime& onnx_model, OnnxControllerConfig& config) {
  auto maybe_commands = onnx_model.getCustomMetadata(keys::kCommands);
  if (!maybe_commands.has_value()) return true;

  const auto maybe_commands_json = metadata::safe_json_parse(maybe_commands.value());
  if (!maybe_commands_json) {
    GENERIC_LOG_STREAM(ERROR, "Failed to parse command metadata: " << maybe_commands_json.error());
    return false;
  }
  const auto& cmd_json = maybe_commands_json.value();

  for (const auto& [name, cmd_json] : cmd_json.items()) {
    auto maybe_type = metadata::safe_json_get<std::string>(cmd_json, keys::kType);
    if (!maybe_type) {
      GENERIC_LOG_STREAM(ERROR, "Failed to get type for command " << name << ": " << maybe_type.error());
      return false;
    }
    const std::string& type = maybe_type.value();
    if (type == keys::kSE2Velocity) {
      auto maybe_meta = metadata::safe_json_get<metadata::SE2VelocityCommandMetadata>(cmd_json);
      if (!maybe_meta) {
        GENERIC_LOG_STREAM(ERROR, "Failed to parse metadata for SE2Velocity command " << name << ": " << maybe_meta.error());
        return false;
      }
      const metadata::SE2VelocityCommandMetadata meta = maybe_meta.value();
      auto cfg = SE2VelocityConfig{};
      if (meta.ranges.has_value()) {
        cfg.ranges = SE2VelocityRanges{
            .lin_vel_x = {.min = meta.ranges.value().lin_vel_x.min, .max = meta.ranges.value().lin_vel_x.max},
            .lin_vel_y = {.min = meta.ranges.value().lin_vel_y.min, .max = meta.ranges.value().lin_vel_y.max},
            .ang_vel_z = {.min = meta.ranges.value().ang_vel_z.min, .max = meta.ranges.value().ang_vel_z.max},
        };
        if (cfg.ranges->lin_vel_x.min > 0 || cfg.ranges->lin_vel_y.min > 0 || cfg.ranges->ang_vel_z.min > 0) {
          GENERIC_LOG_STREAM(ERROR, "Minimum range values must be non-positive for command " << name);
          return false;
        }
        if (cfg.ranges->lin_vel_x.max < 0 || cfg.ranges->lin_vel_y.max < 0 || cfg.ranges->ang_vel_z.max < 0) {
          GENERIC_LOG_STREAM(ERROR, "Maximum range values must be non-negative for command " << name);
          return false;
        }
      }
      config.se2_velocity_keys_to_data[name] = Se2VelocityData{
          .ranges = cfg.ranges,
      };
      config.command_name_to_type[name] = type;
    } else if (type == keys::kSE3Pose) {
      config.command_name_to_type[name] = type;
    } else if (type == keys::kBooleanSelector) {
      config.command_name_to_type[name] = type;
    } else {
      GENERIC_LOG_STREAM(ERROR, "Unknown command type " << type);
      return false;
    }
    config.known_input_keys.insert(name);
  }

  return true;
}

bool parseOutputMetadata(OnnxRuntime& onnx_model, OnnxControllerConfig& config) {
  auto maybe_outputs = onnx_model.getCustomMetadata(keys::kOutputs);
  if (!maybe_outputs.has_value()) return true;

  const auto maybe_outputs_json = metadata::safe_json_parse(maybe_outputs.value());
  if (!maybe_outputs_json) {
    GENERIC_LOG_STREAM(ERROR, "Failed to parse output metadata: " << maybe_outputs_json.error());
    return false;
  }
  const auto& outputs_json = maybe_outputs_json.value();

  for (const auto& [output_name, output_json] : outputs_json.items()) {
    auto maybe_type = metadata::safe_json_get<std::string>(output_json, keys::kType);
    if (!maybe_type) {
      GENERIC_LOG_STREAM(ERROR, "Failed to get type for output " << output_name << ": " << maybe_type.error());
      return false;
    }
    const std::string& output_type = maybe_type.value();
    config.output_name_to_type[output_name] = output_type;

    if (output_type == keys::kJointTargets) {
      auto maybe_meta = metadata::safe_json_get<metadata::JointOutputMetadata>(output_json);
      if (!maybe_meta) {
        GENERIC_LOG_STREAM(ERROR, "Failed to parse metadata for JointTargets output " << output_name << ": " << maybe_meta.error());
        return false;
      }
      const metadata::JointOutputMetadata meta = maybe_meta.value();

      if (meta.names.size() != meta.stiffness.size() || meta.names.size() != meta.damping.size()) {
        GENERIC_LOG(ERROR, "Joint output metadata must have the same number of joint names, stiffness, and damping.");
        return false;
      }

      auto pos_name = fmt::format("{}.{}", output_name, "pos");
      auto vel_name = fmt::format("{}.{}", output_name, "vel");
      auto eff_name = fmt::format("{}.{}", output_name, "effort");

      config.joint_target_keys_to_data[output_name] = JointTargetData{
          .pos_name = pos_name,
          .vel_name = vel_name,
          .eff_name = eff_name,
          .names = meta.names,
          .stiffness = meta.stiffness,
          .damping = meta.damping,
      };

      config.known_output_keys.insert(pos_name);
      config.known_output_keys.insert(vel_name);
      config.known_output_keys.insert(eff_name);
    } else if (output_type == keys::kSE2Velocity) {
      auto maybe_meta = metadata::safe_json_get<metadata::Se2VelocityOutputMetadata>(output_json);
      if (!maybe_meta) {
        GENERIC_LOG_STREAM(ERROR, "Failed to parse metadata for SE2Velocity output " << output_name << ": " << maybe_meta.error());
        return false;
      }
      const auto meta = maybe_meta.value();
      config.se2_velocity_keys_to_data[output_name] = Se2VelocityData{
          .target_frame = meta.target_frame,
      };
      config.known_output_keys.insert(output_name);
    } else {
      GENERIC_LOG_STREAM(ERROR, "Unknown output type " << output_type << " found for " << output_name);
      return false;
    }
  }

  return true;
}

bool parseImuNames(OnnxRuntime& onnx_model, OnnxControllerConfig& config) {
  for (const auto& key : onnx_model.inputNames()) {
    std::vector<std::string> split_key = splitString(key, '.');
    if (split_key.size() == 0 || split_key[0] != keys::kIMUData) continue;
    if (split_key.size() != 3) {
      GENERIC_LOG_STREAM(ERROR, "Found key starting with " << keys::kIMUData << ", expected `imu_data.<interface>.<imu_name>`, but got " << key);
      return false;
    }
    config.imu_keys_to_data[key] = BodyData{.name = split_key[2], .interface = split_key[1]};
    config.known_input_keys.insert(key);
  }
  return true;
}

bool parseBodyNames(OnnxRuntime& onnx_model, OnnxControllerConfig& config) {
  for (const auto& key : onnx_model.inputNames()) {
    std::vector<std::string> split_key = splitString(key, '.');
    if (split_key.size() == 0 || split_key[0] != keys::kBody) continue;
    if (split_key.size() != 3) {
      GENERIC_LOG_STREAM(ERROR, "Found key starting with " << keys::kBody << ", expected `body.<body_name>.<interface>`, but got " << key);
      return false;
    }
    config.body_keys_to_data[key] = BodyData{.name = split_key[1], .interface = split_key[2]};
    config.known_input_keys.insert(key);
  }
  return true;
}

bool parseMemory(OnnxRuntime& onnx_model, OnnxControllerConfig& config) {
  for (const auto& key : onnx_model.inputNames()) {
    std::vector<std::string> split_key = splitString(key, '.');

    if (split_key.size() == 0 || split_key[0] != keys::kMemory) continue;
    if (split_key.size() < 3) {
      constexpr auto msg = "Found key starting with %s, expected `memory.<output>.in|out`, but got `%s`";
      GENERIC_LOG(ERROR, msg, keys::kMemory, key.c_str());
      return false;
    }

    split_key.back() = "out";
    const auto output_key = fmt::format("{}", fmt::join(split_key, "."));

    if (!onnx_model.outputNames().contains(output_key)) {
      constexpr auto msg = "Found memory input for %s but key is not present in output.";
      GENERIC_LOG(ERROR, msg, output_key.c_str());
      return false;
    }

    config.memorized_outputs_to_key[key] = output_key;
    config.known_input_keys.insert(key);
    config.known_output_keys.insert(output_key);
  }
  return true;
}

}  // namespace

std::optional<OnnxControllerConfig> parseOnnxControllerConfig(OnnxRuntime& onnx_model) {
  OnnxControllerConfig config;

  // Add known input and output keys which are always handled.
  config.known_input_keys = {keys::kStepCount,        keys::kPosBaseInWorld,   keys::kQuatBaseInWorld,
                             keys::kLinVelBaseInBase, keys::kAngVelBaseInBase, keys::kJointPos,
                             keys::kJointVel};
  config.known_output_keys = {keys::kObservation, keys::kActions};

  if (!parsePolicyMetadata(onnx_model, config)) return std::nullopt;
  if (!parseSensorMetadata(onnx_model, config)) return std::nullopt;
  if (!parseCommandMetadata(onnx_model, config)) return std::nullopt;
  if (!parseOutputMetadata(onnx_model, config)) return std::nullopt;
  if (!parseImuNames(onnx_model, config)) return std::nullopt;
  if (!parseBodyNames(onnx_model, config)) return std::nullopt;
  if (!parseMemory(onnx_model, config)) return std::nullopt;

  // Input/Output Check
  for (const auto& key : onnx_model.inputNames()) {
    if (config.known_input_keys.contains(key)) continue;
    GENERIC_LOG_STREAM(ERROR,
                "Input model contains unknown key '" << key << "'. You are probably trying to run a policy which contains new "
                "features that are not supported by this version of the controller.");
    return std::nullopt;
  }

  for (const auto& key : onnx_model.outputNames()) {
    if (config.known_output_keys.contains(key) || key.starts_with("debug.")) continue;
    GENERIC_LOG_STREAM(ERROR,
                "Output model contains unknown key '" << key << "'. You are probably trying to run a policy which contains new "
                "features that are not supported by this version of the controller.");
    return std::nullopt;
  }

  return config;
}

}  // namespace rai::cs::control::common::onnx
