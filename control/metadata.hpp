// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

#include "interfaces.hpp"
#include "logging_utils.hpp"

namespace exploy::control {

using json = nlohmann::json;

/**
 * @brief Parse Range from JSON array [min, max].
 *
 * @param j JSON object containing a 2-element array.
 * @param cmd Range object to populate.
 */
inline void from_json(const json& j, Range& cmd) {
  j.at(0).get_to(cmd.min);
  j.at(1).get_to(cmd.max);
}

/**
 * @brief Parse SE2VelocityRanges from JSON object.
 *
 * @param j JSON object with keys "lin_vel_x", "lin_vel_y", "ang_vel_z".
 * @param ranges SE2VelocityRanges object to populate.
 */
inline void from_json(const json& j, SE2VelocityRanges& ranges) {
  j.at("lin_vel_x").get_to(ranges.lin_vel_x);
  j.at("lin_vel_y").get_to(ranges.lin_vel_y);
  j.at("ang_vel_z").get_to(ranges.ang_vel_z);
}

}  // namespace exploy::control

/**
 * @namespace exploy::control::metadata
 * @brief Namespace for ONNX model metadata structures.
 *
 * Contains data structures for representing ONNX model metadata including sensor
 * configurations, command specifications, and output parameters parsed from the
 * model's custom metadata fields.
 */
namespace exploy::control::metadata {

using json = nlohmann::json;

/**
 * @brief Safely parse JSON string into a typed object.
 *
 * @tparam T Target type to deserialize.
 * @param str JSON string to parse.
 * @return std::optional<T> containing the parsed object on success, or std::nullopt on failure.
 */
template <typename T>
std::optional<T> safe_json_get(const std::string& str) {
  try {
    nlohmann::json j = nlohmann::json::parse(str);
    return j.get<T>();
  } catch (const std::exception& e) {
    LOG_STREAM(ERROR, "Failed to parse JSON: " << e.what());
    return std::nullopt;
  }
}

/**
 * @brief Metadata for SE(2) velocity commands.
 *
 * Specifies optional constraints on planar velocity commands including
 * linear and angular velocity ranges.
 */
struct SE2VelocityCommandMetadata {
  std::optional<SE2VelocityRanges> ranges{};  ///< Optional velocity range constraints.
};

/**
 * @brief Parse SE2VelocityCommandMetadata from JSON.
 *
 * @param j JSON object optionally containing "ranges" field.
 * @param cmd SE2VelocityCommandMetadata object to populate.
 */
inline void from_json(const json& j, SE2VelocityCommandMetadata& cmd) {
  if (j.contains("ranges") && j["ranges"].is_object()) {
    SE2VelocityRanges ranges;
    j.at("ranges").get_to(ranges);
    cmd.ranges = ranges;
  } else {
    cmd.ranges = std::nullopt;
  }
}

/**
 * @brief Metadata for height scan sensors.
 *
 * Specifies the grid pattern configuration for terrain height scanning including
 * resolution, size, and offset relative to the base frame.
 */
struct HeightScanMetadata {
  std::string pattern_type{};  ///< Grid pattern type (e.g., "grid", "radial").
  double resolution{};         ///< Grid resolution (spacing between points).
  double size_x{};             ///< Grid size in x direction.
  double size_y{};             ///< Grid size in y direction.
  double offset_x{};           ///< Grid offset in x direction from base.
  double offset_y{};           ///< Grid offset in y direction from base.
};

/**
 * @brief Parse HeightScanMetadata from JSON.
 *
 * @param j JSON object containing height scan configuration.
 * @param hs HeightScanMetadata object to populate.
 */
inline void from_json(const json& j, HeightScanMetadata& hs) {
  j.at("pattern_type").get_to(hs.pattern_type);
  j.at("resolution").get_to(hs.resolution);
  j.at("size_x").get_to(hs.size_x);
  j.at("size_y").get_to(hs.size_y);
  hs.offset_x = j.value("offset_x", 0.0);
  hs.offset_y = j.value("offset_y", 0.0);
}

/**
 * @brief Metadata for LiDAR range image sensors.
 *
 * Specifies the configuration for range images including resolution, field of view,
 * and sentinel value for unobserved points.
 */
struct RangeImageMetadata {
  std::string pattern_type{};  ///< Range image pattern type.
  int v_res{};                 ///< Vertical resolution (number of vertical scan lines).
  int h_res{};                 ///< Horizontal resolution (points per scan line).
  double v_fov_min_deg{};      ///< Minimum vertical field of view in degrees.
  double v_fov_max_deg{};      ///< Maximum vertical field of view in degrees.
  double unobserved_value{};   ///< Sentinel value for unobserved/invalid points.
};

/**
 * @brief Parse RangeImageMetadata from JSON.
 *
 * @param j JSON object containing range image configuration.
 * @param ri RangeImageMetadata object to populate.
 */
inline void from_json(const json& j, RangeImageMetadata& ri) {
  j.at("pattern_type").get_to(ri.pattern_type);
  j.at("v_res").get_to(ri.v_res);
  j.at("h_res").get_to(ri.h_res);
  j.at("v_fov_min_deg").get_to(ri.v_fov_min_deg);
  j.at("v_fov_max_deg").get_to(ri.v_fov_max_deg);
  j.at("unobserved_value").get_to(ri.unobserved_value);
}

/**
 * @brief Metadata for camera depth image sensors.
 *
 * Specifies camera configuration including image dimensions and intrinsic parameters.
 */
struct DepthImageMetadata {
  std::string pattern_type{};  ///< Depth image pattern type.
  int width{};                 ///< Image width in pixels.
  int height{};                ///< Image height in pixels.
  double fx{};                 ///< Focal length in x direction (pixels).
  double fy{};                 ///< Focal length in y direction (pixels).
  double cx{};                 ///< Principal point x-coordinate (pixels).
  double cy{};                 ///< Principal point y-coordinate (pixels).
};

/**
 * @brief Parse DepthImageMetadata from JSON.
 *
 * @param j JSON object containing depth image configuration.
 * @param di DepthImageMetadata object to populate.
 */
inline void from_json(const json& j, DepthImageMetadata& di) {
  j.at("pattern_type").get_to(di.pattern_type);
  j.at("width").get_to(di.width);
  j.at("height").get_to(di.height);
  j.at("fx").get_to(di.fx);
  j.at("fy").get_to(di.fy);
  j.at("cx").get_to(di.cx);
  j.at("cy").get_to(di.cy);
}

/**
 * @brief Metadata for joint output commands.
 *
 * Specifies joint names and PD controller gains (stiffness and damping) for
 * position-controlled joints.
 */
struct JointOutputMetadata {
  std::vector<std::string> names{};  ///< Joint names in order.
  std::vector<double> stiffness{};   ///< Proportional gains (stiffness) for each joint.
  std::vector<double> damping{};     ///< Derivative gains (damping) for each joint.
};

/**
 * @brief Parse JointOutputMetadata from JSON.
 *
 * @param j JSON object containing joint output configuration.
 * @param jo JointOutputMetadata object to populate.
 */
inline void from_json(const json& j, JointOutputMetadata& jo) {
  j.at("names").get_to(jo.names);
  j.at("stiffness").get_to(jo.stiffness);
  j.at("damping").get_to(jo.damping);
}

/**
 * @brief Metadata for SE(2) velocity output commands.
 *
 * Specifies the target frame for planar velocity commands.
 */
struct Se2VelocityOutputMetadata {
  std::string target_frame{};  ///< Name of the target frame for velocity commands.
};

/**
 * @brief Parse Se2VelocityOutputMetadata from JSON.
 *
 * @param j JSON object containing SE(2) velocity output configuration.
 * @param vo Se2VelocityOutputMetadata object to populate.
 */
inline void from_json(const json& j, Se2VelocityOutputMetadata& vo) {
  j.at("target_frame").get_to(vo.target_frame);
}

/**
 * @brief Metadata for joint configurations.
 *
 * Specifies the list of joint names for joint-related inputs.
 */
struct JointMetadata {
  std::vector<std::string> names{};  ///< Joint names.
};

/**
 * @brief Parse JointMetadata from JSON.
 *
 * @param j JSON object containing joint configuration.
 * @param jm JointMetadata object to populate.
 */
inline void from_json(const json& j, JointMetadata& jm) {
  j.at("joint_names").get_to(jm.names);
}

}  // namespace exploy::control::metadata
