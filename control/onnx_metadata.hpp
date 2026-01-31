// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include <optional>
#include <string>
#include <vector>

#include "interfaces.hpp"

namespace rai::cs::control::common::onnx {

using json = nlohmann::json;

inline void from_json(const json& j, Range& cmd) {
  j.at(0).get_to(cmd.min);
  j.at(1).get_to(cmd.max);
}

inline void from_json(const json& j, SE2VelocityRanges& ranges) {
  j.at("lin_vel_x").get_to(ranges.lin_vel_x);
  j.at("lin_vel_y").get_to(ranges.lin_vel_y);
  j.at("ang_vel_z").get_to(ranges.ang_vel_z);
}

}  // namespace rai::cs::control::common::onnx

namespace rai::cs::control::common::onnx::metadata {

using json = nlohmann::json;

template <typename T>
tl::expected<T, std::string> safe_json_get(const std::string& str) {
  try {
    nlohmann::json j = nlohmann::json::parse(str);
    return j.get<T>();
  } catch (const std::exception& e) {
    return tl::unexpected{fmt::format("Failed to get JSON value: {}", e.what())};
  }
}

struct SE2VelocityCommandMetadata {
  std::optional<SE2VelocityRanges> ranges{};
};

inline void from_json(const json& j, SE2VelocityCommandMetadata& cmd) {
  if (j.contains("ranges") && j["ranges"].is_object()) {
    SE2VelocityRanges ranges;
    j.at("ranges").get_to(ranges);
    cmd.ranges = ranges;
  } else {
    cmd.ranges = std::nullopt;
  }
}

struct HeightScanMetadata {
  std::string pattern_type{};
  double resolution{};
  double size_x{};
  double size_y{};
  double offset_x{};
  double offset_y{};
};

inline void from_json(const json& j, HeightScanMetadata& hs) {
  j.at("pattern_type").get_to(hs.pattern_type);
  j.at("resolution").get_to(hs.resolution);
  j.at("size_x").get_to(hs.size_x);
  j.at("size_y").get_to(hs.size_y);
  hs.offset_x = j.value("offset_x", 0.0);
  hs.offset_y = j.value("offset_y", 0.0);
}

struct RangeImageMetadata {
  std::string pattern_type{};
  double v_res{};
  double h_res{};
  double v_fov_min_deg{};
  double v_fov_max_deg{};
  double unobserved_value{};
};

inline void from_json(const json& j, RangeImageMetadata& ri) {
  j.at("pattern_type").get_to(ri.pattern_type);
  j.at("v_res").get_to(ri.v_res);
  j.at("h_res").get_to(ri.h_res);
  j.at("v_fov_min_deg").get_to(ri.v_fov_min_deg);
  j.at("v_fov_max_deg").get_to(ri.v_fov_max_deg);
  j.at("unobserved_value").get_to(ri.unobserved_value);
}

struct DepthImageMetadata {
  std::string pattern_type{};
  int width{};
  int height{};
  double fx{};
  double fy{};
  double cx{};
  double cy{};
};

inline void from_json(const json& j, DepthImageMetadata& di) {
  j.at("pattern_type").get_to(di.pattern_type);
  j.at("width").get_to(di.width);
  j.at("height").get_to(di.height);
  j.at("fx").get_to(di.fx);
  j.at("fy").get_to(di.fy);
  j.at("cx").get_to(di.cx);
  j.at("cy").get_to(di.cy);
}

struct JointOutputMetadata {
  std::vector<std::string> names{};
  std::vector<double> stiffness{};
  std::vector<double> damping{};
};

inline void from_json(const json& j, JointOutputMetadata& jo) {
  j.at("names").get_to(jo.names);
  j.at("stiffness").get_to(jo.stiffness);
  j.at("damping").get_to(jo.damping);
}

struct Se2VelocityOutputMetadata {
  std::string target_frame{};
};

inline void from_json(const json& j, Se2VelocityOutputMetadata& vo) {
  j.at("target_frame").get_to(vo.target_frame);
}

struct JointMetadata {
  std::vector<std::string> names{};
};

inline void from_json(const json& j, JointMetadata& jm) {
  j.at("names").get_to(jm.names);
}

}  // namespace rai::cs::control::common::onnx::metadata
