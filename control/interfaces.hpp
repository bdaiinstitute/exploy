// Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optional>
#include <vector>
#include <string>
#include <string_view>

namespace rai::cs::control::common::onnx {

using Position = Eigen::Vector3d;
using Quaternion = Eigen::Quaterniond;

using SE2Velocity = Eigen::Vector3d;

using LinearVelocity = Eigen::Vector3d;
using AngularVelocity = Eigen::Vector3d;

struct SE3Pose {
  Position position;
  Quaternion orientation;
};

struct Range {
  double min{};
  double max{};
};

struct SE2VelocityRanges {
  Range lin_vel_x;
  Range lin_vel_y;
  Range ang_vel_z;
};

/**
 * @brief A flattened height scan with optional color information (r, g, b).
 *
 * The height scan was flattened from a grid pattern according to IsaacLab conventions.
 * height and colors come from a single pattern and need to have the same length.
 * As soon as the onnx wrapper supports multi-dimensional inputs, we should change this to a 2D array.
 */
struct HeightScan {
  std::vector<double> height;
  struct ColorScan {
    std::vector<double> r;
    std::vector<double> g;
    std::vector<double> b;
  };
  std::optional<ColorScan> color;
};

namespace References {
namespace SE2Velocity {
inline constexpr std::string_view lin_vel_x = "twist_linear_velocity_x";
inline constexpr std::string_view lin_vel_y = "twist_linear_velocity_y";
inline constexpr std::string_view ang_vel_z = "twist_angular_velocity_z";
}  // namespace SE2Velocity

namespace SE3Pose {
inline constexpr std::string_view x = "pose_position_x";
inline constexpr std::string_view y = "pose_position_y";
inline constexpr std::string_view z = "pose_position_z";
inline constexpr std::string_view qw = "pose_orientation_w";
inline constexpr std::string_view qx = "pose_orientation_x";
inline constexpr std::string_view qy = "pose_orientation_y";
inline constexpr std::string_view qz = "pose_orientation_z";
}  // namespace SE3Pose

namespace BooleanSelector {
inline constexpr std::string_view selector = "boolean_selector";
}
}  // namespace References

}  // namespace rai::cs::control::common::onnx
