// Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
 * @brief A flattened height scan with layers.
 *
 * The height scan was flattened from a grid pattern according to IsaacLab conventions.
 * height and colors come from a single pattern and need to have the same length.
 * As soon as the onnx wrapper supports multi-dimensional inputs, we should change this to a 2D
 * array.
 */
struct HeightScan {
  std::unordered_map<std::string, std::vector<double>> layers;
};

}  // namespace rai::cs::control::common::onnx
