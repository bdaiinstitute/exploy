// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file interfaces.hpp
 * @brief Common interface types for ONNX-based control systems.
 *
 * This file defines common data structures and type aliases used throughout
 * the ONNX controller implementation, including pose representations, velocity
 * types, and sensor data structures.
 */

namespace exploy::control {

/**
 * @brief 3D position vector.
 *
 * Represents a position in 3D Cartesian space using Eigen's Vector3d.
 */
using Position = Eigen::Vector3d;

/**
 * @brief Unit quaternion for 3D orientation.
 *
 * Represents a rotation in 3D space using Eigen's quaternion representation.
 * The quaternion should be normalized (unit quaternion).
 */
using Quaternion = Eigen::Quaterniond;

/**
 * @brief SE(2) velocity vector [v_x, v_y, omega_z].
 *
 * Represents planar velocity with linear components in x and y directions
 * and angular velocity around the z-axis.
 */
using SE2Velocity = Eigen::Vector3d;

/**
 * @brief 3D linear velocity vector.
 *
 * Represents linear velocity in 3D Cartesian space.
 */
using LinearVelocity = Eigen::Vector3d;

/**
 * @brief 3D angular velocity vector.
 *
 * Represents angular velocity in 3D space (rotation rates around x, y, z axes).
 */
using AngularVelocity = Eigen::Vector3d;

/**
 * @brief SE(3) pose representation.
 *
 * Represents a 6-DOF pose in 3D space combining position and orientation.
 */
struct SE3Pose {
  Position position;       ///< Position in 3D space
  Quaternion orientation;  ///< Orientation as unit quaternion
};

/**
 * @brief Value range constraint.
 *
 * Defines minimum and maximum bounds for a scalar value.
 */
struct Range {
  double min{};  ///< Minimum allowed value
  double max{};  ///< Maximum allowed value
};

/**
 * @brief SE(2) velocity constraints.
 *
 * Defines the allowable ranges for planar velocity commands including
 * linear velocities in x and y directions and angular velocity around z-axis.
 */
struct SE2VelocityRanges {
  Range lin_vel_x;  ///< Linear velocity range in x direction
  Range lin_vel_y;  ///< Linear velocity range in y direction
  Range ang_vel_z;  ///< Angular velocity range around z-axis
};

/**
 * @brief A flattened height scan with multiple layers.
 *
 * The height scan was flattened from a grid pattern according to IsaacLab conventions.
 * Each layer represents different terrain properties (e.g., height, color, normals).
 * All layers come from the same grid pattern and must have the same length.
 *
 * @note As soon as the ONNX wrapper supports multi-dimensional inputs, this should
 *       be changed to a 2D array representation for better performance and clarity.
 */
struct HeightScan {
  /**
   * @brief Map of layer names to their data vectors.
   *
   * Each entry maps a layer name (e.g., "height", "color") to a vector of
   * double values representing the flattened grid data for that layer.
   */
  std::unordered_map<std::string, std::vector<double>> layers;
};

}  // namespace exploy::control
