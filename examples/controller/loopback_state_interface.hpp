// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

/**
 * @file loopback_state_interface.hpp
 * @brief RobotStateInterface that feeds commanded values back as measured state.
 *
 * This is intentionally minimal: every init*() succeeds, every getter returns a
 * sensible default (zeros / identity), and the set*() methods store the commanded
 * value so that it is returned the following cycle — modelling a perfect,
 * zero-delay actuator.
 */

#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <utility>

#include "state_interface.hpp"

namespace exploy::control::examples {

/**
 * @brief Per-joint loopback state (position, velocity, effort).
 */
struct JointState {
  double position{0.0};
  double velocity{0.0};
  double effort{0.0};
};

/**
 * @brief Concrete RobotStateInterface with perfect loopback semantics.
 *
 * Commanded joint targets written via setJoint*() are returned by
 * jointPosition() / jointVelocity() / jointEffort() on the next query,
 * modelling a zero-latency, perfectly tracking actuator.
 *
 * All body/IMU/base state values are fixed at identity / zero so that an
 * ONNX policy can run without a physics simulation.
 */
class LoopbackRobotStateInterface : public RobotStateInterface {
 public:
  bool initBasePosW() override { return true; }
  bool initBaseQuatW() override { return true; }

  std::optional<Position> basePosW() const override {
    return Position{0.0, 0.0, 0.8};
  }
  std::optional<Quaternion> baseQuatW() const override {
    return Quaternion::Identity();
  }

  bool initBaseLinVelB() override { return true; }
  bool initBaseAngVelB() override { return true; }

  std::optional<LinearVelocity> baseLinVelB() const override {
    return LinearVelocity::Zero();
  }
  std::optional<AngularVelocity> baseAngVelB() const override {
    return AngularVelocity::Zero();
  }

  bool initJointPosition(const std::string& joint_name) override {
    joint_states_.emplace(joint_name, JointState{});
    return true;
  }
  bool initJointVelocity(const std::string& joint_name) override {
    joint_states_.emplace(joint_name, JointState{});
    return true;
  }
  bool initJointEffort(const std::string& joint_name) override {
    joint_states_.emplace(joint_name, JointState{});
    return true;
  }
  bool initJointOutput(const std::string& /*joint_name*/) override {
    return true;
  }

  std::optional<double> jointPosition(const std::string& joint_name) const override {
    if (joint_states_.contains(joint_name)) {
      return joint_states_.at(joint_name).position;
    }
    return std::nullopt;
  }
  std::optional<double> jointVelocity(const std::string& joint_name) const override {
    if (joint_states_.contains(joint_name)) {
      return joint_states_.at(joint_name).velocity;
    }
    return std::nullopt;
  }
  std::optional<double> jointEffort(const std::string& joint_name) const override {
    if (joint_states_.contains(joint_name)) {
      return joint_states_.at(joint_name).effort;
    }
    return std::nullopt;
  }

  /// Loopback: store the commanded position so it is returned next cycle.
  bool setJointPosition(const std::string& joint_name, double position) override {
    joint_states_[joint_name].position = position;
    return true;
  }
  bool setJointVelocity(const std::string& joint_name, double velocity) override {
    joint_states_[joint_name].velocity = velocity;
    return true;
  }
  bool setJointEffort(const std::string& joint_name, double effort) override {
    joint_states_[joint_name].effort = effort;
    return true;
  }
  bool setJointPGain(const std::string& /*joint_name*/, double /*p_gain*/) override {
    return true;
  }
  bool setJointDGain(const std::string& /*joint_name*/, double /*d_gain*/) override {
    return true;
  }

  // Read-only access to the full joint state map (used by main.cpp for logging).
  const std::unordered_map<std::string, JointState>& jointStates() const {
    return joint_states_;
  }

  bool initSe2Velocity(const std::string& frame_name) override {
    se2_velocities_.emplace(frame_name, SE2Velocity::Zero());
    return true;
  }
  bool setSe2Velocity(const std::string& frame_name, const SE2Velocity& velocity) override {
    se2_velocities_[frame_name] = velocity;
    return true;
  }

  // Read-only access to all SE(2) velocity outputs (used by main.cpp for logging).
  const std::unordered_map<std::string, SE2Velocity>& se2VelocityOutputs() const {
    return se2_velocities_;
  }

  bool initImuOrientationW(const std::string& /*imu_name*/) override { return true; }

  std::optional<Quaternion> imuOrientationW(const std::string& /*imu_name*/) const override {
    return Quaternion::Identity();
  }

  bool initBodyPositionW(const std::string& /*body_name*/) override { return true; }
  bool initBodyOrientationW(const std::string& /*body_name*/) override { return true; }

  std::optional<Position> bodyPositionW(const std::string& /*body_name*/) const override {
    return Position{0.0, 0.0, 0.8};
  }
  std::optional<Quaternion> bodyOrientationW(const std::string& /*body_name*/) const override {
    return Quaternion::Identity();
  }

  bool initHeightScan(const std::string& sensor_name,
                      const HeightScanConfig& config) override {
    // Pre-allocate zero-filled layers sized by the grid formula.
    const auto& p = config.pattern;
    const std::size_t nx = static_cast<std::size_t>(std::round(p.size.x() / p.resolution)) + 1;
    const std::size_t ny = static_cast<std::size_t>(std::round(p.size.y() / p.resolution)) + 1;
    const std::size_t n = nx * ny;

    HeightScan scan;
    for (const auto& layer : config.layer_names) {
      scan.layers[layer] = std::vector<double>(n, 0.0);
    }
    height_scans_[sensor_name] = std::move(scan);
    return true;
  }

  std::optional<HeightScan*> heightScan(
      const std::string& sensor_name,
      const std::unordered_set<std::string>& /*layer_names*/,
      const Position& /*base_pos_w*/,
      const Quaternion& /*base_quat_w*/) override {
    if (!height_scans_.contains(sensor_name)) {
      return std::nullopt;
    }
    return &height_scans_.at(sensor_name);
  }

 private:
  std::unordered_map<std::string, JointState> joint_states_;
  std::unordered_map<std::string, SE2Velocity> se2_velocities_;
  std::unordered_map<std::string, HeightScan> height_scans_;
};

}  // namespace exploy::control::examples
