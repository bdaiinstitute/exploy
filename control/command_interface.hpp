// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <array>
#include <optional>
#include <string>

#include "interfaces.hpp"
#include "logging_utils.hpp"

namespace rai::cs::control::common::onnx {

/**
 * @brief Configuration for SE2 velocity commands.
 */
struct SE2VelocityConfig {
  std::optional<SE2VelocityRanges> ranges{};
};

/**
 * @class CommandInterface
 *
 * @brief Interface which provides methods to send commands to the controllers.
 *
 */
class CommandInterface {
 public:
  virtual ~CommandInterface() = default;
  /**
   * @brief Initialize data source of commanded se2 velocity.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param command_name The name of the command.
   * @param config The configuration for the commanded se2 velocity.
   */
  virtual bool initSe2Velocity(const std::string& command_name,
                               const SE2VelocityConfig& /*config*/) {
    GENERIC_LOG_STREAM(ERROR, "initSe2Velocity() not implemented for command: " << command_name);
    return false;
  }
  /**
   * @brief Get commanded se2 velocity.
   *
   * @param command_name The name of the command.
   * @return The commanded se2 velocity (lin x, lin y, ang z).
   */
  virtual std::optional<SE2Velocity> se2Velocity(const std::string& command_name) {
    GENERIC_LOG_STREAM(ERROR, "se2Velocity() not implemented for command: " << command_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of commanded SE3 pose.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param command_name The name of the command.
   */
  virtual bool initSe3Pose(const std::string& command_name) {
    GENERIC_LOG_STREAM(ERROR, "initSe3Pose() not implemented for command: " << command_name);
    return false;
  }
  /**
   * @brief Get commanded SE3 some pose.
   *
   * @param command_name The name of the command.
   * @return The commanded SE3 pose.
   */
  virtual std::optional<SE3Pose> se3Pose(const std::string& command_name) const {
    GENERIC_LOG_STREAM(ERROR, "se3Pose() not implemented for command: " << command_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of a boolean selector.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param command_name The name of the command.
   */
  virtual bool initBooleanSelector(const std::string& command_name) {
    GENERIC_LOG_STREAM(ERROR,
                       "initBooleanSelector() not implemented for command: " << command_name);
    return false;
  }
  /**
   * @brief Get commanded boolean selector.
   *
   * @param command_name The name of the command.
   * @return The commanded bool.
   */
  virtual std::optional<bool> booleanSelector(const std::string& command_name) {
    GENERIC_LOG_STREAM(ERROR, "booleanSelector() not implemented for command: " << command_name);
    return std::nullopt;
  }
};

}  // namespace rai::cs::control::common::onnx
