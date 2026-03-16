// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <array>
#include <optional>
#include <string>

#include "interfaces.hpp"
#include "logging_utils.hpp"

namespace exploy::control {

/**
 * @brief Configuration for SE2 velocity commands.
 */
struct SE2VelocityConfig {
  std::optional<SE2VelocityRanges> ranges{};
};

/**
 * @brief Configuration for SE3 pose commands.
 */
struct SE3PoseConfig {
  // Currently no configuration options, but this struct is defined for consistency and future
  // extensibility.
};

/**
 * @brief Configuration for boolean selector commands.
 */
struct BooleanSelectorConfig {
  // Currently no configuration options, but this struct is defined for consistency and future
  // extensibility.
};

/**
 * @brief Configuration for float scalar commands.
 */
struct FloatScalarConfig {
  std::optional<Range> range{};
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
   * @return True if initialization succeeded, false otherwise.
   */
  virtual bool initSe2Velocity(const std::string& command_name,
                               const SE2VelocityConfig& /*config*/) {
    LOG_STREAM(ERROR, "initSe2Velocity() not implemented for command: " << command_name);
    return false;
  }
  /**
   * @brief Get commanded se2 velocity.
   *
   * @param command_name The name of the command.
   * @return The commanded se2 velocity (lin x, lin y, ang z).
   */
  virtual std::optional<SE2Velocity> se2Velocity(const std::string& command_name) {
    LOG_STREAM(ERROR, "se2Velocity() not implemented for command: " << command_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of commanded SE3 pose.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param command_name The name of the command.
   * @param config The configuration for the commanded SE3 pose.
   * @return True if initialization succeeded, false otherwise.
   */
  virtual bool initSe3Pose(const std::string& command_name, const SE3PoseConfig& /*config*/) {
    LOG_STREAM(ERROR, "initSe3Pose() not implemented for command: " << command_name);
    return false;
  }
  /**
   * @brief Get commanded SE3 some pose.
   *
   * @param command_name The name of the command.
   * @return The commanded SE3 pose.
   */
  virtual std::optional<SE3Pose> se3Pose(const std::string& command_name) const {
    LOG_STREAM(ERROR, "se3Pose() not implemented for command: " << command_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of a boolean.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param command_name The name of the command.
   * @param config The configuration for the commanded boolean selector.
   * @return True if initialization succeeded, false otherwise.
   */
  virtual bool initBooleanSelector(const std::string& command_name,
                                   const BooleanSelectorConfig& /*config*/) {
    LOG_STREAM(ERROR, "initBooleanSelector() not implemented for command: " << command_name);
    return false;
  }
  /**
   * @brief Get commanded boolean selector.
   *
   * @param command_name The name of the command.
   * @return The commanded bool.
   */
  virtual std::optional<bool> booleanSelector(const std::string& command_name) const {
    LOG_STREAM(ERROR, "booleanSelector() not implemented for command: " << command_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of a float.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param command_name The name of the command.
   * @param config The configuration for the commanded float scalar.
   * @return True if initialization succeeded, false otherwise.
   */
  virtual bool initFloatValue(const std::string& command_name,
                              const FloatScalarConfig& /*config*/) {
    LOG_STREAM(ERROR, "initFloatValue() not implemented for command: " << command_name);
    return false;
  }
  /**
   * @brief Get commanded float value.
   *
   * @param command_name The name of the command.
   * @return The commanded float.
   */
  virtual std::optional<float> floatValue(const std::string& command_name) const {
    LOG_STREAM(ERROR, "floatValue() not implemented for command: " << command_name);
    return std::nullopt;
  }
};

}  // namespace exploy::control
