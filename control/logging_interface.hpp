// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <iostream>
#include <string_view>

namespace exploy::control {

/**
 * @class LoggingInterface
 *
 * @brief Interface for custom logging backends.
 *
 * Implement this interface to redirect log messages from the controller to a
 * custom logging system (e.g., ROS logger, spdlog, etc.).
 */
class LoggingInterface {
 public:
  enum class Level { Error, Warn, Info };

  virtual ~LoggingInterface() = default;

  /**
   * @brief Log a message at the given severity level.
   *
   * @param level Severity level of the message.
   * @param message The log message.
   */
  virtual void log(Level level, std::string_view message) = 0;
};

/**
 * @class StdoutLogger
 *
 * @brief Default logging backend that writes to stdout.
 */
class StdoutLogger : public LoggingInterface {
 public:
  void log(Level level, std::string_view message) override {
    switch (level) {
      case Level::Error:
        std::cout << "[ERROR] " << message << std::endl;
        break;
      case Level::Warn:
        std::cout << "[WARN] " << message << std::endl;
        break;
      case Level::Info:
        std::cout << "[INFO] " << message << std::endl;
        break;
    }
  }
};

}  // namespace exploy::control
