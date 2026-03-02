// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include "logging_interface.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <iostream>
#include <sstream>
#include <string_view>

namespace exploy::control {

namespace log {

inline LoggingInterface*& globalLogger() {
  static LoggingInterface* logger = nullptr;
  return logger;
}

inline LoggingInterface::Level macroLevelToEnum(const char* level_str) {
  std::string_view lv(level_str);
  if (lv == "WARN" || lv == "WARNING") return LoggingInterface::Level::Warn;
  if (lv == "INFO") return LoggingInterface::Level::Info;
  // NOTE: Any unknown level string (including new macros like DEBUG) will fall
  // back to Error. If you introduce new log levels, update this function.
  return LoggingInterface::Level::Error;
}

}  // namespace log

/**
 * @brief Set the active logging backend.
 *
 * All subsequent LOG and LOG_STREAM calls will route through
 * this backend. Pass nullptr to fall back to stdout.
 *
 * @param logger Pointer to the logging backend, or nullptr for stdout.
 */
inline void setLogger(LoggingInterface* logger) {
  log::globalLogger() = logger;
}

/**
 * @brief Get the currently active logging backend.
 *
 * @return Pointer to the active LoggingInterface, or nullptr if using stdout.
 */
inline LoggingInterface* getLogger() {
  return log::globalLogger();
}

}  // namespace exploy::control

/**
 * @brief Printf-style logging macro with severity level.
 *
 * Routes through the active logging backend (set via setLogger()), or falls
 * back to stdout if none is set. Uses fmt::printf for formatting.
 *
 * @param LEVEL Severity level identifier (e.g., ERROR, WARNING, INFO).
 * @param ... Printf-style format string and arguments.
 *
 * Example: LOG(ERROR, "Failed to initialize %s\n", component_name);
 */
#define LOG(LEVEL, ...)                                                      \
  do {                                                                       \
    auto* _exploy_logger_ = ::exploy::control::log::globalLogger();          \
    if (_exploy_logger_) {                                                   \
      _exploy_logger_->log(::exploy::control::log::macroLevelToEnum(#LEVEL), \
                           fmt::sprintf(__VA_ARGS__));                       \
    } else {                                                                 \
      std::cout << "[" << #LEVEL << "] ";                                    \
      fmt::printf(__VA_ARGS__);                                              \
      std::cout << std::endl;                                                \
    }                                                                        \
  } while (0);

/**
 * @brief Stream-style logging macro with severity level.
 *
 * Routes through the active logging backend (set via setLogger()), or falls
 * back to stdout if none is set. Supports C++ stream insertion operators (<<).
 *
 * @param LEVEL Severity level identifier (e.g., ERROR, WARNING, INFO).
 * @param ... Stream-compatible expressions to log.
 *
 * Example: LOG_STREAM(ERROR, "Failed to initialize " << component_name);
 */
#define LOG_STREAM(LEVEL, ...)                                                                    \
  do {                                                                                            \
    auto* _exploy_logger_ = ::exploy::control::log::globalLogger();                               \
    if (_exploy_logger_) {                                                                        \
      std::ostringstream _exploy_oss_;                                                            \
      _exploy_oss_ << __VA_ARGS__;                                                                \
      _exploy_logger_->log(::exploy::control::log::macroLevelToEnum(#LEVEL), _exploy_oss_.str()); \
    } else {                                                                                      \
      std::cout << "[" << #LEVEL << "] " << __VA_ARGS__ << std::endl;                             \
    }                                                                                             \
  } while (0);
