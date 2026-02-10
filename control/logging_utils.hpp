// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <iostream>

/**
 * @brief Printf-style logging macro with severity level.
 *
 * Outputs formatted log messages to stdout with a severity level prefix.
 * Uses fmt::printf for formatting.
 *
 * @param LEVEL Severity level identifier (e.g., ERROR, WARNING, INFO).
 * @param ... Printf-style format string and arguments.
 *
 * Example: GENERIC_LOG(ERROR, "Failed to initialize %s\n", component_name);
 */
#define GENERIC_LOG(LEVEL, ...)         \
  do {                                  \
    std::cout << "[" << #LEVEL << "] "; \
    fmt::printf(__VA_ARGS__);           \
    std::cout << std::endl;             \
  } while (0);

/**
 * @brief Stream-style logging macro with severity level.
 *
 * Outputs log messages to stdout with a severity level prefix using stream operators.
 * Supports C++ stream insertion operators (<<).
 *
 * @param LEVEL Severity level identifier (e.g., ERROR, WARNING, INFO).
 * @param ... Stream-compatible expressions to log.
 *
 * Example: GENERIC_LOG_STREAM(ERROR, "Failed to initialize " << component_name);
 */
#define GENERIC_LOG_STREAM(LEVEL, ...)                              \
  do {                                                              \
    std::cout << "[" << #LEVEL << "] " << __VA_ARGS__ << std::endl; \
  } while (0);
