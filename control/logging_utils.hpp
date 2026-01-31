#pragma once

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <iostream>

#define GENERIC_LOG(LEVEL, ...)         \
  do {                                  \
    std::cout << "[" << #LEVEL << "] "; \
    fmt::printf(__VA_ARGS__);           \
    std::cout << std::endl;             \
  } while (0);

#define GENERIC_LOG_STREAM(LEVEL, ...)                              \
  do {                                                              \
    std::cout << "[" << #LEVEL << "] " << __VA_ARGS__ << std::endl; \
  } while (0);
