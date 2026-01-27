#pragma once

#include <iostream>
#include <fmt/core.h>
#include <fmt/printf.h>
#include <fmt/ostream.h>

#define GENERIC_LOG(LEVEL, ...) \
    do { \
        std::cout << "[" << #LEVEL << "] "; \
        fmt::printf(__VA_ARGS__); \
        std::cout << std::endl; \
    } while (0);

#define GENERIC_LOG_STREAM(LEVEL, ...) \
    do { \
        std::cout << "[" << #LEVEL << "] " << __VA_ARGS__ << std::endl; \
    } while (0);

#define SCOPED_LOG GENERIC_LOG
#define SCOPED_LOG_STREAM GENERIC_LOG_STREAM

#define CS_TRACE_SCOPED_ZONE
