// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <gmock/gmock.h>
#include "logging_interface.hpp"

namespace exploy::control {

class MockLoggingInterface : public LoggingInterface {
 public:
  MOCK_METHOD(void, log, (Level level, std::string_view message), (override));
};

}  // namespace exploy::control
