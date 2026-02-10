// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <gmock/gmock.h>
#include "command_interface.hpp"

namespace exploy::control {

class MockCommandInterface : public CommandInterface {
 public:
  MOCK_METHOD(bool, initSe2Velocity,
              (const std::string& command_name, const SE2VelocityConfig& config), (override));
  MOCK_METHOD(std::optional<SE2Velocity>, se2Velocity, (const std::string& command_name),
              (override));
  MOCK_METHOD(bool, initSe3Pose, (const std::string& command_name), (override));
  MOCK_METHOD(std::optional<SE3Pose>, se3Pose, (const std::string& command_name),
              (const, override));
  MOCK_METHOD(bool, initBooleanSelector, (const std::string& command_name), (override));
  MOCK_METHOD(std::optional<bool>, booleanSelector, (const std::string& command_name),
              (const override));
  MOCK_METHOD(bool, initFloatValue, (const std::string& command_name), (override));
  MOCK_METHOD(std::optional<float>, floatValue, (const std::string& command_name),
              (const override));
};

}  // namespace exploy::control
