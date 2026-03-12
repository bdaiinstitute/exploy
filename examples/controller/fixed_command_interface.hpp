// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <optional>
#include <string>

#include "command_interface.hpp"
#include "interfaces.hpp"

namespace exploy::control::examples {

/**
 * @brief Configuration for the fixed command interface.
 */
struct FixedCommandConfig {
  // SE(2) velocity command [vx (m/s), vy (m/s), omega_z (rad/s)].
  SE2Velocity se2_velocity{0.5, 0.0, 0.0};
};

/**
 * @class FixedCommandInterface
 *
 * @brief A CommandInterface that returns a static SE(2) velocity command.
 *
 * Only intended for the example where no external operator input is
 * available. Returns a configurable SE(2) velocity command (linear and angular).
 */
class FixedCommandInterface : public CommandInterface {
 public:
  explicit FixedCommandInterface(FixedCommandConfig config = {}) : config_(std::move(config)) {}


  bool initSe2Velocity(const std::string& /*command_name*/,
                       const SE2VelocityConfig& /*config*/) override {
    return true;
  }

  std::optional<SE2Velocity> se2Velocity(const std::string& /*command_name*/) override {
    return config_.se2_velocity;
  }

  void setSe2Velocity(const SE2Velocity& velocity) { config_.se2_velocity = velocity; }

  const SE2Velocity& se2Velocity() const { return config_.se2_velocity; }

 private:
  FixedCommandConfig config_{};
};

}  // namespace exploy::control::examples
