// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#pragma once

#include "command_interface.hpp"
#include "context.hpp"
#include "data_collection_interface.hpp"
#include "interfaces.hpp"
#include "onnx_runtime.hpp"
#include "state_interface.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace exploy::control {

/**
 * @class OnnxRLController
 *
 * @brief Controller wrapping around an ONNX policy.
 *
 * This class can be used to control a robot with an ONNX policy.
 *
 */
class OnnxRLController {
 public:
  /**
   * @brief Constructor of the OnnxRLController.
   *
   * @param state A RobotStateInterface to communicate with the robot.
   * @param command A CommandInterface to send commands to the controller.
   * @param data_collection A DataCollection interface for data collection.
   */
  explicit OnnxRLController(RobotStateInterface& state, CommandInterface& command,
                            DataCollectionInterface& data_collection);
  /**
   * @brief Create the ONNX model and context.
   *
   * This function only parses the configuration needed to interface with the ONNX model. Call
   * init() to fully initialize the controller.
   *
   * @param onnx_model_path Path to the ONNX model file.
   * @return True if parsing succeeds, false otherwise.
   */
  bool create(const std::string& onnx_model_path);
  /**
   * @brief Initialize the controller.
   *
   * @param enable_data_collection Whether to enable data collection.
   * @return True if initialization succeeds, false otherwise.
   */
  bool init(bool enable_data_collection);
  /**
   * @brief Reset the controller.
   */
  void reset();
  /**
   * @brief Update the controller. Read from state, evaluate ONNX model and write to state. Should
   * be run at updateRate()
   *
   * @param time_us Timestamp in microseconds.
   * @return True if update succeeds, false otherwise.
   */
  bool update(uint64_t time_us);

  OnnxContext& context() { return context_; }

 private:
  bool initCommands();
  bool initSensors();

  OnnxContext context_{};
  OnnxRuntime onnx_model_{};
  RobotStateInterface& state_;
  CommandInterface& command_;

  // Data collection.
  DataCollectionInterface& data_collection_;
  double inference_duration_s_{};
};

}  // namespace exploy::control
