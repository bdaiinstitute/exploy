// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#pragma once

#include "data_collection_interface.hpp"
#include "interfaces.hpp"
#include "command_interface.hpp"
#include "onnx_config_parser.hpp"
#include "state_interface.hpp"
#include "onnx_runtime.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rai::cs::control::common::onnx {

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
                            operation::common::data_collection::DataCollectionInterface& data_collection);
  /**
   * @brief Load the ONNX model and parse configuration.
   *
   * This function only parses the configuration needed to interface with the ONNX model. Call init() to fully
   * initialize the controller.
   *
   * @param onnx_model_path Path to the ONNX model file.
   * @return True if parsing succeeds, false otherwise.
   */
  bool load(const std::string& onnx_model_path);
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
   * @brief Update the controller. Read from state, evaluate ONNX model and write to state. Should be run at
   * updateRate()
   *
   * @param time_us Timestamp in microseconds.
   * @return True if update succeeds, false otherwise.
   */
  bool update(uint64_t time_us);

  /**
   * @brief Get the update rate of the ONNX policy.
   *
   * @return The update rate of the ONNX policy in Hertz.
   */
  int updateRate() const { return config_.update_rate; }
  /**
   * @brief Get the set of joints managed by the ONNX policy.
   *
   * The set of joints handled by the ONNX policy includes both joints that are actively controlled (e.g., ones for
   * which we generate torques from a PD controller) and passive joints (e.g., the front wheel of a driving robot). The
   * list includes all the joints that were visible to the controller when the ONNX file was generated.
   * If the ONNX policy does not generate any joint commands, the method returns std::nullopt.
   *
   * Note:
   * The list does not necessarily include all the joints that are available or controllable. For example, the ONNX file
   * may have been generated for a system that included only the upper body of a humanoid robot. In this case, the list
   * will not include the lower body joints.
   *
   * @return A set of joint names managed by the ONNX policy. If no joints are managed, returns std::nullopt.
   */
  std::optional<std::unordered_set<std::string>> jointNames() const {
    // The policy manages joints only if there are joint target outputs configured.
    if (config_.joint_target_keys_to_data.empty()) return std::nullopt;
    return std::unordered_set<std::string>(config_.joint_names.begin(), config_.joint_names.end());
  }

 private:
  bool initCommands();
  bool initSensors();

  bool readCommands();
  bool readIMU();
  bool readBody();
  bool readJointState();
  bool readBasePosInWorld();
  bool readBaseQuatInWorld();
  bool readBaseLinVelInBase();
  bool readBaseAngVelInBase();
  bool readSensors();

  bool writeActions();
  bool writeMemory();
  bool writeOutputs();
  void increaseStepCount();

  OnnxControllerConfig config_{};
  OnnxRuntime onnx_model_{};
  RobotStateInterface& state_;
  CommandInterface& command_;

  // Data collection.
  operation::common::data_collection::DataCollectionInterface& data_collection_;
  double inference_duration_s_{};
};

}  // namespace rai::cs::control::common::onnx
