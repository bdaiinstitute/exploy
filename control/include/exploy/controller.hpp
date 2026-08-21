// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#pragma once

#include "exploy/command_interface.hpp"
#include "exploy/context.hpp"
#include "exploy/data_collection_interface.hpp"
#include "exploy/onnx_runtime.hpp"
#include "exploy/state_interface.hpp"
#include "exploy/worker.hpp"

#include <memory>
#include <string>

namespace exploy::control {

/**
 * @brief Selects the ONNX inference execution strategy.
 *
 * Pass this enum to `OnnxRLController::init()` to choose between:
 * - `SYNC`  — inference blocks the calling thread
 * - `ASYNC` — inference runs on a background thread
 */
enum class WorkerMode {
  SYNC,   ///< Run inference synchronously on the calling thread.
  ASYNC,  ///< Run inference on a background thread (see AsyncWorker).
};

/** @brief Worker mode and optional asynchronous thread scheduling. */
struct WorkerOptions {
  WorkerMode mode = WorkerMode::SYNC;
  ThreadSchedulingOptions scheduling;
};

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
   * If called more than once (e.g. to reload a model), default matchers are registered only on
   * the first call where register_default_matchers is true; custom matchers added via context()
   * are preserved across calls.
   *
   * @param onnx_model_path Path to the ONNX model file.
   * @param register_default_matchers If true (default), all built-in matchers (including
   * StepCountMatcher) are registered the first time create() is called with this parameter set to
   * true. Passing false disables all built-in matchers for that call; only matchers added via
   * context() will be used, and a later call with true can still register the built-in matchers.
   * @param options Runtime options (e.g. execution provider) forwarded to the ONNX runtime.
   * @return True if parsing succeeds, false otherwise.
   */
  bool create(const std::string& onnx_model_path, bool register_default_matchers = true,
              const OnnxRuntimeOptions& options = {});
  /**
   * @brief Initialize the controller.
   *
   * @param enable_data_collection Whether to enable data collection.
   * @param options Worker mode and asynchronous thread scheduling options.
   * @return True if initialization succeeds, false otherwise.
   */
  bool init(bool enable_data_collection, const WorkerOptions& options = {});

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

  bool readInputs();
  bool writeOutputs();
  bool runObservers(uint64_t time_us);

  OnnxContext context_{};
  OnnxRuntime onnx_model_{};
  RobotStateInterface& state_;
  CommandInterface& command_;

  bool default_matchers_registered_{false};

  // Data collection.
  DataCollectionInterface& data_collection_;
  // Written by the work_fn (possibly on a background thread) before work_finished_
  // is set under the worker's mutex. Read on the main thread only after observing
  // work_finished_ under that same mutex — no atomic needed.
  double inference_duration_s_{};

  // Set on the main thread in update() before invoking the worker, so the write_fn
  // (which runs the observers) can forward the current timestamp to data collection.
  uint64_t update_time_us_{};

  std::unique_ptr<Worker> worker_{nullptr};
};

}  // namespace exploy::control
