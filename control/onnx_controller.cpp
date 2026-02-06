// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#include "onnx_controller.hpp"
#include "logging_utils.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <span>
#include <vector>

namespace rai::cs::control::common::onnx {

OnnxRLController::OnnxRLController(
    RobotStateInterface& state, CommandInterface& command,
    operation::common::data_collection_public::DataCollectionInterface& data_collection)
    : state_(state), command_(command), data_collection_(data_collection) {
  // Register all matchers
  context_.registerMatcher(std::make_unique<JointPositionMatcher>());
  context_.registerMatcher(std::make_unique<JointVelocityMatcher>());
  context_.registerMatcher(std::make_unique<BasePositionMatcher>());
  context_.registerMatcher(std::make_unique<BaseOrientationMatcher>());
  context_.registerMatcher(std::make_unique<BaseLinearVelocityMatcher>());
  context_.registerMatcher(std::make_unique<BaseAngularVelocityMatcher>());
  context_.registerMatcher(std::make_unique<SE2VelocityMatcher>());
  context_.registerMatcher(std::make_unique<IMUAngularVelocityMatcher>());
  context_.registerMatcher(std::make_unique<IMUOrientationMatcher>());
  context_.registerMatcher(std::make_unique<RangeImageMatcher>());
  context_.registerMatcher(std::make_unique<DepthImageMatcher>());
  context_.registerMatcher(std::make_unique<BodyPositionMatcher>());
  context_.registerMatcher(std::make_unique<BodyOrientationMatcher>());
  context_.registerMatcher(std::make_unique<CommandSE3PoseMatcher>());
  context_.registerMatcher(std::make_unique<CommandSE2VelocityMatcher>());
  context_.registerMatcher(std::make_unique<CommandBooleanMatcher>());
  context_.registerMatcher(std::make_unique<CommandFloatMatcher>());
  context_.registerMatcher(std::make_unique<StepCountMatcher>());
  // Register all group matchers
  context_.registerGroupMatcher(std::make_unique<JointTargetMatcher>());
  context_.registerGroupMatcher(std::make_unique<HeightScanMatcher>());
  context_.registerGroupMatcher(std::make_unique<MemoryMatcher>());
}

bool OnnxRLController::create(const std::string& onnx_model_path) {
  if (!onnx_model_.initialize(onnx_model_path)) {
    GENERIC_LOG_STREAM(ERROR, "Error creating OnnxEvaluator from policy: " << onnx_model_path);
    return false;
  }
  return true;
}

bool OnnxRLController::init(bool enable_data_collection) {
  if (!onnx_model_.isInitialized()) {
    GENERIC_LOG_STREAM(ERROR, "ONNX model is not initialized.");
    return false;
  }

  if (!context_.createContext(onnx_model_)) {
    GENERIC_LOG_STREAM(ERROR, "Error creating context.");
    return false;
  }

  for (const auto& input : context_.getInputs()) {
    if (!input->init(state_, command_)) {
      GENERIC_LOG_STREAM(ERROR, "Error initializing input.");
      return false;
    }
  }
  for (const auto& output : context_.getOutputs()) {
    if (!output->init(state_, command_)) {
      GENERIC_LOG_STREAM(ERROR, "Error initializing output.");
      return false;
    }
  }

  if (enable_data_collection) {
    for (const auto& name : onnx_model_.inputNames()) {
      auto maybe_buffer = onnx_model_.inputBuffer<float>(name);
      if (!maybe_buffer.has_value()) continue;
      if (!data_collection_.registerDataSource("model/input/" + name, maybe_buffer.value())) {
        GENERIC_LOG_STREAM(ERROR, "Registering data source for model/input/" << name << " failed.");
        return false;
      }
    }
    for (const auto& name : onnx_model_.outputNames()) {
      auto maybe_buffer = onnx_model_.outputBuffer<float>(name);
      if (!maybe_buffer.has_value()) continue;
      if (!data_collection_.registerDataSource("model/output/" + name, maybe_buffer.value())) {
        GENERIC_LOG_STREAM(ERROR,
                           "Registering data source for model/output/" << name << " failed.");
        return false;
      }
    }
    if (!data_collection_.registerDataSource("model/inference/duration_s", inference_duration_s_)) {
      GENERIC_LOG(ERROR, "Registering data source for inference duration failed.");
      return false;
    }
  }

  reset();

  return true;
}

void OnnxRLController::reset() {
  onnx_model_.resetBuffers();
}

bool OnnxRLController::update(uint64_t time_us) {
  for (const auto& input : context_.getInputs()) {
    if (!input->read(onnx_model_, state_, command_)) {
      GENERIC_LOG_STREAM(ERROR, "Failed to read input");
      return false;
    }
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  if (!onnx_model_.evaluate()) {
    GENERIC_LOG_STREAM(ERROR, "Policy evaluation failed.");
    return false;
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  inference_duration_s_ = std::chrono::duration<double>(end_time - start_time).count();

  for (const auto& output : context_.getOutputs()) {
    if (!output->write(onnx_model_, state_, command_)) {
      GENERIC_LOG_STREAM(ERROR, "Failed to write output");
      return false;
    }
  }

  if (!data_collection_.collectData(time_us)) {
    GENERIC_LOG(WARN, "Data collection failed.");
  }

  return true;
}

}  // namespace rai::cs::control::common::onnx
