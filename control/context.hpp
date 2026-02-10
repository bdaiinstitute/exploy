// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "components.hpp"
#include "matcher.hpp"
#include "onnx_runtime.hpp"

namespace exploy::control {

/**
 * @brief Manages the context for ONNX model input/output components.
 *
 * OnnxContext is responsible for parsing ONNX model metadata, matching tensor names
 * to appropriate input/output components using registered matchers, and creating the
 * component instances needed to interface between the robot and the ONNX model.
 */
class OnnxContext {
 public:
  /**
   * @brief Register a matcher for single tensor patterns.
   *
   * Matchers are used to identify ONNX tensor names and create corresponding
   * input/output components.
   *
   * @param matcher Unique pointer to a Matcher instance.
   */
  void registerMatcher(std::unique_ptr<Matcher> matcher);

  /**
   * @brief Register a group matcher for multi-tensor patterns.
   *
   * Group matchers handle patterns where multiple related tensors need to be
   * processed together (e.g., joint.pos and joint.vel).
   *
   * @param matcher Unique pointer to a GroupMatcher instance.
   */
  void registerGroupMatcher(std::unique_ptr<GroupMatcher> matcher);

  /**
   * @brief Create the context by parsing ONNX model and generating components.
   *
   * Analyzes the ONNX model's input/output tensors and metadata, matches them
   * against registered matchers, and creates the appropriate input/output components.
   *
   * @param onnx_model Reference to an initialized OnnxRuntime instance.
   * @param strict If true, fails if any tensor cannot be matched; if false, continues with partial
   * matches.
   * @return true if context creation succeeded, false otherwise.
   */
  bool createContext(OnnxRuntime& onnx_model, bool strict = true);

  /**
   * @brief Get all created input components.
   *
   * @return Const reference to vector of input component unique pointers.
   */
  const std::vector<std::unique_ptr<Input>>& getInputs() const { return inputs_; }

  /**
   * @brief Get all created output components.
   *
   * @return Const reference to vector of output component unique pointers.
   */
  const std::vector<std::unique_ptr<Output>>& getOutputs() const { return outputs_; }

  /**
   * @brief Get the control loop update rate from ONNX model metadata.
   *
   * @return Update rate in Hz, or 0 if not specified in model metadata.
   */
  int updateRate() const { return update_rate_; }

 private:
  std::vector<std::unique_ptr<Input>> inputs_;    ///< Input components for reading robot data.
  std::vector<std::unique_ptr<Output>> outputs_;  ///< Output components for writing robot commands.
  std::vector<std::unique_ptr<Matcher>> matchers_;  ///< Registered single-tensor matchers.
  std::vector<std::unique_ptr<GroupMatcher>>
      group_matchers_;  ///< Registered multi-tensor matchers.
  int update_rate_{0};  ///< Control loop update rate in Hz.
};

}  // namespace exploy::control
