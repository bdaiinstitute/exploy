// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#pragma once

#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx_components.hpp"
#include "onnx_matcher.hpp"
#include "onnx_runtime.hpp"

namespace rai::cs::control::common::onnx {

class OnnxContext {
 public:
  // Registration methods for matchers
  void registerMatcher(std::unique_ptr<Matcher> matcher);
  void registerGroupMatcher(std::unique_ptr<GroupMatcher> matcher);

  bool createContext(OnnxRuntime& onnx_model);

  const std::vector<std::unique_ptr<Input>>& getInputs() const { return inputs_; }
  const std::vector<std::unique_ptr<Output>>& getOutputs() const { return outputs_; }

  int updateRate() const { return update_rate_; }

 private:
  GroupMatcher* tryGroupMatchers(const Match& match);
  Matcher* tryMatcher(const Match& match, bool found_match = false);

  std::vector<std::unique_ptr<Input>> inputs_;
  std::vector<std::unique_ptr<Output>> outputs_;
  std::vector<std::unique_ptr<Matcher>> matchers_;
  std::vector<std::unique_ptr<GroupMatcher>> group_matchers_;
  int update_rate_{0};
};

}  // namespace rai::cs::control::common::onnx
