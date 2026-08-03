// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

/**
 * @file memory_visualizer.hpp
 * @brief Example of read-only tensor inspection via the Observer mechanism.
 *
 * This header demonstrates how to inspect internal policy state (e.g. the recurrent
 * hidden state maintained by MemoryOutput components) for debugging, without modifying
 * it. It shows two pieces that live entirely in user/example code:
 *
 *  - A `VisualizationObserver` (an `exploy::control::Observer`) that reads one tensor
 *    buffer read-only and prints it to stdout. Read-only access is enforced by the type
 *    system: `observe()` receives a const reference to the runtime and is const-qualified,
 *    so it cannot mutate the runtime buffers.
 *  - A `MemoryVisualizationMatcher` that creates no input/output components but matches
 *    tensors by regex and creates one `VisualizationObserver` per matched tensor.
 *
 * Because observers do not create input/output components, this matcher can observe
 * tensors (such as `memory.<key>.in`) that are already owned by a built-in matcher
 * without triggering the one-component-per-tensor rule.
 */

#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "exploy/components.hpp"
#include "exploy/matcher.hpp"
#include "exploy/onnx_runtime.hpp"

namespace exploy::control::examples {

/**
 * @brief Read-only observer that prints one tensor's buffer to stdout.
 */
class VisualizationObserver : public Observer {
 public:
  explicit VisualizationObserver(const std::string& tensor_name)
      : Observer("VisualizationObserver[" + tensor_name + "]"), tensor_name_(tensor_name) {}

  void observe(const OnnxRuntime& runtime) const override {
    // The tensor may be either a model input or output; try inputs first, then outputs.
    auto maybe_buffer = runtime.inputBuffer<float>(tensor_name_);
    if (!maybe_buffer.has_value()) maybe_buffer = runtime.outputBuffer<float>(tensor_name_);
    if (!maybe_buffer.has_value()) return;  // Not a float tensor or unknown name.

    const auto data = maybe_buffer.value();
    std::cout << "[viz] " << tensor_name_ << " = [";
    const std::size_t max_print = std::min<std::size_t>(data.size(), 8);
    for (std::size_t i = 0; i < max_print; ++i) {
      std::cout << std::fixed << std::setprecision(4) << data[i];
      if (i + 1 < max_print) std::cout << ", ";
    }
    if (data.size() > max_print) std::cout << ", ... (" << data.size() << " total)";
    std::cout << "]\n";
  }

 private:
  std::string tensor_name_;  ///< ONNX tensor name to read.
};

/**
 * @brief Observer matcher that visualizes tensors whose name matches a regex.
 *
 * Creates no input/output components (only observers), so it does not participate in the
 * one-component-per-tensor ownership check. Defaults to visualizing recurrent memory input
 * tensors (`memory.<key>.in`).
 */
class MemoryVisualizationMatcher : public Matcher {
 public:
  explicit MemoryVisualizationMatcher(std::regex pattern = std::regex{R"(memory\..*\.in)"})
      : Matcher("MemoryVisualizationMatcher"), pattern_(std::move(pattern)) {}

  bool matches(const Match& maybe_match) override {
    if (!std::regex_match(maybe_match.name, pattern_)) return false;
    observed_.push_back(maybe_match.name);
    return true;
  }

  std::vector<std::unique_ptr<Observer>> createObservers() const override {
    std::vector<std::unique_ptr<Observer>> observers;
    for (const auto& name : observed_) {
      observers.push_back(std::make_unique<VisualizationObserver>(name));
    }
    return observers;
  }

 private:
  void reset() override { observed_.clear(); }

  std::regex pattern_;                 ///< Tensor names to observe.
  std::vector<std::string> observed_;  ///< Observed tensor names recorded during observes().
};

}  // namespace exploy::control::examples
