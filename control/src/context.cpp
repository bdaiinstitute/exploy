// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "exploy/context.hpp"
#include "exploy/logging_utils.hpp"
#include "exploy/metadata.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace exploy::control {

namespace {

std::optional<int> parseUpdateRate(OnnxRuntime& onnx_model) {
  const auto maybe_update_rate = onnx_model.getCustomMetadata("update_rate");
  if (!maybe_update_rate.has_value()) {
    LOG(ERROR, "Failed to get update_rate metadata");
    return std::nullopt;
  }
  return static_cast<int>(std::stod(maybe_update_rate.value()));
}

// Parse the `base_names` metadata into a map of {robot_name: base_name} for use in matchers.
std::unordered_map<std::string, std::string> parseBaseNames(const OnnxRuntime& onnx_model) {
  std::unordered_map<std::string, std::string> base_names;
  const auto maybe_base_names = onnx_model.getCustomMetadata("base_names");
  if (!maybe_base_names.has_value()) return base_names;
  try {
    auto json_base_names = json::parse(maybe_base_names.value());
    for (auto it = json_base_names.begin(); it != json_base_names.end(); ++it) {
      base_names[it.key()] = it.value().get<std::string>();
    }
  } catch (const json::exception& e) {
    LOG_STREAM(ERROR, "Failed to parse base_names metadata: " << e.what());
  }
  return base_names;
}

}  // namespace

// Registration methods
void OnnxContext::registerMatcher(std::unique_ptr<Matcher> matcher) {
  matchers_.push_back(std::move(matcher));
}

void OnnxContext::registerGroupMatcher(std::unique_ptr<GroupMatcher> matcher) {
  group_matchers_.push_back(std::move(matcher));
}

bool OnnxContext::createContext(OnnxRuntime& onnx_model, bool strict) {
  // Reset the components and matchers.
  inputs_.clear();
  outputs_.clear();
  observers_.clear();
  for (auto& m : matchers_) m->resetMatcher();
  for (auto& m : group_matchers_) m->resetMatcher();

  // Check if ONNX model is properly loaded before accessing its properties
  if (!onnx_model.isInitialized()) {
    LOG_STREAM(ERROR, "ONNX model not properly loaded, skipping context creation");
    return false;
  }

  // Matchers should now be registered before calling createContext
  if (matchers_.empty() && group_matchers_.empty()) {
    LOG_STREAM(ERROR, "No matchers registered. Please register matchers before creating context.");
    return false;
  }

  if (!metadata::checkExployVersion(onnx_model.getCustomMetadata("exploy_version"))) return false;

  std::optional<int> maybe_update_rate = parseUpdateRate(onnx_model);
  if (!maybe_update_rate.has_value()) return false;
  update_rate_ = maybe_update_rate.value();

  base_names_ = parseBaseNames(onnx_model);

  runAllMatchers(onnx_model);
  collectComponents();
  return validateTensorOwnership(onnx_model, strict);
}

void OnnxContext::runAllMatchers(OnnxRuntime& onnx_model) {
  // Run every matcher against every input and output tensor. matches() records the matched tensors
  // inside each matcher so that createInputs()/createOutputs()/createObservers() can build the
  // corresponding components later. Multiple matchers may match the same tensor (for example a
  // functional matcher and a read-only observer); tensor ownership is validated at the component
  // level, so no exclusivity is enforced here.
  const auto run_matchers = [&](const std::string& tensor_name) {
    Match maybe_match{
        .name = tensor_name,
        .metadata = onnx_model.getCustomMetadata(tensor_name),
        .base_names = base_names_,
    };
    for (auto& group_matcher : group_matchers_) group_matcher->matches(maybe_match);
    for (auto& matcher : matchers_) matcher->matches(maybe_match);
  };

  for (const auto& input_name : onnx_model.inputNames()) run_matchers(input_name);
  for (const auto& output_name : onnx_model.outputNames()) {
    if (output_name == "actions" || output_name == "obs") continue;
    run_matchers(output_name);
  }

  for (auto& group_matcher : group_matchers_) {
    group_matcher->populateGroupMetadata([&onnx_model](const std::string& name) {
      return onnx_model.getCustomMetadata(name);
    });
  }
}

void OnnxContext::collectComponents() {
  auto collect_components = [](auto& components, auto& matchers, auto creator_fn) {
    for (auto& matcher : matchers) {
      auto items = (matcher.get()->*creator_fn)();
      components.insert(components.end(), std::make_move_iterator(items.begin()),
                        std::make_move_iterator(items.end()));
    }
  };

  collect_components(inputs_, matchers_, &Matcher::createInputs);
  collect_components(inputs_, group_matchers_, &GroupMatcher::createInputs);
  collect_components(outputs_, matchers_, &Matcher::createOutputs);
  collect_components(outputs_, group_matchers_, &GroupMatcher::createOutputs);
  collect_components(observers_, matchers_, &Matcher::createObservers);
  collect_components(observers_, group_matchers_, &GroupMatcher::createObservers);
}

bool OnnxContext::validateTensorOwnership(OnnxRuntime& onnx_model, bool strict) const {
  // Every input and output tensor must be served by exactly one input/output component. Observers
  // are read-only and do not participate in this check. Build a map of tensor name -> owning
  // component names from the components that were created above.
  std::unordered_map<std::string, std::vector<std::string>> tensor_owners;
  for (const auto& input : inputs_) {
    for (const auto& tensor_name : input->tensorNames()) {
      tensor_owners[tensor_name].push_back(input->getName());
    }
  }
  for (const auto& output : outputs_) {
    for (const auto& tensor_name : output->tensorNames()) {
      tensor_owners[tensor_name].push_back(output->getName());
    }
  }

  const auto check_tensor_owner = [&](const std::string& tensor_name, const char* kind) -> bool {
    const auto it = tensor_owners.find(tensor_name);
    if (it == tensor_owners.end() || it->second.empty()) {
      LOG_STREAM(WARNING, fmt::format("No component found for {} '{}'", kind, tensor_name));
      return !strict;
    }
    if (it->second.size() > 1) {
      LOG_STREAM(ERROR,
                 fmt::format("Multiple components ({}) found for {} '{}': [{}]", it->second.size(),
                             kind, tensor_name, fmt::join(it->second, ", ")));
      return false;
    }
    return true;
  };

  for (const auto& input_name : onnx_model.inputNames()) {
    if (!check_tensor_owner(input_name, "input")) return false;
  }
  for (const auto& output_name : onnx_model.outputNames()) {
    if (output_name == "actions" || output_name == "obs") continue;
    if (!check_tensor_owner(output_name, "output")) return false;
  }

  return true;
}

}  // namespace exploy::control
