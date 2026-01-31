// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "onnx_context.hpp"
#include "logging_utils.hpp"
#include "onnx_metadata.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cmath>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace rai::cs::control::common::onnx {

namespace {

std::optional<int> parseUpdateRate(OnnxRuntime& onnx_model) {
  const auto maybe_update_rate = onnx_model.getCustomMetadata("update_rate");
  if (!maybe_update_rate.has_value()) {
    GENERIC_LOG(ERROR, "Failed to get update_rate metadata");
    return std::nullopt;
  }
  return static_cast<int>(std::stod(maybe_update_rate.value()));
}

}  // namespace

// Registration methods
void OnnxContext::registerMatcher(std::unique_ptr<Matcher> matcher) {
  matchers_.push_back(std::move(matcher));
}

void OnnxContext::registerGroupMatcher(std::unique_ptr<GroupMatcher> matcher) {
  group_matchers_.push_back(std::move(matcher));
}

GroupMatcher* OnnxContext::tryGroupMatchers(const Match& match) {
  GroupMatcher* matched = nullptr;
  for (const auto& matcher : group_matchers_) {
    auto maybe_group_name = matcher->matches(match.name);
    if (maybe_group_name.has_value()) {
      if (matched) {
        GENERIC_LOG_STREAM(ERROR, fmt::format("{} matches multiple matcher patterns.", match.name));
        return nullptr;
      }
      matched = matcher.get();
    }
  }
  return matched;
}

Matcher* OnnxContext::tryMatcher(const Match& match, bool found_match) {
  Matcher* matched = nullptr;
  for (const auto& matcher : matchers_) {
    if (matcher->matches(match.name)) {
      if (matched || found_match) {
        GENERIC_LOG_STREAM(ERROR, fmt::format("{} matches multiple matcher patterns.", match.name));
        return nullptr;
      }
      matched = matcher.get();
    }
  }
  return matched;
}

bool OnnxContext::createContext(OnnxRuntime& onnx_model) {
  // Check if ONNX model is properly loaded before accessing its properties
  if (!onnx_model.isInitialized()) {
    GENERIC_LOG_STREAM(ERROR, "ONNX model not properly loaded, skipping context creation");
    return false;
  }

  // Matchers should now be registered before calling createContext
  if (matchers_.empty() && group_matchers_.empty()) {
    GENERIC_LOG_STREAM(ERROR,
                       "No matchers registered. Please register matchers before creating context.");
    return false;
  }

  std::optional<int> maybe_update_rate = parseUpdateRate(onnx_model);
  if (!maybe_update_rate.has_value()) return false;
  update_rate_ = maybe_update_rate.value();

  std::unordered_map<std::string, GroupMatch> group_name_to_matches;
  std::unordered_map<std::string, GroupMatcher*> group_name_to_matcher;

  for (const auto& input_name : onnx_model.inputNames()) {
    Match maybe_match{
        .name = input_name,
        .metadata = onnx_model.getCustomMetadata(input_name),
    };
    GroupMatcher* group_matcher = tryGroupMatchers(maybe_match);
    if (group_matcher) {
      auto group_name = group_matcher->matches(maybe_match.name).value();
      group_name_to_matches[group_name].input_matches.push_back(maybe_match);
      group_name_to_matcher[group_name] = group_matcher;
    }
    Matcher* matcher = tryMatcher(maybe_match, group_matcher != nullptr);
    if (matcher) {
      auto maybe_input = matcher->createInput(maybe_match);
      if (maybe_input) inputs_.push_back(std::move(maybe_input));
    }
    if (!matcher && !group_matcher) {
      GENERIC_LOG_STREAM(ERROR, fmt::format("No matcher found for input {}", input_name));
      return false;
    }
  }

  for (const auto& output_name : onnx_model.outputNames()) {
    if (output_name == "actions" || output_name == "obs") continue;
    Match maybe_match{
        .name = output_name,
        .metadata = onnx_model.getCustomMetadata(output_name),
    };
    auto group_matcher = tryGroupMatchers(maybe_match);
    if (group_matcher) {
      auto group_name = group_matcher->matches(maybe_match.name).value();
      group_name_to_matches[group_name].input_matches.push_back(maybe_match);
      group_name_to_matcher[group_name] = group_matcher;
    }
    Matcher* matcher = tryMatcher(maybe_match, group_matcher != nullptr);
    if (matcher) {
      auto maybe_output = matcher->createOutput(maybe_match);
      if (maybe_output) outputs_.push_back(std::move(maybe_output));
    }
    if (!matcher && !group_matcher) {
      GENERIC_LOG_STREAM(ERROR, fmt::format("No matcher found for output {}", output_name));
      return false;
    }
  }

  for (auto& [group_name, group_match] : group_name_to_matches) {
    GroupMatcher* group_matcher = group_name_to_matcher[group_name];
    if (!group_matcher) {
      GENERIC_LOG_STREAM(ERROR, fmt::format("No group matcher found for group {}", group_name));
      return false;
    }

    group_match.name = group_name;
    group_match.metadata = onnx_model.getCustomMetadata(group_name);

    auto maybe_input = group_matcher->createInput(group_match);
    if (maybe_input) inputs_.push_back(std::move(maybe_input));
    auto maybe_output = group_matcher->createOutput(group_match);
    if (maybe_output) outputs_.push_back(std::move(maybe_output));
  }

  return true;
}

}  // namespace rai::cs::control::common::onnx
