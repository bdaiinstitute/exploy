// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <optional>
#include <ranges>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "components.hpp"
#include "matcher.hpp"

namespace exploy::control {

namespace {

// Builds a regex pattern for base tensor matching.
// Returns std::nullopt when base_names is empty, causing matchers to reject the tensor.
std::optional<std::regex> buildBasePattern(
    const std::unordered_map<std::string, std::string>& base_names, std::string_view field) {
  if (base_names.empty()) return std::nullopt;
  auto pairs = base_names | std::views::transform([](const auto& p) {
                 return fmt::format(R"({}\.{})", p.first, p.second);
               });
  return std::regex(fmt::format(R"(obj\.({})\.{})", fmt::join(pairs, "|"), field));
}

}  // namespace

// ---------------  Joint matchers --------------------------------
bool JointMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern_) && match.size() > 1) {
    auto group_name = match[1].str();
    found_matches_[group_name].input_matches.push_back(maybe_match);
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> JointMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [group_name, group_match] : found_matches_) {
    if (!group_match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::JointMetadata>(group_match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    auto joint_names = maybe_metadata.value().names;

    std::string type;
    for (const auto& input_match : group_match.input_matches) {
      std::smatch match;
      if (std::regex_match(input_match.name, match, pattern_) && match.size() > 2) {
        type = match[2].str();
        if (type == "pos") {
          inputs.push_back(std::make_unique<JointPositionInput>(input_match.name, joint_names));
        } else if (type == "vel") {
          inputs.push_back(std::make_unique<JointVelocityInput>(input_match.name, joint_names));
        }
      }
    }
  }
  return inputs;
}

// ---------------------------------------------------------------

// ---------------  Base matchers --------------------------------
bool BasePositionMatcher::matches(const Match& maybe_match) {
  auto maybe_pattern = buildBasePattern(maybe_match.base_names, "pos_b_rt_w_in_w");
  if (maybe_pattern.has_value() && std::regex_match(maybe_match.name, maybe_pattern.value())) {
    found_matches_[maybe_match.name] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> BasePositionMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(std::make_unique<BasePositionInput>(match.name));
  }
  return inputs;
}

bool BaseOrientationMatcher::matches(const Match& maybe_match) {
  auto maybe_pattern = buildBasePattern(maybe_match.base_names, "w_Q_b");
  if (maybe_pattern.has_value() && std::regex_match(maybe_match.name, maybe_pattern.value())) {
    found_matches_[maybe_match.name] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> BaseOrientationMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(std::make_unique<BaseOrientationInput>(match.name));
  }
  return inputs;
}

bool BaseLinearVelocityMatcher::matches(const Match& maybe_match) {
  auto maybe_pattern = buildBasePattern(maybe_match.base_names, "lin_vel_b_rt_w_in_b");
  if (maybe_pattern.has_value() && std::regex_match(maybe_match.name, maybe_pattern.value())) {
    found_matches_[maybe_match.name] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> BaseLinearVelocityMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(std::make_unique<BaseLinearVelocityInput>(match.name));
  }
  return inputs;
}

bool BaseAngularVelocityMatcher::matches(const Match& maybe_match) {
  auto maybe_pattern = buildBasePattern(maybe_match.base_names, "ang_vel_b_rt_w_in_b");
  if (maybe_pattern.has_value() && std::regex_match(maybe_match.name, maybe_pattern.value())) {
    found_matches_[maybe_match.name] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> BaseAngularVelocityMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(std::make_unique<BaseAngularVelocityInput>(match.name));
  }
  return inputs;
}
// ---------------------------------------------------------------

// ---------------  Output matchers --------------------------------
bool JointTargetMatcher::matches(const Match& maybe_match) {
  std::regex pattern =
      std::regex(fmt::format("(output\\.joint_targets\\.{})\\.(pos|vel|effort)", kAlphanumeric));
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern)) {
    auto group_name = match[1].str();
    auto& group_match = found_matches_[group_name];
    group_match.name = group_name;
    if (!group_match.metadata.has_value() && maybe_match.metadata.has_value()) {
      group_match.metadata = maybe_match.metadata;
    }
    group_match.output_matches.push_back(maybe_match);
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Output>> JointTargetMatcher::createOutputs() const {
  std::vector<std::unique_ptr<Output>> outputs;
  for (const auto& [name, match] : found_matches_) {
    if (!match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::JointOutputMetadata>(match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    outputs.push_back(std::make_unique<JointTargetOutput>(
        fmt::format("{}.pos", match.name), fmt::format("{}.vel", match.name),
        fmt::format("{}.effort", match.name), maybe_metadata.value()));
  }
  return outputs;
}

bool SE2VelocityMatcher::matches(const Match& maybe_match) {
  std::regex pattern = std::regex(fmt::format("output\\.se2_velocity\\.{}", kAlphanumeric));
  if (std::regex_match(maybe_match.name, pattern)) {
    found_matches_[maybe_match.name] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Output>> SE2VelocityMatcher::createOutputs() const {
  std::vector<std::unique_ptr<Output>> outputs;
  for (const auto& [name, match] : found_matches_) {
    if (!match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::Se2VelocityOutputMetadata>(match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    outputs.push_back(std::make_unique<SE2VelocityOutput>(match.name, maybe_metadata.value()));
  }
  return outputs;
}
// ---------------------------------------------------------------

// ---------------  Sensor matchers ------------------------------
bool IMUAngularVelocityMatcher::matches(const Match& maybe_match) {
  std::regex pattern =
      std::regex(fmt::format("sensor\\.imu\\.({})\\.ang_vel_b_rt_w_in_b", kAlphanumeric));
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern) && match.size() > 1) {
    found_matches_[match[1].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> IMUAngularVelocityMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [imu_name, found_match] : found_matches_) {
    inputs.push_back(std::make_unique<IMUAngularVelocityInput>(found_match.name, imu_name));
  }
  return inputs;
}

bool IMUOrientationMatcher::matches(const Match& maybe_match) {
  std::regex pattern = std::regex(fmt::format("sensor\\.imu\\.({})\\.w_Q_b", kAlphanumeric));
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern) && match.size() > 1) {
    found_matches_[match[1].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> IMUOrientationMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [imu_name, found_match] : found_matches_) {
    inputs.push_back(std::make_unique<IMUOrientationInput>(found_match.name, imu_name));
  }
  return inputs;
}

bool HeightScanMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern_) && match.size() > 1) {
    auto group_name = match[1].str();
    found_matches_[group_name].input_matches.push_back(maybe_match);
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> HeightScanMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [group_name, group_match] : found_matches_) {
    if (!group_match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::HeightScanMetadata>(group_match.metadata.value());
    if (!maybe_metadata.has_value()) continue;

    std::unordered_set<std::string> layer_names;
    std::string sensor_name;
    for (const auto& input_match : group_match.input_matches) {
      std::smatch match;
      if (std::regex_match(input_match.name, match, pattern_) && match.size() > 3) {
        layer_names.insert(match[3].str());
        sensor_name = match[2].str();
      }
    }

    inputs.push_back(std::make_unique<HeightScanInput>(group_name, sensor_name, layer_names,
                                                       maybe_metadata.value()));
  }
  return inputs;
}

bool SphericalImageMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern_) && match.size() > 1) {
    auto group_name = match[1].str();
    found_matches_[group_name].input_matches.push_back(maybe_match);
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> SphericalImageMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [group_name, group_match] : found_matches_) {
    if (!group_match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::SphericalImageMetadata>(group_match.metadata.value());
    if (!maybe_metadata.has_value()) continue;

    std::unordered_set<std::string> channel_names;
    std::string sensor_name;
    for (const auto& input_match : group_match.input_matches) {
      std::smatch match;
      if (std::regex_match(input_match.name, match, pattern_) && match.size() > 3) {
        channel_names.insert(match[3].str());
        sensor_name = match[2].str();
      }
    }

    inputs.push_back(std::make_unique<SphericalImageInput>(group_name, sensor_name, channel_names,
                                                           maybe_metadata.value()));
  }
  return inputs;
}

bool PinholeImageMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern_) && match.size() > 1) {
    auto group_name = match[1].str();
    found_matches_[group_name].input_matches.push_back(maybe_match);
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> PinholeImageMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [group_name, group_match] : found_matches_) {
    if (!group_match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::PinholeImageMetadata>(group_match.metadata.value());
    if (!maybe_metadata.has_value()) continue;

    std::unordered_set<std::string> channel_names;
    std::string sensor_name;
    for (const auto& input_match : group_match.input_matches) {
      std::smatch match;
      if (std::regex_match(input_match.name, match, pattern_) && match.size() > 3) {
        channel_names.insert(match[3].str());
        sensor_name = match[2].str();
      }
    }

    inputs.push_back(std::make_unique<PinholeImageInput>(group_name, sensor_name, channel_names,
                                                         maybe_metadata.value()));
  }
  return inputs;
}
// ---------------------------------------------------------------

// ---------------  Body matchers ------------------------------
bool BodyPositionMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  std::regex pattern = std::regex(
      fmt::format("obj\\.({})\\.bodies\\.({})\\.pos_b_rt_w_in_w", kAlphanumeric, kAlphanumeric));
  if (std::regex_match(maybe_match.name, match, pattern) && match.size() > 2) {
    found_matches_[match[2].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> BodyPositionMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(std::make_unique<BodyPositionInput>(match.name, name));
  }
  return inputs;
}

bool BodyOrientationMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  std::regex pattern =
      std::regex(fmt::format("obj\\.({})\\.bodies\\.({})\\.w_Q_b", kAlphanumeric, kAlphanumeric));
  if (std::regex_match(maybe_match.name, match, pattern) && match.size() > 2) {
    found_matches_[match[2].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> BodyOrientationMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(std::make_unique<BodyOrientationInput>(match.name, name));
  }
  return inputs;
}
// ---------------------------------------------------------------

// ---------------  Command matchers ------------------------------
bool CommandSE3PoseMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  std::regex pattern = std::regex(fmt::format("cmd\\.se3_pose\\.({})", kAlphanumeric));
  if (std::regex_match(maybe_match.name, match, pattern)) {
    found_matches_[match[1].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> CommandSE3PoseMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(std::make_unique<CommandSE3PoseInput>(match.name, name));
  }
  return inputs;
}

bool CommandBooleanMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  std::regex pattern = std::regex(fmt::format("cmd\\.boolean\\.({})", kAlphanumeric));
  if (std::regex_match(maybe_match.name, match, pattern)) {
    found_matches_[match[1].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> CommandBooleanMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(std::make_unique<CommandBooleanInput>(match.name, name));
  }
  return inputs;
}

bool CommandFloatMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  std::regex pattern = std::regex(fmt::format("cmd\\.float\\.({})", kAlphanumeric));
  if (std::regex_match(maybe_match.name, match, pattern)) {
    found_matches_[match[1].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> CommandFloatMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    metadata::FloatCommandMetadata metadata;
    if (match.metadata.has_value()) {
      auto maybe_metadata =
          metadata::safe_json_get<metadata::FloatCommandMetadata>(match.metadata.value());
      if (maybe_metadata.has_value()) metadata = maybe_metadata.value();
    }
    inputs.push_back(std::make_unique<CommandFloatInput>(match.name, name, metadata));
  }
  return inputs;
}

bool CommandSE2VelocityMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  std::regex pattern = std::regex(fmt::format("cmd\\.se2_velocity\\.({})", kAlphanumeric));
  if (std::regex_match(maybe_match.name, match, pattern)) {
    found_matches_[match[1].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> CommandSE2VelocityMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    metadata::SE2VelocityCommandMetadata metadata;
    if (match.metadata.has_value()) {
      auto maybe_metadata =
          metadata::safe_json_get<metadata::SE2VelocityCommandMetadata>(match.metadata.value());
      if (maybe_metadata.has_value()) metadata = maybe_metadata.value();
    }
    inputs.push_back(std::make_unique<CommandSE2VelocityInput>(match.name, name, metadata));
  }
  return inputs;
}
// ---------------------------------------------------------------

// ---------------  Context matcher ------------------------------
bool MemoryMatcher::matches(const Match& maybe_match) {
  std::regex pattern = std::regex(fmt::format("memory\\.(.*)\\.(in|out)"));
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern) && match.size() > 1) {
    found_matches_[match[1].str()].output_matches.push_back(maybe_match);
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Output>> MemoryMatcher::createOutputs() const {
  std::vector<std::unique_ptr<Output>> outputs;
  for (const auto& [name, match] : found_matches_) {
    outputs.push_back(std::make_unique<MemoryOutput>(name));
  }
  return outputs;
}

bool StepCountMatcher::matches(const Match& maybe_match) {
  if (maybe_match.name == "ctx.step_count") {
    found_matches_[maybe_match.name] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> StepCountMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(std::make_unique<StepCountInput>(match.name));
  }
  return inputs;
}
// ---------------------------------------------------------------

}  // namespace exploy::control
