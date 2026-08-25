// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <ranges>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "exploy/components.hpp"
#include "exploy/matcher.hpp"

namespace exploy::control {

// ---------------  Joint matchers --------------------------------
JointMatcher::JointMatcher() : GroupMatcher("JointMatcher") {}

bool JointMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern_) && match.size() > 3) {
    auto group_name = match[1].str();
    found_matches_[group_name].input_matches.push_back(maybe_match);
    articulation_names_[group_name] = match[2].str();
    input_types_[maybe_match.name] = match[3].str();
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
    const auto& articulation_name = articulation_names_.at(group_name);

    for (const auto& input_match : group_match.input_matches) {
      const auto& type = input_types_.at(input_match.name);
      if (type == "pos") {
        inputs.push_back(
            std::make_unique<JointPositionInput>(input_match.name, articulation_name, joint_names));
      } else if (type == "vel") {
        inputs.push_back(
            std::make_unique<JointVelocityInput>(input_match.name, articulation_name, joint_names));
      }
    }
  }
  return inputs;
}

void JointMatcher::reset() {
  articulation_names_.clear();
  input_types_.clear();
}

// ---------------------------------------------------------------

// ---------------  Base matchers --------------------------------
BaseMatcher::BaseMatcher(const std::string& name, std::string field)
    : Matcher(name), field_(std::move(field)) {}

bool BaseMatcher::matches(const Match& maybe_match) {
  for (const auto& [articulation, base] : maybe_match.base_names) {
    if (maybe_match.name == fmt::format("obj.{}.{}.{}", articulation, base, field_)) {
      found_matches_[maybe_match.name] = maybe_match;
      articulation_names_[maybe_match.name] = articulation;
      return true;
    }
  }
  return false;
}

void BaseMatcher::reset() {
  articulation_names_.clear();
}

BasePositionMatcher::BasePositionMatcher()
    : BaseMatcher("BasePositionMatcher", "pos_b_rt_w_in_w") {}

std::vector<std::unique_ptr<Input>> BasePositionMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(std::make_unique<BasePositionInput>(match.name, articulation_names_.at(name)));
  }
  return inputs;
}

BaseOrientationMatcher::BaseOrientationMatcher() : BaseMatcher("BaseOrientationMatcher", "w_Q_b") {}

std::vector<std::unique_ptr<Input>> BaseOrientationMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(
        std::make_unique<BaseOrientationInput>(match.name, articulation_names_.at(name)));
  }
  return inputs;
}

BaseLinearVelocityMatcher::BaseLinearVelocityMatcher()
    : BaseMatcher("BaseLinearVelocityMatcher", "lin_vel_b_rt_w_in_b") {}

std::vector<std::unique_ptr<Input>> BaseLinearVelocityMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(
        std::make_unique<BaseLinearVelocityInput>(match.name, articulation_names_.at(name)));
  }
  return inputs;
}

BaseAngularVelocityMatcher::BaseAngularVelocityMatcher()
    : BaseMatcher("BaseAngularVelocityMatcher", "ang_vel_b_rt_w_in_b") {}

std::vector<std::unique_ptr<Input>> BaseAngularVelocityMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(
        std::make_unique<BaseAngularVelocityInput>(match.name, articulation_names_.at(name)));
  }
  return inputs;
}
// ---------------------------------------------------------------

// ---------------  Output matchers --------------------------------
JointTargetMatcher::JointTargetMatcher() : GroupMatcher("JointTargetMatcher") {}

bool JointTargetMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern_) && match.size() > 3) {
    auto group_name = match[1].str();
    auto& group_match = found_matches_[group_name];
    group_match.name = group_name;
    if (!group_match.metadata.has_value() && maybe_match.metadata.has_value()) {
      group_match.metadata = maybe_match.metadata;
    }
    group_match.output_matches.push_back(maybe_match);
    articulation_names_[group_name] = match[2].str();
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
        fmt::format("{}.effort", match.name), articulation_names_.at(name),
        maybe_metadata.value()));
  }
  return outputs;
}

void JointTargetMatcher::reset() {
  articulation_names_.clear();
}

SE2VelocityMatcher::SE2VelocityMatcher() : Matcher("SE2VelocityMatcher") {}

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
IMULinearVelocityMatcher::IMULinearVelocityMatcher() : Matcher("IMULinearVelocityMatcher") {}

bool IMULinearVelocityMatcher::matches(const Match& maybe_match) {
  std::regex pattern =
      std::regex(fmt::format("sensor\\.imu\\.({})\\.lin_vel_b_rt_w_in_b", kAlphanumeric));
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern) && match.size() > 1) {
    found_matches_[match[1].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> IMULinearVelocityMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [imu_name, found_match] : found_matches_) {
    inputs.push_back(std::make_unique<IMULinearVelocityInput>(found_match.name, imu_name));
  }
  return inputs;
}

IMUAngularVelocityMatcher::IMUAngularVelocityMatcher() : Matcher("IMUAngularVelocityMatcher") {}

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

IMUOrientationMatcher::IMUOrientationMatcher() : Matcher("IMUOrientationMatcher") {}

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

SensorGroupMatcher::SensorGroupMatcher(const std::string& name, std::regex pattern)
    : GroupMatcher(name), pattern_(std::move(pattern)) {}

bool SensorGroupMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern_) && match.size() > 3) {
    auto group_name = match[1].str();
    found_matches_[group_name].input_matches.push_back(maybe_match);
    sensor_names_[group_name] = match[2].str();
    channel_names_[group_name].insert(match[3].str());
    return true;
  }
  return false;
}

void SensorGroupMatcher::reset() {
  sensor_names_.clear();
  channel_names_.clear();
}

HeightScanMatcher::HeightScanMatcher()
    : SensorGroupMatcher(
          "HeightScanMatcher",
          std::regex(fmt::format("(sensor\\.ray_caster\\.({}))\\.(height|r|g|b)", kAlphanumeric))) {
}

std::vector<std::unique_ptr<Input>> HeightScanMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [group_name, group_match] : found_matches_) {
    if (!group_match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::HeightScanMetadata>(group_match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    inputs.push_back(std::make_unique<HeightScanInput>(group_name, sensor_names_.at(group_name),
                                                       channel_names_.at(group_name),
                                                       maybe_metadata.value()));
  }
  return inputs;
}

SphericalImageMatcher::SphericalImageMatcher()
    : SensorGroupMatcher("SphericalImageMatcher",
                         std::regex(fmt::format("(sensor\\.spherical_image\\.({}))\\.({})",
                                                kAlphanumeric, kAlphanumeric))) {}

std::vector<std::unique_ptr<Input>> SphericalImageMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [group_name, group_match] : found_matches_) {
    if (!group_match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::SphericalImageMetadata>(group_match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    inputs.push_back(std::make_unique<SphericalImageInput>(group_name, sensor_names_.at(group_name),
                                                           channel_names_.at(group_name),
                                                           maybe_metadata.value()));
  }
  return inputs;
}

PinholeImageMatcher::PinholeImageMatcher()
    : SensorGroupMatcher("PinholeImageMatcher",
                         std::regex(fmt::format("(sensor\\.pinhole_image\\.({}))\\.({})",
                                                kAlphanumeric, kAlphanumeric))) {}

std::vector<std::unique_ptr<Input>> PinholeImageMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [group_name, group_match] : found_matches_) {
    if (!group_match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::PinholeImageMetadata>(group_match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    inputs.push_back(std::make_unique<PinholeImageInput>(group_name, sensor_names_.at(group_name),
                                                         channel_names_.at(group_name),
                                                         maybe_metadata.value()));
  }
  return inputs;
}
// ---------------------------------------------------------------

// ---------------  Body matchers ------------------------------
BodyMatcher::BodyMatcher(const std::string& name, std::string field)
    : Matcher(name), field_(std::move(field)) {}

bool BodyMatcher::matches(const Match& maybe_match) {
  std::regex pattern;
  if (maybe_match.base_names.empty()) {
    pattern =
        std::regex(fmt::format(R"(obj\.({})\.({})\.{})", kAlphanumeric, kAlphanumeric, field_));
  } else {
    auto pairs = maybe_match.base_names | std::views::transform([](const auto& p) {
                   return fmt::format(R"({}\.{}\.)", p.first, p.second);
                 });
    pattern = std::regex(fmt::format(R"(obj\.(?!{})({})\.({})\.{})", fmt::join(pairs, "|"),
                                     kAlphanumeric, kAlphanumeric, field_));
  }
  std::smatch match;
  if (std::regex_match(maybe_match.name, match, pattern) && match.size() > 2) {
    found_matches_[maybe_match.name] = maybe_match;
    articulation_and_body_[maybe_match.name] = {match[1].str(), match[2].str()};
    return true;
  }
  return false;
}

void BodyMatcher::reset() {
  articulation_and_body_.clear();
}

BodyPositionMatcher::BodyPositionMatcher()
    : BodyMatcher("BodyPositionMatcher", "pos_b_rt_w_in_w") {}

std::vector<std::unique_ptr<Input>> BodyPositionMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    const auto& [articulation, body] = articulation_and_body_.at(name);
    inputs.push_back(std::make_unique<BodyPositionInput>(match.name, articulation, body));
  }
  return inputs;
}

BodyOrientationMatcher::BodyOrientationMatcher() : BodyMatcher("BodyOrientationMatcher", "w_Q_b") {}

std::vector<std::unique_ptr<Input>> BodyOrientationMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    const auto& [articulation, body] = articulation_and_body_.at(name);
    inputs.push_back(std::make_unique<BodyOrientationInput>(match.name, articulation, body));
  }
  return inputs;
}

BodyLinearVelocityMatcher::BodyLinearVelocityMatcher()
    : BodyMatcher("BodyLinearVelocityMatcher", "lin_vel_b_rt_w_in_b") {}

std::vector<std::unique_ptr<Input>> BodyLinearVelocityMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    const auto& [articulation, body] = articulation_and_body_.at(name);
    inputs.push_back(std::make_unique<BodyLinearVelocityInput>(match.name, articulation, body));
  }
  return inputs;
}

BodyAngularVelocityMatcher::BodyAngularVelocityMatcher()
    : BodyMatcher("BodyAngularVelocityMatcher", "ang_vel_b_rt_w_in_b") {}

std::vector<std::unique_ptr<Input>> BodyAngularVelocityMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    const auto& [articulation, body] = articulation_and_body_.at(name);
    inputs.push_back(std::make_unique<BodyAngularVelocityInput>(match.name, articulation, body));
  }
  return inputs;
}
// ---------------------------------------------------------------

// ---------------  Command matchers ------------------------------
CommandSE3PoseMatcher::CommandSE3PoseMatcher() : Matcher("CommandSE3PoseMatcher") {}

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

CommandBooleanMatcher::CommandBooleanMatcher() : Matcher("CommandBooleanMatcher") {}

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

CommandFloatMatcher::CommandFloatMatcher() : Matcher("CommandFloatMatcher") {}

bool CommandFloatMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  // Float commands are a dot-separated path: a scalar command (`cmd.float.<command>`) or
  // a field of a composite one (`cmd.float.<command>.<field>`), at arbitrary depth.
  std::regex pattern =
      std::regex(fmt::format("cmd\\.float\\.({}(?:\\.{})*)", kAlphanumeric, kAlphanumeric));
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

CommandJointPositionMatcher::CommandJointPositionMatcher()
    : Matcher("CommandJointPositionMatcher") {}

bool CommandJointPositionMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  std::regex pattern = std::regex(fmt::format("cmd\\.joint_pos\\.({})", kAlphanumeric));
  if (std::regex_match(maybe_match.name, match, pattern)) {
    found_matches_[match[1].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> CommandJointPositionMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    if (!match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::JointPositionCommandMetadata>(match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    if (maybe_metadata.value().joint_names.empty()) continue;
    inputs.push_back(
        std::make_unique<CommandJointPositionInput>(match.name, name, maybe_metadata.value()));
  }
  return inputs;
}

CommandSE2VelocityMatcher::CommandSE2VelocityMatcher() : Matcher("CommandSE2VelocityMatcher") {}

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
MemoryMatcher::MemoryMatcher() : GroupMatcher("MemoryMatcher") {}

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

StepCountMatcher::StepCountMatcher() : Matcher("StepCountMatcher") {}

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
