#include <fmt/core.h>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "onnx_components.hpp"
#include "onnx_matcher.hpp"

namespace rai::cs::control::common::onnx {

bool IMUAngularVelocityMatcher::matches(const Match& maybe_match) {
  std::regex pattern =
      std::regex(fmt::format("articulation\\.bodies\\.({})\\.ang_vel_body", alphanumeric));
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

bool JointPositionMatcher::matches(const Match& maybe_match) {
  std::regex pattern = std::regex("articulation\\.joint\\.pos");
  if (std::regex_match(maybe_match.name, pattern)) {
    found_matches_[maybe_match.name] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> JointPositionMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    if (!match.metadata.has_value()) continue;
    auto maybe_metadata = metadata::safe_json_get<metadata::JointMetadata>(match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    inputs.push_back(
        std::make_unique<JointPositionInput>(match.name, maybe_metadata.value().names));
  }
  return inputs;
}

bool JointVelocityMatcher::matches(const Match& maybe_match) {
  std::regex pattern = std::regex("articulation\\.joint\\.vel");
  if (std::regex_match(maybe_match.name, pattern)) {
    found_matches_[maybe_match.name] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> JointVelocityMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    if (!match.metadata.has_value()) continue;
    auto maybe_metadata = metadata::safe_json_get<metadata::JointMetadata>(match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    inputs.push_back(
        std::make_unique<JointVelocityInput>(match.name, maybe_metadata.value().names));
  }
  return inputs;
}

bool BasePositionMatcher::matches(const Match& maybe_match) {
  std::regex pattern =
      std::regex(fmt::format("articulation\\.bodies\\.({})\\.pos_base_in_w", alphanumeric));
  if (std::regex_match(maybe_match.name, pattern)) {
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
  std::regex pattern =
      std::regex(fmt::format("articulation\\.bodies\\.({})\\.world_Q_body", alphanumeric));
  if (std::regex_match(maybe_match.name, pattern)) {
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
  std::regex pattern =
      std::regex(fmt::format("articulation\\.bodies\\.({})\\.lin_vel_base_in_base", alphanumeric));
  if (std::regex_match(maybe_match.name, pattern)) {
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
  std::regex pattern =
      std::regex(fmt::format("articulation\\.bodies\\.({})\\.ang_vel_base_in_base", alphanumeric));
  if (std::regex_match(maybe_match.name, pattern)) {
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

bool JointTargetMatcher::matches(const Match& maybe_match) {
  std::regex pattern = std::regex(fmt::format("(output\\.joint_targets)\\.(pos|vel|effort)"));
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
  std::regex pattern = std::regex(fmt::format("output\\.se2_velocity"));
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

bool RangeImageMatcher::matches(const Match& maybe_match) {
  std::regex pattern = std::regex(fmt::format("sensor\\.range_image\\.({})", alphanumeric));
  if (std::regex_match(maybe_match.name, pattern)) {
    found_matches_[maybe_match.name] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> RangeImageMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    if (!match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::RangeImageMetadata>(match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    inputs.push_back(std::make_unique<RangeImageInput>(match.name, maybe_metadata.value()));
  }
  return inputs;
}

bool DepthImageMatcher::matches(const Match& maybe_match) {
  std::regex pattern = std::regex(fmt::format("sensor\\.depth_image\\.({})", alphanumeric));
  if (std::regex_match(maybe_match.name, pattern)) {
    found_matches_[maybe_match.name] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> DepthImageMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    if (!match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::DepthImageMetadata>(match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    inputs.push_back(std::make_unique<DepthImageInput>(match.name, maybe_metadata.value()));
  }
  return inputs;
}

bool BodyOrientationMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  std::regex pattern = std::regex(fmt::format("rigid_bodies\\.({})\\.body_Q_body", alphanumeric));
  if (std::regex_match(maybe_match.name, match, pattern) && match.size() > 1) {
    found_matches_[match[1].str()] = maybe_match;
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

bool SE3PoseMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  std::regex pattern = std::regex(fmt::format("command\\.se3_pose\\.({})", alphanumeric));
  if (std::regex_match(maybe_match.name, match, pattern)) {
    found_matches_[match[1].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> SE3PoseMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    inputs.push_back(std::make_unique<SE3PoseInput>(match.name, name));
  }
  return inputs;
}

bool CommandSE2VelocityMatcher::matches(const Match& maybe_match) {
  std::smatch match;
  std::regex pattern = std::regex(fmt::format("command\\.se2_velocity\\.({})", alphanumeric));
  if (std::regex_match(maybe_match.name, match, pattern)) {
    found_matches_[match[1].str()] = maybe_match;
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<Input>> CommandSE2VelocityMatcher::createInputs() const {
  std::vector<std::unique_ptr<Input>> inputs;
  for (const auto& [name, match] : found_matches_) {
    if (!match.metadata.has_value()) continue;
    auto maybe_metadata =
        metadata::safe_json_get<metadata::SE2VelocityCommandMetadata>(match.metadata.value());
    if (!maybe_metadata.has_value()) continue;
    inputs.push_back(
        std::make_unique<CommandSE2VelocityInput>(match.name, name, maybe_metadata.value()));
  }
  return inputs;
}

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
  if (maybe_match.name == "step_count") {
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

}  // namespace rai::cs::control::common::onnx
