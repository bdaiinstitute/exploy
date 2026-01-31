#include <fmt/core.h>
#include <optional>
#include <regex>
#include <string>
#include <string_view>

#include "onnx_components.hpp"
#include "onnx_matcher.hpp"

namespace rai::cs::control::common::onnx {

constexpr std::string_view alphanumeric = "[a-zA-Z0-9_]+";

bool IMUAngularVelocityMatcher::matches(const std::string& name) {
  std::regex pattern =
      std::regex(fmt::format("articulation\\.bodies\\.({})\\.ang_vel_body", alphanumeric));
  std::smatch match;
  if (std::regex_match(name, match, pattern) && match.size() > 1) {
    imu_name_ = match[1].str();
    return true;
  }
  return false;
}

std::unique_ptr<Input> IMUAngularVelocityMatcher::createInput(const Match& match) const {
  return std::make_unique<IMUAngularVelocityInput>(match.name, imu_name_);
}

bool JointPositionMatcher::matches(const std::string& name) {
  std::regex pattern = std::regex("articulation\\.joint\\.pos");
  return std::regex_match(name, pattern);
}

std::unique_ptr<Input> JointPositionMatcher::createInput(const Match& match) const {
  if (!match.metadata.has_value()) return nullptr;
  auto maybe_metadata = metadata::safe_json_get<metadata::JointMetadata>(match.metadata.value());
  if (!maybe_metadata.has_value()) return nullptr;
  return std::make_unique<JointPositionInput>(match.name, maybe_metadata.value().names);
}

bool JointVelocityMatcher::matches(const std::string& name) {
  std::regex pattern = std::regex("articulation\\.joint\\.vel");
  return std::regex_match(name, pattern);
}

std::unique_ptr<Input> JointVelocityMatcher::createInput(const Match& match) const {
  if (!match.metadata.has_value()) return nullptr;
  auto maybe_metadata = metadata::safe_json_get<metadata::JointMetadata>(match.metadata.value());
  if (!maybe_metadata.has_value()) return nullptr;
  return std::make_unique<JointVelocityInput>(match.name, maybe_metadata.value().names);
}

bool BasePositionMatcher::matches(const std::string& name) {
  std::regex pattern =
      std::regex(fmt::format("articulation\\.bodies\\.({})\\.pos_base_in_w", alphanumeric));
  return std::regex_match(name, pattern);
}

std::unique_ptr<Input> BasePositionMatcher::createInput(const Match& match) const {
  return std::make_unique<BasePositionInput>(match.name);
}

bool BaseOrientationMatcher::matches(const std::string& name) {
  std::regex pattern =
      std::regex(fmt::format("articulation\\.bodies\\.({})\\.world_Q_body", alphanumeric));
  return std::regex_match(name, pattern);
}

std::unique_ptr<Input> BaseOrientationMatcher::createInput(const Match& match) const {
  return std::make_unique<BaseOrientationInput>(match.name);
}

bool BaseLinearVelocityMatcher::matches(const std::string& name) {
  std::regex pattern =
      std::regex(fmt::format("articulation\\.bodies\\.({})\\.lin_vel_base_in_base", alphanumeric));
  return std::regex_match(name, pattern);
}

std::unique_ptr<Input> BaseLinearVelocityMatcher::createInput(const Match& match) const {
  return std::make_unique<BaseLinearVelocityInput>(match.name);
}

bool BaseAngularVelocityMatcher::matches(const std::string& name) {
  std::regex pattern =
      std::regex(fmt::format("articulation\\.bodies\\.({})\\.ang_vel_base_in_base", alphanumeric));
  return std::regex_match(name, pattern);
}

std::unique_ptr<Input> BaseAngularVelocityMatcher::createInput(const Match& match) const {
  return std::make_unique<BaseAngularVelocityInput>(match.name);
}

std::optional<std::string> JointTargetMatcher::matches(const std::string& name) {
  std::regex pattern = std::regex(fmt::format("(output\\.joint_targets)\\.(pos|vel|effort)"));
  std::smatch match;
  if (std::regex_match(name, match, pattern)) {
    auto group_name = match[1].str();
    return group_name;
  }
  return std::nullopt;
}

std::unique_ptr<Output> JointTargetMatcher::createOutput(const GroupMatch& match) const {
  if (!match.metadata.has_value()) return nullptr;
  auto maybe_metadata =
      metadata::safe_json_get<metadata::JointOutputMetadata>(match.metadata.value());
  if (!maybe_metadata.has_value()) return nullptr;
  return std::make_unique<JointTargetOutput>(
      fmt::format("{}.pos", match.name), fmt::format("{}.vel", match.name),
      fmt::format("{}.effort", match.name), maybe_metadata.value());
}

bool SE2VelocityMatcher::matches(const std::string& name) {
  std::regex pattern = std::regex(fmt::format("output\\.se2_velocity"));
  return std::regex_match(name, pattern);
}

std::unique_ptr<Output> SE2VelocityMatcher::createOutput(const Match& match) const {
  if (!match.metadata.has_value()) return nullptr;
  auto maybe_metadata =
      metadata::safe_json_get<metadata::Se2VelocityOutputMetadata>(match.metadata.value());
  if (!maybe_metadata.has_value()) return nullptr;
  return std::make_unique<SE2VelocityOutput>(match.name, maybe_metadata.value());
}

std::optional<std::string> HeightScanMatcher::matches(const std::string& name) {
  std::regex pattern =
      std::regex(fmt::format("(sensor\\.height_scanner\\.{})\\.(height|r|g|b)", alphanumeric));
  std::smatch match;
  if (std::regex_match(name, match, pattern)) {
    auto group_name = match[1].str();
    return group_name;
  }
  return std::nullopt;
}

std::unique_ptr<Input> HeightScanMatcher::createInput(const GroupMatch& match) const {
  if (!match.metadata.has_value()) return nullptr;
  auto maybe_metadata =
      metadata::safe_json_get<metadata::HeightScanMetadata>(match.metadata.value());
  if (!maybe_metadata.has_value()) return nullptr;
  return std::make_unique<HeightScanInput>(match.name, maybe_metadata.value());
}

bool RangeImageMatcher::matches(const std::string& name) {
  std::regex pattern = std::regex(fmt::format("sensor\\.range_image\\.({})", alphanumeric));
  return std::regex_match(name, pattern);
}

std::unique_ptr<Input> RangeImageMatcher::createInput(const Match& match) const {
  if (!match.metadata.has_value()) return nullptr;
  auto maybe_metadata =
      metadata::safe_json_get<metadata::RangeImageMetadata>(match.metadata.value());
  if (!maybe_metadata.has_value()) return nullptr;
  return std::make_unique<RangeImageInput>(match.name, maybe_metadata.value());
}

bool DepthImageMatcher::matches(const std::string& name) {
  std::regex pattern = std::regex(fmt::format("sensor\\.depth_image\\.({})", alphanumeric));
  return std::regex_match(name, pattern);
}

std::unique_ptr<Input> DepthImageMatcher::createInput(const Match& match) const {
  if (!match.metadata.has_value()) return nullptr;
  auto maybe_metadata =
      metadata::safe_json_get<metadata::DepthImageMetadata>(match.metadata.value());
  if (!maybe_metadata.has_value()) return nullptr;
  return std::make_unique<DepthImageInput>(match.name, maybe_metadata.value());
}

bool BodyOrientationMatcher::matches(const std::string& name) {
  std::smatch match;
  std::regex pattern = std::regex(fmt::format("rigid_bodies\\.({})\\.body_Q_body", alphanumeric));
  if (std::regex_match(name, match, pattern) && match.size() > 1) {
    body_name_ = match[1].str();
    return true;
  }
  return std::regex_match(name, pattern);
}

std::unique_ptr<Input> BodyOrientationMatcher::createInput(const Match& match) const {
  return std::make_unique<BodyOrientationInput>(match.name, body_name_);
}

bool SE3PoseMatcher::matches(const std::string& name) {
  std::smatch match;
  std::regex pattern = std::regex(fmt::format("command\\.se3_pose\\.({})", alphanumeric));
  if (std::regex_match(name, match, pattern)) {
    command_name_ = match[1].str();
    return true;
  }
  return std::regex_match(name, pattern);
}

std::unique_ptr<Input> SE3PoseMatcher::createInput(const Match& match) const {
  return std::make_unique<SE3PoseInput>(match.name, command_name_);
}

bool CommandSE2VelocityMatcher::matches(const std::string& name) {
  std::smatch match;
  std::regex pattern = std::regex(fmt::format("command\\.se2_velocity\\.({})", alphanumeric));
  if (std::regex_match(name, match, pattern)) {
    command_name_ = match[1].str();
    return true;
  }
  return false;
}

std::unique_ptr<Input> CommandSE2VelocityMatcher::createInput(const Match& match) const {
  if (!match.metadata.has_value()) return nullptr;
  auto maybe_metadata =
      metadata::safe_json_get<metadata::SE2VelocityCommandMetadata>(match.metadata.value());
  if (!maybe_metadata.has_value()) return nullptr;
  return std::make_unique<CommandSE2VelocityInput>(match.name, command_name_,
                                                   maybe_metadata.value());
}

std::optional<std::string> MemoryMatcher::matches(const std::string& name) {
  std::regex pattern = std::regex(fmt::format("(memory\\..*)\\.(in|out)"));
  std::smatch match;
  if (std::regex_match(name, match, pattern) && match.size() > 1) {
    return match[1].str();
  }
  return std::nullopt;
}

std::unique_ptr<Output> MemoryMatcher::createOutput(const GroupMatch& match) const {
  return std::make_unique<MemoryOutput>(match.name);
}

bool StepCountMatcher::matches(const std::string& name) {
  return name == "step_count";
}

std::unique_ptr<Input> StepCountMatcher::createInput(const Match& match) const {
  return std::make_unique<StepCountInput>(match.name);
}

}  // namespace rai::cs::control::common::onnx
