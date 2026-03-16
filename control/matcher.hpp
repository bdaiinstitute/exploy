// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <functional>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "components.hpp"

namespace exploy::control {

/**
 * @brief Alphanumeric pattern for regex matching.
 *
 * Pattern string used in regex expressions to match alphanumeric identifiers
 * including underscores.
 */
constexpr std::string_view kAlphanumeric = "[a-zA-Z0-9_]+";

/**
 * @brief Represents a single matched ONNX tensor.
 *
 * Contains the tensor name and optional metadata associated with the tensor
 * from the ONNX model.
 */
struct Match {
  std::string name{};                                       ///< ONNX tensor name.
  std::optional<std::string> metadata{};                    ///< Optional JSON metadata string.
  std::unordered_map<std::string, std::string> base_names;  ///< Map of base names.
};

/**
 * @brief Represents a group of related matched ONNX tensors.
 *
 * Used by GroupMatcher to handle patterns where multiple tensors belong together
 * (e.g., joint position and velocity inputs for the same joint group).
 */
struct GroupMatch {
  std::string name{};                     ///< Group identifier name.
  std::optional<std::string> metadata{};  ///< Optional JSON metadata for the group.
  std::vector<Match> input_matches{};     ///< Matched input tensors in this group.
  std::vector<Match> output_matches{};    ///< Matched output tensors in this group.
};

/**
 * @brief Abstract base class for single-tensor pattern matching.
 *
 * Matcher implementations recognize specific ONNX tensor name patterns and create
 * corresponding input/output components. Each matcher handles one type of component.
 */
class Matcher {
 public:
  virtual ~Matcher() = default;

  /**
   * @brief Check if a tensor matches this matcher's pattern.
   *
   * @param maybe_match Potential match containing tensor name and metadata.
   * @return true if the tensor matches and was stored, false otherwise.
   */
  virtual bool matches(const Match& maybe_match) = 0;

  /**
   * @brief Create input components from all matched tensors.
   *
   * @return Vector of input component unique pointers.
   */
  virtual std::vector<std::unique_ptr<Input>> createInputs() const { return {}; }

  /**
   * @brief Create output components from all matched tensors.
   *
   * @return Vector of output component unique pointers.
   */
  virtual std::vector<std::unique_ptr<Output>> createOutputs() const { return {}; }

 protected:
  std::unordered_map<std::string, Match> found_matches_;  ///< Storage for matched tensors.
};

/**
 * @brief Abstract base class for multi-tensor group pattern matching.
 *
 * GroupMatcher implementations recognize patterns where multiple related tensors
 * need to be processed together (e.g., joints.pos and joints.vel).
 */
class GroupMatcher {
 public:
  virtual ~GroupMatcher() = default;

  /**
   * @brief Check if a tensor matches this matcher's pattern and add to group.
   *
   * @param maybe_match Potential match containing tensor name and metadata.
   * @return true if the tensor matches and was added to a group, false otherwise.
   */
  virtual bool matches(const Match& maybe_match) = 0;

  /**
   * @brief Create input components from all matched tensor groups.
   *
   * @return Vector of input component unique pointers.
   */
  virtual std::vector<std::unique_ptr<Input>> createInputs() const { return {}; }

  /**
   * @brief Create output components from all matched tensor groups.
   *
   * @return Vector of output component unique pointers.
   */
  virtual std::vector<std::unique_ptr<Output>> createOutputs() const { return {}; }

  /**
   * @brief Populate metadata for all matched groups using a metadata getter function.
   *
   * @param get_metadata Function that retrieves metadata string for a given tensor name.
   */
  void populateGroupMetadata(
      std::function<std::optional<std::string>(const std::string&)> get_metadata) {
    for (auto& [name, group_match] : found_matches_) {
      group_match.metadata = get_metadata(name);
    }
  }

 protected:
  std::unordered_map<std::string, GroupMatch> found_matches_;  ///< Storage for matched groups.
};

// ---------------  Joint matchers --------------------------------
/**
 * @brief Matcher for joint position and velocity input tensors.
 *
 * Matches patterns like "obj.{robot_name}.joints.pos" and "obj.{robot_name}.joints.vel"
 * and creates JointPositionInput and JointVelocityInput components.
 */
class JointMatcher : public GroupMatcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;

 private:
  const std::regex pattern_ =
      std::regex(fmt::format("(obj\\.{}\\.joints)\\.(pos|vel)", kAlphanumeric));
};
// ---------------------------------------------------------------

// ---------------  Base matchers --------------------------------
/**
 * @brief Matcher for robot base position input tensors.
 *
 * Matches patterns for base position in world frame and creates BasePositionInput components.
 */
class BasePositionMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for robot base orientation input tensors.
 *
 * Matches patterns for base orientation (quaternion) in world frame and creates
 * BaseOrientationInput components.
 */
class BaseOrientationMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for robot base linear velocity input tensors.
 *
 * Matches patterns for base linear velocity and creates BaseLinearVelocityInput components.
 */
class BaseLinearVelocityMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for robot base angular velocity input tensors.
 *
 * Matches patterns for base angular velocity and creates BaseAngularVelocityInput components.
 */
class BaseAngularVelocityMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};
// ---------------------------------------------------------------

// ---------------  Output matchers --------------------------------
/**
 * @brief Matcher for joint target output tensors.
 *
 * Matches patterns for joint position, velocity, and effort outputs and creates
 * JointTargetOutput components with PD controller gains.
 */
class JointTargetMatcher : public GroupMatcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Output>> createOutputs() const override;
};

/**
 * @brief Matcher for SE(2) velocity output tensors.
 *
 * Matches patterns for planar velocity commands and creates SE2VelocityOutput components.
 */
class SE2VelocityMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Output>> createOutputs() const override;
};
// ---------------------------------------------------------------

// ---------------  Sensor matchers ------------------------------
/**
 * @brief Matcher for height scan sensor input tensors.
 *
 * Matches patterns for terrain height scan layers (height, r, g, b) and creates
 * HeightScanInput components with multi-layer support.
 */
class HeightScanMatcher : public GroupMatcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;

 private:
  const std::regex pattern_ =
      std::regex(fmt::format("(sensor\\.ray_caster\\.({}))\\.(height|r|g|b)", kAlphanumeric));
};

/**
 * @brief Matcher for spherical image sensor input tensors.
 *
 * Matches patterns for spherical image channels (e.g., range, risk) and creates
 * SphericalImageInput components with multi-channel support.
 */
class SphericalImageMatcher : public GroupMatcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;

 private:
  const std::regex pattern_ = std::regex(
      fmt::format("(sensor\\.spherical_image\\.({}))\\.({})", kAlphanumeric, kAlphanumeric));
};

/**
 * @brief Matcher for pinhole camera image input tensors.
 *
 * Matches patterns for pinhole image channels (e.g., depth, risk) and creates
 * PinholeImageInput components with multi-channel support.
 */
class PinholeImageMatcher : public GroupMatcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;

 private:
  const std::regex pattern_ = std::regex(
      fmt::format("(sensor\\.pinhole_image\\.({}))\\.({})", kAlphanumeric, kAlphanumeric));
};

/**
 * @brief Matcher for IMU angular velocity input tensors.
 *
 * Matches patterns for IMU angular velocity data and creates IMUAngularVelocityInput components.
 */
class IMUAngularVelocityMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for IMU orientation input tensors.
 *
 * Matches patterns for IMU orientation (quaternion) data and creates IMUOrientationInput
 * components.
 */
class IMUOrientationMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};
// ---------------------------------------------------------------

// ---------------  Command matchers ------------------------------
/**
 * @brief Matcher for SE(3) pose command input tensors.
 *
 * Matches patterns for SE(3) pose commands and creates CommandSE3PoseInput components.
 */
class CommandSE3PoseMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for SE(2) velocity command input tensors.
 *
 * Matches patterns for planar velocity commands and creates CommandSE2VelocityInput components.
 */
class CommandSE2VelocityMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for boolean command input tensors.
 *
 * Matches patterns for boolean selector commands and creates CommandBooleanInput components.
 */
class CommandBooleanMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for float command input tensors.
 *
 * Matches patterns for scalar float commands and creates CommandFloatInput components.
 */
class CommandFloatMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};
// ---------------------------------------------------------------

// ---------------  Body matchers --------------------------------
/**
 * @brief Matcher for rigid body position input tensors.
 *
 * Matches patterns for body position data and creates BodyPositionInput components.
 */
class BodyPositionMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for rigid body orientation input tensors.
 *
 * Matches patterns for body orientation (quaternion) data and creates BodyOrientationInput
 * components.
 */
class BodyOrientationMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};
// ---------------------------------------------------------------

// ---------------  Context matchers --------------------------------
/**
 * @brief Matcher for memory/state tensors for recurrent policies.
 *
 * Matches patterns for memory input/output pairs and creates MemoryOutput components
 * to maintain hidden state across control steps.
 */
class MemoryMatcher : public GroupMatcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Output>> createOutputs() const override;
};

/**
 * @brief Matcher for step count input tensors.
 *
 * Matches patterns for step counter tensors and creates StepCountInput components.
 */
class StepCountMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};
// ---------------------------------------------------------------

}  // namespace exploy::control
