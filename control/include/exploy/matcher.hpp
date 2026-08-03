// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <functional>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "exploy/components.hpp"

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
  explicit Matcher(const std::string& name) : name_(name) {}
  virtual ~Matcher() = default;

  /**
   * @brief Check if a tensor matches this matcher's pattern.
   *
   * @param maybe_match Potential match containing tensor name and metadata.
   * @return true if the tensor matches and was stored, false otherwise.
   */
  virtual bool matches(const Match& /*maybe_match*/) { return false; }

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

  /**
   * @brief Create read-only observer components from all matched tensors.
   *
   * A matcher that creates observers (and no inputs/outputs) is a read-only observer: the
   * tensors it matches are not counted as functionally owned, so it may observe tensors that
   * are already owned by another matcher.
   *
   * @return Vector of observer component unique pointers.
   */
  virtual std::vector<std::unique_ptr<Observer>> createObservers() const { return {}; }

  /**
   * @brief Reset the matcher by clearing all found matches and any subclass state.
   *
   * Non-virtual: clears the base `found_matches_` storage and then dispatches to the
   * virtual `reset()` hook.
   */
  void resetMatcher() {
    found_matches_.clear();
    reset();
  }

  /**
   * @brief Get the name of this matcher.
   *
   * @return Name of the matcher.
   */
  const std::string& getName() const { return name_; }

 protected:
  /**
   * @brief Hook for subclasses to clear any additional state during a reset.
   *
   * Called by `resetMatcher()` after `found_matches_` has been cleared. Default is
   * a no-op; override only if the subclass stores extra side maps.
   */
  virtual void reset() {}

  std::unordered_map<std::string, Match> found_matches_;  ///< Storage for matched tensors.

 private:
  std::string name_;  ///< Name of this matcher.
};

/**
 * @brief Abstract base class for multi-tensor group pattern matching.
 *
 * GroupMatcher implementations recognize patterns where multiple related tensors
 * need to be processed together (e.g., joints.pos and joints.vel).
 */
class GroupMatcher {
 public:
  explicit GroupMatcher(const std::string& name) : name_(name) {}
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
   * @brief Create read-only observer components from all matched tensor groups.
   *
   * A group matcher that creates observers (and no inputs/outputs) is a read-only observer:
   * the tensors it matches are not counted as functionally owned, so it may observe tensors
   * that are already owned by another matcher.
   *
   * @return Vector of observer component unique pointers.
   */
  virtual std::vector<std::unique_ptr<Observer>> createObservers() const { return {}; }

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

  /**
   * @brief Reset the matcher by clearing all found matches and any subclass state.
   *
   * Non-virtual: clears the base `found_matches_` storage and then dispatches to the
   * virtual `reset()` hook.
   */
  void resetMatcher() {
    found_matches_.clear();
    reset();
  }

  /**
   * @brief Get the name of this matcher.
   *
   * @return Name of the matcher.
   */
  const std::string& getName() const { return name_; }

 protected:
  /**
   * @brief Hook for subclasses to clear any additional state during a reset.
   *
   * Called by `resetMatcher()` after `found_matches_` has been cleared. Default is
   * a no-op; override only if the subclass stores extra side maps.
   */
  virtual void reset() {}

  std::unordered_map<std::string, GroupMatch> found_matches_;  ///< Storage for matched groups.

 private:
  std::string name_;  ///< Name of this matcher.
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
  JointMatcher();
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;

 private:
  void reset() override;
  const std::regex pattern_ =
      std::regex(fmt::format("(obj\\.({})\\.joints)\\.(pos|vel)", kAlphanumeric));
  /// Map from group name to the resolved articulation name.
  std::unordered_map<std::string, std::string> articulation_names_;
  /// Map from input tensor name to the resolved type ("pos" or "vel").
  std::unordered_map<std::string, std::string> input_types_;
};
// ---------------------------------------------------------------

// ---------------  Base matchers --------------------------------
/**
 * @brief Common base class for matchers operating on `obj.<articulation>.<base>.<field>`
 * tensors.
 *
 * Stores the resolved articulation name for each matched tensor at match time so
 * createInputs() can look it up directly without re-resolving against the regex.
 */
class BaseMatcher : public Matcher {
 public:
  BaseMatcher(const std::string& name, std::string field);
  bool matches(const Match& maybe_match) override;

 protected:
  std::string field_;  ///< Field suffix (e.g. "pos_b_rt_w_in_w") this matcher targets.
  /// Map from matched tensor name to the resolved articulation name.
  std::unordered_map<std::string, std::string> articulation_names_;

 private:
  void reset() override;
};

/**
 * @brief Matcher for robot base position input tensors.
 *
 * Matches patterns for base position in world frame and creates BasePositionInput components.
 */
class BasePositionMatcher : public BaseMatcher {
 public:
  BasePositionMatcher();
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for robot base orientation input tensors.
 *
 * Matches patterns for base orientation (quaternion) in world frame and creates
 * BaseOrientationInput components.
 */
class BaseOrientationMatcher : public BaseMatcher {
 public:
  BaseOrientationMatcher();
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for robot base linear velocity input tensors.
 *
 * Matches patterns for base linear velocity and creates BaseLinearVelocityInput components.
 */
class BaseLinearVelocityMatcher : public BaseMatcher {
 public:
  BaseLinearVelocityMatcher();
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for robot base angular velocity input tensors.
 *
 * Matches patterns for base angular velocity and creates BaseAngularVelocityInput components.
 */
class BaseAngularVelocityMatcher : public BaseMatcher {
 public:
  BaseAngularVelocityMatcher();
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
  JointTargetMatcher();
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Output>> createOutputs() const override;

 private:
  void reset() override;
  const std::regex pattern_ =
      std::regex(fmt::format("(output\\.joint_targets\\.({}))\\.(pos|vel|effort)", kAlphanumeric));
  /// Map from group name to the resolved articulation name.
  std::unordered_map<std::string, std::string> articulation_names_;
};

/**
 * @brief Matcher for SE(2) velocity output tensors.
 *
 * Matches patterns for planar velocity commands and creates SE2VelocityOutput components.
 */
class SE2VelocityMatcher : public Matcher {
 public:
  SE2VelocityMatcher();
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Output>> createOutputs() const override;
};
// ---------------------------------------------------------------

// ---------------  Sensor matchers ------------------------------
/**
 * @brief Common base class for sensor group matchers that key on
 * `sensor.<kind>.<sensor_name>.<channel>`.
 *
 * Stores the resolved sensor name and the set of channel/layer names per group at
 * match time so createInputs() can read them directly without re-running the regex.
 */
class SensorGroupMatcher : public GroupMatcher {
 public:
  SensorGroupMatcher(const std::string& name, std::regex pattern);
  bool matches(const Match& maybe_match) override;

 protected:
  std::regex pattern_;  ///< Sensor pattern with capture groups: (group)(sensor_name)(channel).
  /// Map from group name to the resolved sensor name.
  std::unordered_map<std::string, std::string> sensor_names_;
  /// Map from group name to the set of resolved channel/layer names.
  std::unordered_map<std::string, std::unordered_set<std::string>> channel_names_;

 private:
  void reset() override;
};

/**
 * @brief Matcher for height scan sensor input tensors.
 *
 * Matches patterns for terrain height scan layers (height, r, g, b) and creates
 * HeightScanInput components with multi-layer support.
 */
class HeightScanMatcher : public SensorGroupMatcher {
 public:
  HeightScanMatcher();
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for spherical image sensor input tensors.
 *
 * Matches patterns for spherical image channels (e.g., range, risk) and creates
 * SphericalImageInput components with multi-channel support.
 */
class SphericalImageMatcher : public SensorGroupMatcher {
 public:
  SphericalImageMatcher();
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for pinhole camera image input tensors.
 *
 * Matches patterns for pinhole image channels (e.g., depth, risk) and creates
 * PinholeImageInput components with multi-channel support.
 */
class PinholeImageMatcher : public SensorGroupMatcher {
 public:
  PinholeImageMatcher();
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for IMU angular velocity input tensors.
 *
 * Matches patterns for IMU angular velocity data and creates IMUAngularVelocityInput components.
 */
class IMUAngularVelocityMatcher : public Matcher {
 public:
  IMUAngularVelocityMatcher();
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for IMU linear velocity input tensors.
 *
 * Matches patterns like "sensor.imu.{imu_name}.lin_vel_b_rt_w_in_b" and creates
 * IMULinearVelocityInput components.
 */
class IMULinearVelocityMatcher : public Matcher {
 public:
  IMULinearVelocityMatcher();
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
  IMUOrientationMatcher();
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
  CommandSE3PoseMatcher();
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
  CommandSE2VelocityMatcher();
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
  CommandBooleanMatcher();
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
  CommandFloatMatcher();
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for joint position command input tensors.
 *
 * Matches patterns like "cmd.joint_pos.{name}" and creates CommandJointPositionInput
 * components. The joint names are read from the tensor's JSON metadata field
 * ("joint_names" array).
 */
class CommandJointPositionMatcher : public Matcher {
 public:
  CommandJointPositionMatcher();
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};
// ---------------------------------------------------------------

// ---------------  Body matchers --------------------------------
/**
 * @brief Common base class for matchers operating on `obj.<articulation>.<body>.<field>`
 * tensors that are NOT registered base pairs.
 *
 * Stores the resolved (articulation, body) pair for each matched tensor at match time so
 * createInputs() can read it directly.
 */
class BodyMatcher : public Matcher {
 public:
  BodyMatcher(const std::string& name, std::string field);
  bool matches(const Match& maybe_match) override;

 protected:
  std::string field_;  ///< Field suffix (e.g. "pos_b_rt_w_in_w") this matcher targets.
  /// Map from matched tensor name to the resolved (articulation, body) pair.
  std::unordered_map<std::string, std::pair<std::string, std::string>> articulation_and_body_;

 private:
  void reset() override;
};

/**
 * @brief Matcher for rigid body position input tensors.
 *
 * Matches patterns for body position data and creates BodyPositionInput components.
 */
class BodyPositionMatcher : public BodyMatcher {
 public:
  BodyPositionMatcher();
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for rigid body orientation input tensors.
 *
 * Matches patterns for body orientation (quaternion) data and creates BodyOrientationInput
 * components.
 */
class BodyOrientationMatcher : public BodyMatcher {
 public:
  BodyOrientationMatcher();
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for rigid body linear velocity input tensors.
 *
 * Matches patterns for body linear velocity data and creates BodyLinearVelocityInput
 * components.
 */
class BodyLinearVelocityMatcher : public BodyMatcher {
 public:
  BodyLinearVelocityMatcher();
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

/**
 * @brief Matcher for rigid body angular velocity input tensors.
 *
 * Matches patterns for body angular velocity data and creates BodyAngularVelocityInput
 * components.
 */
class BodyAngularVelocityMatcher : public BodyMatcher {
 public:
  BodyAngularVelocityMatcher();
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
  MemoryMatcher();
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
  StepCountMatcher();
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};
// ---------------------------------------------------------------

}  // namespace exploy::control
