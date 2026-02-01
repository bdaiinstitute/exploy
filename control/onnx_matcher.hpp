#pragma once

#include <optional>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx_components.hpp"

namespace rai::cs::control::common::onnx {

constexpr std::string_view alphanumeric = "[a-zA-Z0-9_]+";

struct Match {
  std::string name{};
  std::optional<std::string> metadata{};
};

struct GroupMatch {
  std::string name{};
  std::optional<std::string> metadata{};
  std::vector<Match> input_matches{};
  std::vector<Match> output_matches{};
};

class Matcher {
 public:
  virtual ~Matcher() = default;
  virtual bool matches(const Match& maybe_match) = 0;
  virtual std::vector<std::unique_ptr<Input>> createInputs() const { return {}; }
  virtual std::vector<std::unique_ptr<Output>> createOutputs() const { return {}; }

 protected:
  std::unordered_map<std::string, Match> found_matches_;
};

class GroupMatcher {
 public:
  virtual ~GroupMatcher() = default;
  virtual bool matches(const Match& maybe_match) = 0;
  virtual std::vector<std::unique_ptr<Input>> createInputs() const { return {}; }
  virtual std::vector<std::unique_ptr<Output>> createOutputs() const { return {}; }
  void populateGroupMetadata(
      std::function<std::optional<std::string>(const std::string&)> get_metadata) {
    for (auto& [name, group_match] : found_matches_) {
      group_match.metadata = get_metadata(name);
    }
  }

 protected:
  std::unordered_map<std::string, GroupMatch> found_matches_;
};

class IMUAngularVelocityMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class JointPositionMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class JointVelocityMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class BasePositionMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class BaseOrientationMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class BaseLinearVelocityMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class BaseAngularVelocityMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class JointTargetMatcher : public GroupMatcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Output>> createOutputs() const override;
};

class SE2VelocityMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Output>> createOutputs() const override;
};

class HeightScanMatcher : public GroupMatcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;

 private:
  const std::regex pattern_ =
      std::regex(fmt::format("(sensor\\.height_scanner\\.({}))\\.(height|r|g|b)", alphanumeric));
};

class RangeImageMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class DepthImageMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class BodyOrientationMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class SE3PoseMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class CommandSE2VelocityMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

class MemoryMatcher : public GroupMatcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Output>> createOutputs() const override;
};

class StepCountMatcher : public Matcher {
 public:
  bool matches(const Match& maybe_match) override;
  std::vector<std::unique_ptr<Input>> createInputs() const override;
};

}  // namespace rai::cs::control::common::onnx
