#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace rai::cs::control::common::onnx {

// Forward declarations
struct Input;
struct Output;

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

struct Matcher {
  virtual ~Matcher() = default;
  virtual bool matches(const std::string& name) = 0;
  virtual std::unique_ptr<Input> createInput(const Match& match) const { return nullptr; };
  virtual std::unique_ptr<Output> createOutput(const Match& match) const { return nullptr; };
};

struct GroupMatcher {
  virtual ~GroupMatcher() = default;
  virtual std::optional<std::string> matches(const std::string& name) = 0;
  virtual std::unique_ptr<Input> createInput(const GroupMatch& match) const { return nullptr; };
  virtual std::unique_ptr<Output> createOutput(const GroupMatch& match) const { return nullptr; };
};

class IMUAngularVelocityMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;

 private:
  std::string imu_name_;
};

class JointPositionMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;
};

class JointVelocityMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;
};

class BasePositionMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;
};

class BaseOrientationMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;
};

class BaseLinearVelocityMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;
};

class BaseAngularVelocityMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;
};

class JointTargetMatcher : public GroupMatcher {
 public:
  std::optional<std::string> matches(const std::string& name) override;
  std::unique_ptr<Output> createOutput(const GroupMatch& match) const override;
};

class SE2VelocityMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Output> createOutput(const Match& match) const override;
};

class HeightScanMatcher : public GroupMatcher {
 public:
  std::optional<std::string> matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const GroupMatch& match) const override;
};

class RangeImageMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;
};

class DepthImageMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;
};

class BodyOrientationMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;

 private:
  std::string body_name_;
};

class SE3PoseMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;

 private:
  std::string command_name_;
};

class CommandSE2VelocityMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;

 private:
  std::string command_name_;
};

class MemoryMatcher : public GroupMatcher {
 public:
  std::optional<std::string> matches(const std::string& name) override;
  std::unique_ptr<Output> createOutput(const GroupMatch& match) const override;
};

class StepCountMatcher : public Matcher {
 public:
  bool matches(const std::string& name) override;
  std::unique_ptr<Input> createInput(const Match& match) const override;
};

}  // namespace rai::cs::control::common::onnx
