// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "exploy/context.hpp"
#include "exploy/matcher.hpp"
#include "exploy/onnx_runtime.hpp"
#include "mock_command_interface.hpp"
#include "mock_state_interface.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

namespace exploy::control {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

// Mock Matcher for testing
class MockMatcher : public Matcher {
 public:
  MockMatcher() : Matcher("MockMatcher") {}
  MOCK_METHOD(bool, matches, (const Match& maybe_match), (override));
  MOCK_METHOD((std::vector<std::unique_ptr<Input>>), createInputs, (), (const, override));
  MOCK_METHOD((std::vector<std::unique_ptr<Output>>), createOutputs, (), (const, override));
};

// Mock GroupMatcher for testing
class MockGroupMatcher : public GroupMatcher {
 public:
  MockGroupMatcher() : GroupMatcher("MockGroupMatcher") {}
  MOCK_METHOD(bool, matches, (const Match& maybe_match), (override));
  MOCK_METHOD((std::vector<std::unique_ptr<Input>>), createInputs, (), (const, override));
  MOCK_METHOD((std::vector<std::unique_ptr<Output>>), createOutputs, (), (const, override));
};

// Simple test matcher that always matches a specific pattern
class SimpleTestMatcher : public Matcher {
 public:
  explicit SimpleTestMatcher(std::string pattern)
      : Matcher("SimpleTestMatcher"), pattern_(std::move(pattern)) {}

  bool matches(const Match& maybe_match) override {
    if (maybe_match.name.find(pattern_) != std::string::npos) {
      found_matches_[maybe_match.name] = maybe_match;
      return true;
    }
    return false;
  }

  std::vector<std::unique_ptr<Input>> createInputs() const override {
    std::vector<std::unique_ptr<Input>> inputs;
    for (const auto& [name, match] : found_matches_) {
      (void)name;
      (void)match;
      // do nothing
    }
    return inputs;
  }

 private:
  std::string pattern_;
};

// Read-only observer that records nothing but proves observers can be created and run.
class NoopObserver : public Observer {
 public:
  explicit NoopObserver(std::string tensor_name)
      : Observer("NoopObserver"), tensor_name_(std::move(tensor_name)) {}

  void observe(const OnnxRuntime& runtime) const override {
    // Read-only access only; must compile with the const buffer accessors.
    (void)runtime.inputBuffer<float>(tensor_name_);
  }

 private:
  std::string tensor_name_;
};

// Minimal input component that reports the ONNX tensor it serves for ownership checks.
class TestInput : public Input {
 public:
  explicit TestInput(std::string tensor_name) : Input("TestInput", {std::move(tensor_name)}) {}

  bool read(OnnxRuntime& /*runtime*/, RobotStateInterface& /*state*/,
            CommandInterface& /*command*/) override {
    return true;
  }
};

// Functional matcher that creates one input component per tensor whose name contains a substring.
class SingleInputMatcher : public Matcher {
 public:
  explicit SingleInputMatcher(std::string pattern)
      : Matcher("SingleInputMatcher"), pattern_(std::move(pattern)) {}

  bool matches(const Match& maybe_match) override {
    if (maybe_match.name.find(pattern_) != std::string::npos) {
      found_matches_[maybe_match.name] = maybe_match;
      return true;
    }
    return false;
  }

  std::vector<std::unique_ptr<Input>> createInputs() const override {
    std::vector<std::unique_ptr<Input>> inputs;
    for (const auto& [name, match] : found_matches_) {
      (void)match;
      inputs.push_back(std::make_unique<TestInput>(name));
    }
    return inputs;
  }

 private:
  std::string pattern_;
};

// Observer-only matcher: creates no input/output components, only observers. It records tensors
// whose name contains a substring and creates a read-only observer for each.
class TestObserverMatcher : public Matcher {
 public:
  explicit TestObserverMatcher(std::string pattern)
      : Matcher("TestObserverMatcher"), pattern_(std::move(pattern)) {}

  bool matches(const Match& maybe_match) override {
    if (maybe_match.name.find(pattern_) != std::string::npos) {
      observed_.push_back(maybe_match.name);
      return true;
    }
    return false;
  }

  std::vector<std::unique_ptr<Observer>> createObservers() const override {
    std::vector<std::unique_ptr<Observer>> observers;
    for (const auto& name : observed_) {
      observers.push_back(std::make_unique<NoopObserver>(name));
    }
    return observers;
  }

 private:
  void reset() override { observed_.clear(); }

  std::string pattern_;
  std::vector<std::string> observed_;
};

// Test fixture for OnnxContext tests
class OnnxContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Load the test ONNX model
    std::string test_model_path =
        (std::filesystem::path(TEST_DATA_DIR) / "test_export.onnx").string();
    ASSERT_TRUE(runtime_.initialize(test_model_path));
  }

  OnnxContext context_;
  OnnxRuntime runtime_;
  NiceMock<MockRobotStateInterface> state_mock_;
  NiceMock<MockCommandInterface> command_mock_;
};

// ========== Registration Tests ==========

TEST_F(OnnxContextTest, RegisterMatcher_MultipleMatchers) {
  context_.registerMatcher(std::make_unique<SimpleTestMatcher>("joint"));
  context_.registerMatcher(std::make_unique<SimpleTestMatcher>("base"));
  context_.registerMatcher(std::make_unique<SimpleTestMatcher>("sensor"));
  EXPECT_TRUE(context_.createContext(runtime_, false));
}

TEST_F(OnnxContextTest, RegisterGroupMatcher_MultipleGroupMatchers) {
  auto group_matcher1 = std::make_unique<NiceMock<MockGroupMatcher>>();
  auto group_matcher2 = std::make_unique<NiceMock<MockGroupMatcher>>();
  ON_CALL(*group_matcher1, matches(_)).WillByDefault(Return(true));
  ON_CALL(*group_matcher2, matches(_)).WillByDefault(Return(false));
  context_.registerGroupMatcher(std::move(group_matcher1));
  context_.registerGroupMatcher(std::move(group_matcher2));
  EXPECT_TRUE(context_.createContext(runtime_, false));
}

// ========== Context Creation Tests ==========

TEST_F(OnnxContextTest, CreateContext_ParsesUpdateRate) {
  context_.registerMatcher(std::make_unique<SimpleTestMatcher>("joint"));
  bool result = context_.createContext(runtime_, false);
  EXPECT_TRUE(result);
  EXPECT_EQ(context_.updateRate(), 10);
}

TEST_F(OnnxContextTest, CreateContext_CreatesInputsFromMatchers) {
  auto mock_matcher = std::make_unique<NiceMock<MockMatcher>>();
  auto* matcher_ptr = mock_matcher.get();
  ON_CALL(*matcher_ptr, matches(_)).WillByDefault(Return(true));
  EXPECT_CALL(*matcher_ptr, createInputs()).Times(1);
  EXPECT_CALL(*matcher_ptr, createOutputs()).Times(1);
  context_.registerMatcher(std::move(mock_matcher));
  EXPECT_TRUE(context_.createContext(runtime_, false));
}

TEST_F(OnnxContextTest, CreateContext_CreatesInputsFromGroupMatchers) {
  auto mock_group_matcher = std::make_unique<NiceMock<MockGroupMatcher>>();
  auto* matcher_ptr = mock_group_matcher.get();
  ON_CALL(*matcher_ptr, matches(_)).WillByDefault(Return(true));
  EXPECT_CALL(*matcher_ptr, createInputs()).Times(1);
  EXPECT_CALL(*matcher_ptr, createOutputs()).Times(1);
  context_.registerGroupMatcher(std::move(mock_group_matcher));
  EXPECT_TRUE(context_.createContext(runtime_, false));
}

// ========== Integration Tests ==========

TEST_F(OnnxContextTest, CreateContext_ErrorOnMultipleMatchersForSameTensor) {
  // Two matchers each create an input component for the same tensor(s), so at least one tensor
  // ends up served by two components, which must fail context creation.
  context_.registerMatcher(std::make_unique<SingleInputMatcher>("joint"));
  context_.registerMatcher(std::make_unique<SingleInputMatcher>("joint"));
  EXPECT_FALSE(context_.createContext(runtime_, false));
}

TEST_F(OnnxContextTest, Integration_RealMatchersWithTestModel) {
  context_.registerGroupMatcher(std::make_unique<JointMatcher>());
  context_.registerMatcher(std::make_unique<BasePositionMatcher>());
  context_.registerMatcher(std::make_unique<BaseOrientationMatcher>());
  EXPECT_TRUE(context_.createContext(runtime_, false));
  EXPECT_GT(context_.getInputs().size(), 0);
  EXPECT_EQ(context_.updateRate(), 10);
}

// ========== Observer Tests ==========

TEST_F(OnnxContextTest, CreateContext_ObserverCoexistsWithFunctionalMatcher) {
  // A functional matcher claims "joint" tensors, and an observer matcher observes the SAME
  // tensors. Observation must not participate in the one-component-per-tensor rule, so context
  // creation succeeds and observers are created.
  context_.registerMatcher(std::make_unique<SingleInputMatcher>("joint"));
  context_.registerMatcher(std::make_unique<TestObserverMatcher>("joint"));
  EXPECT_TRUE(context_.createContext(runtime_, false));
  EXPECT_GT(context_.getObservers().size(), 0);
}

TEST_F(OnnxContextTest, CreateContext_ObserverDoesNotCountAsFunctionalMatch) {
  // Only an observer matcher is registered. Observers are created, but they do not create
  // input/output components.
  context_.registerMatcher(std::make_unique<TestObserverMatcher>("joint"));
  EXPECT_TRUE(context_.createContext(runtime_, false));
  EXPECT_GT(context_.getObservers().size(), 0);
  EXPECT_EQ(context_.getInputs().size(), 0);
  EXPECT_EQ(context_.getOutputs().size(), 0);
}

TEST_F(OnnxContextTest, CreateContext_ObserversClearedOnRecreate) {
  context_.registerMatcher(std::make_unique<TestObserverMatcher>("joint"));
  ASSERT_TRUE(context_.createContext(runtime_, false));
  const std::size_t first_count = context_.getObservers().size();
  ASSERT_GT(first_count, 0);
  // Re-creating the context must rebuild (not accumulate) observers.
  ASSERT_TRUE(context_.createContext(runtime_, false));
  EXPECT_EQ(context_.getObservers().size(), first_count);
}

}  // namespace exploy::control
