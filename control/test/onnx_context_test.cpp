// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "onnx_context.hpp"

#include "mock_command_interface.hpp"
#include "mock_state_interface.hpp"
#include "onnx_matcher.hpp"
#include "onnx_runtime.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

namespace rai::cs::control::common::onnx {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

// Mock Matcher for testing
class MockMatcher : public Matcher {
 public:
  MOCK_METHOD(bool, matches, (const Match& maybe_match), (override));
  MOCK_METHOD((std::vector<std::unique_ptr<Input>>), createInputs, (), (const, override));
  MOCK_METHOD((std::vector<std::unique_ptr<Output>>), createOutputs, (), (const, override));
};

// Mock GroupMatcher for testing
class MockGroupMatcher : public GroupMatcher {
 public:
  MOCK_METHOD(bool, matches, (const Match& maybe_match), (override));
  MOCK_METHOD((std::vector<std::unique_ptr<Input>>), createInputs, (), (const, override));
  MOCK_METHOD((std::vector<std::unique_ptr<Output>>), createOutputs, (), (const, override));
};

// Simple test matcher that always matches a specific pattern
class SimpleTestMatcher : public Matcher {
 public:
  explicit SimpleTestMatcher(std::string pattern) : pattern_(std::move(pattern)) {}

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
      // do nothing
    }
    return inputs;
  }

 private:
  std::string pattern_;
};

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
  EXPECT_TRUE(context_.createContext(runtime_));
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
  EXPECT_TRUE(context_.createContext(runtime_));
}

TEST_F(OnnxContextTest, CreateContext_CreatesInputsFromGroupMatchers) {
  auto mock_group_matcher = std::make_unique<NiceMock<MockGroupMatcher>>();
  auto* matcher_ptr = mock_group_matcher.get();
  ON_CALL(*matcher_ptr, matches(_)).WillByDefault(Return(true));
  EXPECT_CALL(*matcher_ptr, createInputs()).Times(1);
  EXPECT_CALL(*matcher_ptr, createOutputs()).Times(1);
  context_.registerGroupMatcher(std::move(mock_group_matcher));
  EXPECT_TRUE(context_.createContext(runtime_));
}

// ========== Integration Tests ==========

TEST_F(OnnxContextTest, Integration_RealMatchersWithTestModel) {
  context_.registerMatcher(std::make_unique<JointPositionMatcher>());
  context_.registerMatcher(std::make_unique<JointVelocityMatcher>());
  context_.registerMatcher(std::make_unique<BasePositionMatcher>());
  context_.registerMatcher(std::make_unique<BaseOrientationMatcher>());
  EXPECT_TRUE(context_.createContext(runtime_, false));
  EXPECT_GT(context_.getInputs().size(), 0);
  EXPECT_EQ(context_.updateRate(), 10);
}

TEST_F(OnnxContextTest, Integration_MixedMatchersAndGroupMatchers) {
  context_.registerMatcher(std::make_unique<JointPositionMatcher>());
  auto mock_group_matcher = std::make_unique<NiceMock<MockGroupMatcher>>();
  ON_CALL(*mock_group_matcher, matches(_)).WillByDefault(Return(false));
  context_.registerGroupMatcher(std::move(mock_group_matcher));
  EXPECT_TRUE(context_.createContext(runtime_, false));
}

}  // namespace rai::cs::control::common::onnx
