// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "logging_utils.hpp"
#include "mock_logging_interface.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace exploy::control {

using ::testing::_;
using ::testing::StrictMock;

class LoggingTest : public ::testing::Test {
 protected:
  void TearDown() override {
    // Always reset to stdout so global state doesn't leak between tests.
    setLogger(nullptr);
  }

  StrictMock<MockLoggingInterface> logger_;
};

TEST_F(LoggingTest, SetAndGetLogger) {
  EXPECT_EQ(getLogger(), nullptr);
  setLogger(&logger_);
  EXPECT_EQ(getLogger(), &logger_);
}

TEST_F(LoggingTest, NullptrRestoresFallback) {
  setLogger(&logger_);
  setLogger(nullptr);
  EXPECT_EQ(getLogger(), nullptr);
}

TEST_F(LoggingTest, GenericLog_RoutesToLogger) {
  setLogger(&logger_);
  EXPECT_CALL(logger_, log(LoggingInterface::Level::Error, "hello world"));
  LOG(ERROR, "hello %s", "world");
}

TEST_F(LoggingTest, GenericLogStream_RoutesToLogger) {
  setLogger(&logger_);
  EXPECT_CALL(logger_, log(LoggingInterface::Level::Error, "hello world"));
  LOG_STREAM(ERROR, "hello " << "world");
}

TEST_F(LoggingTest, LevelMapping_Error) {
  setLogger(&logger_);
  EXPECT_CALL(logger_, log(LoggingInterface::Level::Error, _));
  LOG_STREAM(ERROR, "msg");
}

TEST_F(LoggingTest, LevelMapping_Warn) {
  setLogger(&logger_);
  EXPECT_CALL(logger_, log(LoggingInterface::Level::Warn, _));
  LOG_STREAM(WARN, "msg");
}

TEST_F(LoggingTest, LevelMapping_Warning) {
  setLogger(&logger_);
  EXPECT_CALL(logger_, log(LoggingInterface::Level::Warn, _));
  LOG_STREAM(WARNING, "msg");
}

TEST_F(LoggingTest, LevelMapping_Info) {
  setLogger(&logger_);
  EXPECT_CALL(logger_, log(LoggingInterface::Level::Info, _));
  LOG_STREAM(INFO, "msg");
}

TEST_F(LoggingTest, NoLoggerSet_DoesNotCrash) {
  // With no logger set, macros fall back to stdout — just verify no crash.
  EXPECT_EQ(getLogger(), nullptr);
  LOG(ERROR, "fallback %s", "output");
  LOG_STREAM(WARN, "fallback " << "stream");
}

TEST_F(LoggingTest, StdoutLogger_DoesNotCrash) {
  StdoutLogger stdout_logger;
  setLogger(&stdout_logger);
  // No expectation — just verify the concrete default implementation doesn't crash.
  // Exercise direct calls to the logger implementation.
  stdout_logger.log(LoggingInterface::Level::Error, "error message");
  stdout_logger.log(LoggingInterface::Level::Warn, "warn message");
  stdout_logger.log(LoggingInterface::Level::Info, "info message");
  // Also exercise routing through the global logger via the macros.
  LOG(ERROR, "stdout logger %s", "macro error");
  LOG_STREAM(WARN, "stdout logger " << "macro warn");
  LOG_STREAM(INFO, "stdout logger " << "macro info");
}

}  // namespace exploy::control
