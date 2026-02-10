// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "onnx_runtime.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <span>

namespace exploy::control {

class OnnxRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_data_dir_ = TEST_DATA_DIR;
    simple_model_path_ = test_data_dir_ + "/test_simple.onnx";
    nonexistent_model_path_ = test_data_dir_ + "/nonexistent.onnx";
  }

  std::string test_data_dir_;
  std::string simple_model_path_;
  std::string nonexistent_model_path_;
};

TEST_F(OnnxRuntimeTest, InitializeWithNonexistentFile) {
  OnnxRuntime runtime;
  ASSERT_FALSE(runtime.initialize(nonexistent_model_path_));
}

TEST_F(OnnxRuntimeTest, InitializeWithValidFile) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));
}

TEST_F(OnnxRuntimeTest, InitializeWithProfiling) {
  OnnxRuntime runtime;
  OnnxRuntimeOptions options{};
  options.profiling_path = "/tmp/onnx_profiling.json";
  ASSERT_TRUE(runtime.initialize(simple_model_path_, options));
}

// Note: we cannot actually test CUDA execution as CI does not have a GPU and CUDA installed. We
// therefore just test that initialization with the CUDA option is successful and fallback is
// working if there is no GPU available.
TEST_F(OnnxRuntimeTest, InitializeWithCuda) {
  OnnxRuntime runtime;
  OnnxRuntimeOptions options{};
  options.provider = OnnxRuntimeOptions::ExecutionProvider::CUDA;
  ASSERT_TRUE(runtime.initialize(simple_model_path_, options));
  ASSERT_TRUE(runtime.evaluate());
}

TEST_F(OnnxRuntimeTest, GetCustomMetadataSimpleModel) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Test existing metadata
  auto model_version = runtime.getCustomMetadata("model_version");
  ASSERT_TRUE(model_version.has_value());
  EXPECT_EQ(model_version.value(), "\"1.0\"");

  auto model_type = runtime.getCustomMetadata("model_type");
  ASSERT_TRUE(model_type.has_value());
  EXPECT_EQ(model_type.value(), "\"simple_test\"");

  // Test non-existent metadata
  auto nonexistent = runtime.getCustomMetadata("nonexistent_key");
  EXPECT_FALSE(nonexistent.has_value());
}

TEST_F(OnnxRuntimeTest, InputTensorNames) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  auto input_names = runtime.inputNames();
  EXPECT_GT(input_names.size(), 0);

  // Test expected input names from the simple model
  EXPECT_TRUE(input_names.contains("float_input"));
  EXPECT_TRUE(input_names.contains("int_input"));
  EXPECT_TRUE(input_names.contains("bool_input"));
}

TEST_F(OnnxRuntimeTest, OutputTensorNames) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  auto output_names = runtime.outputNames();
  EXPECT_GT(output_names.size(), 0);

  // Test expected output names from the simple model
  EXPECT_TRUE(output_names.contains("float_output"));
  EXPECT_TRUE(output_names.contains("int_output"));
  EXPECT_TRUE(output_names.contains("bool_output"));
}

TEST_F(OnnxRuntimeTest, InputBufferFloatType) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Test getting input buffer for existing tensor
  auto float_buffer = runtime.inputBuffer<float>("float_input");
  ASSERT_TRUE(float_buffer.has_value());
  EXPECT_EQ(float_buffer->size(), 3);  // Based on simple model shape (1, 3)

  // Test getting input buffer for non-existent tensor
  auto nonexistent_buffer = runtime.inputBuffer<float>("nonexistent");
  EXPECT_FALSE(nonexistent_buffer.has_value());
}

TEST_F(OnnxRuntimeTest, OutputBufferFloatType) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Test getting output buffer for existing tensor
  auto float_output_buffer = runtime.outputBuffer<float>("float_output");
  ASSERT_TRUE(float_output_buffer.has_value());
  EXPECT_EQ(float_output_buffer->size(), 3);  // Based on simple model output shape (1, 3)

  // Test getting output buffer for non-existent tensor
  auto nonexistent_buffer = runtime.outputBuffer<float>("nonexistent");
  EXPECT_FALSE(nonexistent_buffer.has_value());
}

TEST_F(OnnxRuntimeTest, InputBufferWrongType) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Test getting input buffer with wrong type (float_input is float, asking for int32_t)
  auto wrong_type_buffer = runtime.inputBuffer<int32_t>("float_input");
  EXPECT_FALSE(wrong_type_buffer.has_value());
}

TEST_F(OnnxRuntimeTest, ResetBuffers) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Get input and output buffers and set some non-zero values
  auto float_buffer = runtime.inputBuffer<float>("float_input");
  ASSERT_TRUE(float_buffer.has_value());
  float_buffer.value()[0] = 1.5f;
  float_buffer.value()[1] = 2.5f;

  auto float_output_buffer = runtime.outputBuffer<float>("float_output");
  ASSERT_TRUE(float_output_buffer.has_value());
  float_output_buffer.value()[0] = 3.5f;

  // Reset buffers
  runtime.resetBuffers();

  // Verify buffers are zeroed
  EXPECT_EQ(float_buffer.value()[0], 0.0f);
  EXPECT_EQ(float_buffer.value()[1], 0.0f);
  EXPECT_EQ(float_output_buffer.value()[0], 0.0f);
}

TEST_F(OnnxRuntimeTest, BufferDataPersistence) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Set some input values
  auto float_buffer = runtime.inputBuffer<float>("float_input");
  ASSERT_TRUE(float_buffer.has_value());
  float_buffer.value()[0] = 1.234f;
  float_buffer.value()[1] = 5.678f;

  // Get buffer again and verify data persists
  auto float_buffer2 = runtime.inputBuffer<float>("float_input");
  ASSERT_TRUE(float_buffer2.has_value());
  EXPECT_FLOAT_EQ(float_buffer2.value()[0], 1.234f);
  EXPECT_FLOAT_EQ(float_buffer2.value()[1], 5.678f);

  // Verify they're the same underlying buffer
  EXPECT_FLOAT_EQ(float_buffer2.value()[0], float_buffer.value()[0]);
}

TEST_F(OnnxRuntimeTest, SimpleModelWithDifferentTensorTypes) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Test float tensor
  auto float_buffer = runtime.inputBuffer<float>("float_input");
  ASSERT_TRUE(float_buffer.has_value());
  EXPECT_EQ(float_buffer->size(), 3);
  float_buffer.value()[0] = 1.5f;
  float_buffer.value()[1] = 2.5f;
  float_buffer.value()[2] = 3.5f;

  // Test int32 tensor
  auto int_buffer = runtime.inputBuffer<int32_t>("int_input");
  ASSERT_TRUE(int_buffer.has_value());
  EXPECT_EQ(int_buffer->size(), 3);
  int_buffer.value()[0] = 10;
  int_buffer.value()[1] = 20;
  int_buffer.value()[2] = 30;

  // Test bool tensor
  auto bool_buffer = runtime.inputBuffer<bool>("bool_input");
  ASSERT_TRUE(bool_buffer.has_value());
  EXPECT_EQ(bool_buffer->size(), 3);
  bool_buffer.value()[0] = true;
  bool_buffer.value()[1] = false;
  bool_buffer.value()[2] = true;

  // Run evaluation
  ASSERT_TRUE(runtime.evaluate());

  // Check float output (should be input * 2.0)
  auto float_output = runtime.outputBuffer<float>("float_output");
  ASSERT_TRUE(float_output.has_value());
  EXPECT_FLOAT_EQ(float_output.value()[0], 3.0f);  // 1.5 * 2
  EXPECT_FLOAT_EQ(float_output.value()[1], 5.0f);  // 2.5 * 2
  EXPECT_FLOAT_EQ(float_output.value()[2], 7.0f);  // 3.5 * 2

  // Check int output (should be input + 1)
  auto int_output = runtime.outputBuffer<int32_t>("int_output");
  ASSERT_TRUE(int_output.has_value());
  EXPECT_EQ(int_output.value()[0], 11);  // 10 + 1
  EXPECT_EQ(int_output.value()[1], 21);  // 20 + 1
  EXPECT_EQ(int_output.value()[2], 31);  // 30 + 1

  // Check bool output (should be logical NOT of input)
  auto bool_output = runtime.outputBuffer<bool>("bool_output");
  ASSERT_TRUE(bool_output.has_value());
  EXPECT_EQ(bool_output.value()[0], false);  // NOT true
  EXPECT_EQ(bool_output.value()[1], true);   // NOT false
  EXPECT_EQ(bool_output.value()[2], false);  // NOT true
}

TEST_F(OnnxRuntimeTest, WrongTypeAccess) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Try to access float tensor as int32 - should fail
  auto wrong_int_buffer = runtime.inputBuffer<int32_t>("float_input");
  EXPECT_FALSE(wrong_int_buffer.has_value());

  // Try to access int32 tensor as float - should fail
  auto wrong_float_buffer = runtime.inputBuffer<float>("int_input");
  EXPECT_FALSE(wrong_float_buffer.has_value());

  // Try to access bool tensor as float - should fail
  auto wrong_bool_buffer = runtime.inputBuffer<float>("bool_input");
  EXPECT_FALSE(wrong_bool_buffer.has_value());
}

TEST_F(OnnxRuntimeTest, ResetBuffersDifferentTypes) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Set values in different tensor types
  auto float_buffer = runtime.inputBuffer<float>("float_input");
  ASSERT_TRUE(float_buffer.has_value());
  float_buffer.value()[0] = 42.5f;

  auto int_buffer = runtime.inputBuffer<int32_t>("int_input");
  ASSERT_TRUE(int_buffer.has_value());
  int_buffer.value()[0] = 99;

  auto bool_buffer = runtime.inputBuffer<bool>("bool_input");
  ASSERT_TRUE(bool_buffer.has_value());
  bool_buffer.value()[0] = true;

  // Reset all buffers
  runtime.resetBuffers();

  // Verify all buffers are reset to their default values
  EXPECT_EQ(float_buffer.value()[0], 0.0f);
  EXPECT_EQ(int_buffer.value()[0], 0);
  EXPECT_EQ(bool_buffer.value()[0], false);
}

TEST_F(OnnxRuntimeTest, CopyOutputToInputFloatType) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Set input values and run evaluation to get output
  auto float_input = runtime.inputBuffer<float>("float_input");
  ASSERT_TRUE(float_input.has_value());
  float_input.value()[0] = 1.5f;
  float_input.value()[1] = 2.5f;
  float_input.value()[2] = 3.5f;

  ASSERT_TRUE(runtime.evaluate());

  // Get the output values (should be input * 2.0)
  auto float_output = runtime.outputBuffer<float>("float_output");
  ASSERT_TRUE(float_output.has_value());
  EXPECT_FLOAT_EQ(float_output.value()[0], 3.0f);  // 1.5 * 2
  EXPECT_FLOAT_EQ(float_output.value()[1], 5.0f);  // 2.5 * 2
  EXPECT_FLOAT_EQ(float_output.value()[2], 7.0f);  // 3.5 * 2

  // Reset input to different values
  float_input.value()[0] = 0.0f;
  float_input.value()[1] = 0.0f;
  float_input.value()[2] = 0.0f;

  // Copy output to input
  ASSERT_TRUE(runtime.copyOutputToInput("float_output", "float_input"));

  // Verify input now contains the output values
  EXPECT_FLOAT_EQ(float_input.value()[0], 3.0f);
  EXPECT_FLOAT_EQ(float_input.value()[1], 5.0f);
  EXPECT_FLOAT_EQ(float_input.value()[2], 7.0f);
}

TEST_F(OnnxRuntimeTest, CopyOutputToInputIntType) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Set input values and run evaluation to get output
  auto int_input = runtime.inputBuffer<int32_t>("int_input");
  ASSERT_TRUE(int_input.has_value());
  int_input.value()[0] = 10;
  int_input.value()[1] = 20;
  int_input.value()[2] = 30;

  ASSERT_TRUE(runtime.evaluate());

  // Get the output values (should be input + 1)
  auto int_output = runtime.outputBuffer<int32_t>("int_output");
  ASSERT_TRUE(int_output.has_value());
  EXPECT_EQ(int_output.value()[0], 11);  // 10 + 1
  EXPECT_EQ(int_output.value()[1], 21);  // 20 + 1
  EXPECT_EQ(int_output.value()[2], 31);  // 30 + 1

  // Reset input to different values
  int_input.value()[0] = 0;
  int_input.value()[1] = 0;
  int_input.value()[2] = 0;

  // Copy output to input
  ASSERT_TRUE(runtime.copyOutputToInput("int_output", "int_input"));

  // Verify input now contains the output values
  EXPECT_EQ(int_input.value()[0], 11);
  EXPECT_EQ(int_input.value()[1], 21);
  EXPECT_EQ(int_input.value()[2], 31);
}

TEST_F(OnnxRuntimeTest, CopyOutputToInputBoolType) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Set input values and run evaluation to get output
  auto bool_input = runtime.inputBuffer<bool>("bool_input");
  ASSERT_TRUE(bool_input.has_value());
  bool_input.value()[0] = true;
  bool_input.value()[1] = false;
  bool_input.value()[2] = true;

  ASSERT_TRUE(runtime.evaluate());

  // Get the output values (should be logical NOT of input)
  auto bool_output = runtime.outputBuffer<bool>("bool_output");
  ASSERT_TRUE(bool_output.has_value());
  EXPECT_EQ(bool_output.value()[0], false);  // NOT true
  EXPECT_EQ(bool_output.value()[1], true);   // NOT false
  EXPECT_EQ(bool_output.value()[2], false);  // NOT true

  // Reset input to different values
  bool_input.value()[0] = false;
  bool_input.value()[1] = false;
  bool_input.value()[2] = false;

  // Copy output to input
  ASSERT_TRUE(runtime.copyOutputToInput("bool_output", "bool_input"));

  // Verify input now contains the output values
  EXPECT_EQ(bool_input.value()[0], false);
  EXPECT_EQ(bool_input.value()[1], true);
  EXPECT_EQ(bool_input.value()[2], false);
}

TEST_F(OnnxRuntimeTest, CopyOutputToInputNonexistentTensors) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Test copying from nonexistent output
  EXPECT_FALSE(runtime.copyOutputToInput("float_output", "nonexistent_input"));

  // Test copying to nonexistent input
  EXPECT_FALSE(runtime.copyOutputToInput("nonexistent_output", "float_input"));

  // Test copying both nonexistent
  EXPECT_FALSE(runtime.copyOutputToInput("nonexistent_output", "nonexistent_input"));
}

TEST_F(OnnxRuntimeTest, CopyOutputToInputTypeMismatch) {
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.initialize(simple_model_path_));

  // Test copying float output to int input (should fail due to type mismatch)
  EXPECT_FALSE(runtime.copyOutputToInput("int_output", "float_input"));

  // Test copying int output to float input (should fail due to type mismatch)
  EXPECT_FALSE(runtime.copyOutputToInput("float_output", "int_input"));

  // Test copying bool output to float input (should fail due to type mismatch)
  EXPECT_FALSE(runtime.copyOutputToInput("float_output", "bool_input"));
}

}  // namespace exploy::control
