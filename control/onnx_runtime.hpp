// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#pragma once

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rai::cs::control::common::onnx {

template <typename T>
struct onnx_type {
  static constexpr ONNXTensorElementDataType value = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};

template <>
struct onnx_type<float> {
  static constexpr ONNXTensorElementDataType value = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

template <>
struct onnx_type<int32_t> {
  static constexpr ONNXTensorElementDataType value = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
};

template <>
struct onnx_type<bool> {
  static constexpr ONNXTensorElementDataType value = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
};

/**
 * @brief Configuration options for initializing the ONNX Runtime.
 *
 * This struct encapsulates all configuration parameters needed to initialize an ONNX model.
 */
struct OnnxRuntimeOptions {
  enum class ExecutionProvider {
    CPU,
    CUDA,
  };

  // Execution provider to use (default: CPU).
  ExecutionProvider provider = ExecutionProvider::CPU;
  // Optional path to save profiling data. If not set, profiling is disabled.
  std::optional<std::string> profiling_path = std::nullopt;
};

/**
 * @brief A class for evaluating ONNX models using ONNX Runtime.
 *
 * The `OnnxRuntime` class manages the initialization, configuration, and inference execution of an
 * ONNX model.
 *
 */
class OnnxRuntime {
 public:
  /**
   * @brief Initializes the ONNX model and configures input/output tensors.
   *
   * Loads the ONNX model from the specified file path, sets up session options, and prepares memory
   * information. If the model file does not exist, the initialization will fail.
   *
   * @param model_path The file path to the ONNX model.
   * @param options The runtime options.
   * @return True if initialization is successful, false otherwise.
   */
  bool initialize(const std::string& model_path, const OnnxRuntimeOptions& options = {});

  /**
   * @brief Runs inference on the ONNX model with the buffered input data.
   *
   * This method runs inference and updates the output buffers.
   *
   * @return True evaluation succeeds, false if validation fails.
   *
   */
  bool evaluate();

  /**
   * @brief Getter for custom metadata from ONNX model
   *
   * @param key The key of the custom metadata stored in the ONNX model.
   * @return A string with the corresponding custom metadata, if the key exists, nullopt otherwise.
   *
   */
  std::optional<std::string> getCustomMetadata(const std::string& key) const;

  /**
   * @brief Retrieves a mutable span to the input tensor buffer of the specified name.
   *
   * @tparam T The data type of the tensor elements (e.g., float, int32_t, bool).
   * @param name The name of the input tensor.
   * @return An optional span to the tensor buffer if it exists and matches the requested type,
   * nullopt otherwise.
   *
   */
  template <typename T>
  inline std::optional<std::span<T>> inputBuffer(const std::string& name) {
    if (!input_names_to_index_.contains(name)) return std::nullopt;
    auto index = input_names_to_index_.at(name);
    return getBuffer<T>(input_.tensors[index], input_.data_types[index]);
  }

  /**
   * @brief Retrieves a mutable span to the output tensor buffer of the specified name.
   *
   * @tparam T The data type of the tensor elements (e.g., float, int32_t, bool).
   * @param name The name of the output tensor.
   * @return An optional span to the tensor buffer if it exists and matches the requested type,
   * nullopt otherwise.
   *
   */
  template <typename T>
  inline std::optional<std::span<T>> outputBuffer(const std::string& name) {
    if (!output_names_to_index_.contains(name)) return std::nullopt;
    auto index = output_names_to_index_.at(name);
    return getBuffer<T>(output_.tensors[index], output_.data_types[index]);
  }

  /**
   * @brief Resets all input and output buffers to zero.
   */
  void resetBuffers();

  /**
   * @brief Retrieves the set of input tensor names.
   *
   * @return An unordered set containing the names of all input tensors.
   */
  std::unordered_set<std::string> inputNames() const;
  /**
   * @brief Retrieves the set of output tensor names.
   *
   * @return An unordered set containing the names of all output tensors.
   */
  std::unordered_set<std::string> outputNames() const;

  /**
   * @brief Checks if the ONNX runtime is properly initialized.
   *
   * @return True if the runtime has been initialized with a valid model, false otherwise.
   */
  bool isInitialized() const { return session_ != nullptr; }

  /**
   * @brief Copies the output tensor data to the input tensor.
   *
   * @param output_name The name of the output tensor.
   * @param input_name The name of the input tensor.
   * @return True if the copy was successful, false otherwise.
   *
   */
  bool copyOutputToInput(const std::string& output_name, const std::string& input_name);

 private:
  template <typename T>
  inline std::optional<std::span<T>> getBuffer(Ort::Value& tensor,
                                               ONNXTensorElementDataType data_type) {
    if (data_type != onnx_type<T>::value) return std::nullopt;
    T* data_ptr = tensor.GetTensorMutableData<T>();
    return std::span<T>(data_ptr, tensor.GetTensorTypeAndShapeInfo().GetElementCount());
  }

  std::unique_ptr<Ort::Env> env_{nullptr};
  std::unique_ptr<Ort::Session> session_{nullptr};
  Ort::AllocatorWithDefaultOptions allocator_{};
  Ort::ModelMetadata metadata_{nullptr};
  Ort::RunOptions run_options_{nullptr};

  struct TensorData {
    std::size_t size;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<const char*> names;
    std::vector<Ort::Value> tensors;
    std::vector<ONNXTensorElementDataType> data_types;
    std::vector<Ort::AllocatedStringPtr> allocated_names;
  };

  TensorData input_;
  TensorData output_;

  std::unordered_map<std::string, int> input_names_to_index_{};
  std::unordered_map<std::string, int> output_names_to_index_{};
};

}  // namespace rai::cs::control::common::onnx
