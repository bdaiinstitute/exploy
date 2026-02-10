// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#include "onnx_runtime.hpp"
#include "logging_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <ranges>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace exploy::control {

namespace {

void resetTensorBuffer(Ort::Value& tensor, ONNXTensorElementDataType data_type) {
  const auto count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      std::fill_n(tensor.GetTensorMutableData<float>(), count, 0.0f);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      std::fill_n(tensor.GetTensorMutableData<int32_t>(), count, 0);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
      std::fill_n(tensor.GetTensorMutableData<bool>(), count, false);
      break;
    }
    default:
      GENERIC_LOG_STREAM(ERROR, "Unsupported tensor type for reset: " << data_type);
      break;
  }
}

template <typename TensorDataType>
void initializeTensorData(TensorDataType& tensor_data, std::unique_ptr<Ort::Session>& session,
                          Ort::AllocatorWithDefaultOptions& allocator,
                          std::unordered_map<std::string, int>& names_to_index, bool is_input) {
  tensor_data.size = is_input ? session->GetInputCount() : session->GetOutputCount();

  tensor_data.names.reserve(tensor_data.size);
  tensor_data.shapes.reserve(tensor_data.size);
  tensor_data.data_types.reserve(tensor_data.size);
  tensor_data.tensors.reserve(tensor_data.size);
  tensor_data.allocated_names.reserve(tensor_data.size);

  for (std::size_t n = 0; n < tensor_data.size; n++) {
    auto name_ptr = is_input ? session->GetInputNameAllocated(n, allocator)
                             : session->GetOutputNameAllocated(n, allocator);
    tensor_data.allocated_names.push_back(std::move(name_ptr));
    tensor_data.names.push_back(tensor_data.allocated_names.back().get());

    auto type_info = is_input ? session->GetInputTypeInfo(n) : session->GetOutputTypeInfo(n);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    tensor_data.shapes.push_back(tensor_info.GetShape());
    tensor_data.data_types.push_back(tensor_info.GetElementType());

    tensor_data.tensors.push_back(Ort::Value::CreateTensor(allocator, tensor_data.shapes[n].data(),
                                                           tensor_data.shapes[n].size(),
                                                           tensor_data.data_types[n]));

    names_to_index[std::string(tensor_data.names.back())] = n;
  }
}

// This is an initial configuration based on ONNX documentation. Adjust as needed.
OrtCUDAProviderOptions createCudaProviderOptions() {
  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = 0;
  cuda_options.arena_extend_strategy = 0;
  cuda_options.gpu_mem_limit = SIZE_MAX;
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  cuda_options.do_copy_in_default_stream = 1;
  return cuda_options;
}

}  // namespace

bool OnnxRuntime::initialize(const std::string& model_path, const OnnxRuntimeOptions& options) {
  if (not std::filesystem::exists(model_path)) {
    GENERIC_LOG_STREAM(WARN, "model file not found " << model_path)
    return false;
  }

  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "OnnxRuntime");
  Ort::SessionOptions session_options;

  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (options.profiling_path.has_value())
    session_options.EnableProfiling(options.profiling_path.value().c_str());

  switch (options.provider) {
    case OnnxRuntimeOptions::ExecutionProvider::CUDA:
      try {
        session_options.AppendExecutionProvider_CUDA(createCudaProviderOptions());
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
      } catch (const Ort::Exception& e) {
        GENERIC_LOG_STREAM(WARN, "Failed to enable CUDA execution provider: "
                                     << e.what() << ". Falling back to CPU.");
        OnnxRuntimeOptions fallback_options = options;
        fallback_options.provider = OnnxRuntimeOptions::ExecutionProvider::CPU;
        return initialize(model_path, fallback_options);
      }
      break;
    case OnnxRuntimeOptions::ExecutionProvider::CPU:
    default:
      session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
      break;
  }

  initializeTensorData(input_, session_, allocator_, input_names_to_index_, /*is_input=*/true);
  initializeTensorData(output_, session_, allocator_, output_names_to_index_, /*is_input=*/false);

  metadata_ = session_->GetModelMetadata();

  return true;
}

bool OnnxRuntime::evaluate() {
  try {
    session_->Run(run_options_, input_.names.data(), input_.tensors.data(), input_.size,
                  output_.names.data(), output_.tensors.data(), output_.size);
  } catch (const Ort::Exception& e) {
    GENERIC_LOG_STREAM(ERROR, "ONNX Runtime evaluation failed: " << e.what());
    return false;
  }
  return true;
}

std::optional<std::string> OnnxRuntime::getCustomMetadata(const std::string& key) const {
  auto string_ptr = metadata_.LookupCustomMetadataMapAllocated(key.c_str(), allocator_);
  return string_ptr == nullptr ? std::nullopt : std::make_optional<std::string>(string_ptr.get());
}

void OnnxRuntime::resetBuffers() {
  for (std::size_t n = 0; n < input_.size; n++) {
    resetTensorBuffer(input_.tensors[n], input_.data_types[n]);
  }
  for (std::size_t n = 0; n < output_.size; n++) {
    resetTensorBuffer(output_.tensors[n], output_.data_types[n]);
  }
}

std::unordered_set<std::string> OnnxRuntime::inputNames() const {
  auto keys = input_names_to_index_ | std::views::keys;
  return {keys.begin(), keys.end()};
}

std::unordered_set<std::string> OnnxRuntime::outputNames() const {
  auto keys = output_names_to_index_ | std::views::keys;
  return {keys.begin(), keys.end()};
}

bool OnnxRuntime::copyOutputToInput(const std::string& output_name, const std::string& input_name) {
  if (!output_names_to_index_.contains(output_name)) {
    GENERIC_LOG_STREAM(ERROR, "Output name not found in model: " + output_name);
    return false;
  }
  if (!input_names_to_index_.contains(input_name)) {
    GENERIC_LOG_STREAM(ERROR, "Input name not found in model: " + input_name);
    return false;
  }

  auto output_index = output_names_to_index_[output_name];
  auto input_index = input_names_to_index_[input_name];
  if (output_.data_types[output_index] != input_.data_types[input_index]) {
    GENERIC_LOG_STREAM(
        ERROR, "Data type mismatch for output " << output_name << " and input " << input_name);
    return false;
  }

  auto copyTensorData = [&]<typename T>() {
    const T* src = output_.tensors[output_index].GetTensorData<T>();
    T* dst = input_.tensors[input_index].GetTensorMutableData<T>();
    const auto count = output_.tensors[output_index].GetTensorTypeAndShapeInfo().GetElementCount();
    std::copy_n(src, count, dst);
  };

  const auto data_type = output_.data_types[output_index];
  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      copyTensorData.template operator()<float>();
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      copyTensorData.template operator()<int32_t>();
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      copyTensorData.template operator()<bool>();
      break;
    default:
      GENERIC_LOG_STREAM(ERROR, "Unsupported tensor type for copy: " << data_type);
      return false;
  }

  return true;
}

}  // namespace exploy::control
