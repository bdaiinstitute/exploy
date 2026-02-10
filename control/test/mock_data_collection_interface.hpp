// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <gmock/gmock.h>
#include "data_collection_interface.hpp"

namespace exploy::control {

class MockDataCollectionInterface : public DataCollectionInterface {
 public:
  MOCK_METHOD(bool, registerDataSource, (const std::string& prefix, std::span<const double> data),
              (override));
  MOCK_METHOD(bool, registerDataSource, (const std::string& prefix, std::span<const float> data),
              (override));
  MOCK_METHOD(bool, registerDataSource, (const std::string& prefix, const double& data),
              (override));
  MOCK_METHOD(bool, collectData, (uint64_t time_us), (override));
};

}  // namespace exploy::control
