// Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <gmock/gmock.h>
#include "data_collection_interface.hpp"

namespace rai::cs::operation::common::data_collection_public {

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

}  // namespace rai::cs::operation::common::data_collection_public
