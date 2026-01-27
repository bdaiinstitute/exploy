// Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <span>
#include <string>
#include <vector>
#include <cstdint>

namespace rai::cs::operation::common::data_collection {

/**
 * @class DataCollectionInterface
 *
 * @brief Interface which provides methods to collect data.
 *
 */
class DataCollectionInterface {
 public:
  virtual ~DataCollectionInterface() = default;

  /**
   * @brief Register a span data source for data collection.
   *
   * @param prefix A prefix to identify the data.
   * @param data The data to be logged.
   */
  virtual bool registerDataSource(const std::string& prefix, std::span<const double> data) = 0;
  /**
   * @brief Register a span data source for data collection.
   *
   * @param prefix A prefix to identify the data.
   * @param data The data to be logged.
   */
  virtual bool registerDataSource(const std::string& prefix, std::span<const float> data) = 0;
  /**
   * @brief Register a scalar data source for data collection.
   *
   * @param prefix A prefix to identify the data.
   * @param data The data to be logged.
   */
  virtual bool registerDataSource(const std::string& prefix, const double& data) = 0;
  /**
   * @brief Collect registered data.
   *
   * @param time_us Timestamp in microseconds.
   * @return true if data collection succeeded, false otherwise.
   */
  virtual bool collectData(uint64_t time_us) = 0;
};

}  // namespace rai::cs::operation::common::data_collection
