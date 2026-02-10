// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace exploy::control {

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
  virtual bool registerDataSource(const std::string& /*prefix*/, std::span<const double> /*data*/) {
    return false;
  };
  /**
   * @brief Register a span data source for data collection.
   *
   * @param prefix A prefix to identify the data.
   * @param data The data to be logged.
   */
  virtual bool registerDataSource(const std::string& /*prefix*/, std::span<const float> /*data*/) {
    return false;
  };
  /**
   * @brief Register a scalar data source for data collection.
   *
   * @param prefix A prefix to identify the data.
   * @param data The data to be logged.
   */
  virtual bool registerDataSource(const std::string& /*prefix*/, const double& /*data*/) {
    return false;
  };
  /**
   * @brief Collect registered data.
   *
   * @param time_us Timestamp in microseconds.
   * @return true if data collection succeeded, false otherwise.
   */
  virtual bool collectData(uint64_t time_us) = 0;
};

}  // namespace exploy::control
