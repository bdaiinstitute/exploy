// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interfaces.hpp"
#include "logging_utils.hpp"

namespace exploy::control {

/**
 * @brief Configuration for height scan sensors.
 *
 * Specifies the grid pattern and layers for terrain height scanning.
 */
struct HeightScanConfig {
  /**
   * @brief Grid pattern configuration for height scanning.
   */
  struct Pattern {
    Eigen::Vector2d size{};    ///< Grid size (x, y) in meters.
    double resolution{};       ///< Grid resolution (spacing between points) in meters.
    Eigen::Vector2d offset{};  ///< Grid offset (x, y) from base frame in meters.
  };
  Pattern pattern{};  ///< Grid pattern specification.
  std::unordered_set<std::string>
      layer_names{};  ///< Set of layer names to include (e.g., "height", "r", "g", "b").
};

/**
 * @brief Configuration for range image (LiDAR) sensors.
 *
 * Specifies resolution, field of view, and unobserved value sentinel for range images.
 */
struct RangeImageConfig {
  int v_res{};                ///< Number of pixels in vertical direction.
  int h_res{};                ///< Number of pixels in horizontal direction.
  double v_fov_min_deg{};     ///< Minimum vertical field of view angle in degrees.
  double v_fov_max_deg{};     ///< Maximum vertical field of view angle in degrees.
  double unobserved_value{};  ///< Sentinel value for pixels with no LiDAR return.
};

/**
 * @brief Configuration for depth image (camera) sensors.
 *
 * Specifies image dimensions and camera intrinsic parameters.
 */
struct DepthImageConfig {
  int width{};   ///< Image width in pixels.
  int height{};  ///< Image height in pixels.
  double fx{};   ///< Focal length in x direction (pixels).
  double fy{};   ///< Focal length in y direction (pixels).
  double cx{};   ///< Principal point x-coordinate (pixels).
  double cy{};   ///< Principal point y-coordinate (pixels).
};

/**
 * @class RobotStateInterface
 *
 * @brief Interface which provides methods to communicate with the robot.
 *
 * This class describes the interface for the communication with the robot state and commands.
 *
 */
class RobotStateInterface {
 public:
  virtual ~RobotStateInterface() = default;

  /**
   * @brief Initialize data source of base position in world frame (x,y,z).
   *
   * Called once during initialization (usually non real-time).
   */
  virtual bool initBasePosW() {
    GENERIC_LOG_STREAM(ERROR, "initBasePosW() not implemented");
    return false;
  }
  /**
   * @brief Initialize data source of base orientation in world frame (w,x,y,z).
   *
   * Called once during initialization (usually non real-time).
   */
  virtual bool initBaseQuatW() {
    GENERIC_LOG_STREAM(ERROR, "initBaseQuatW() not implemented");
    return false;
  }
  /**
   * @brief Get base position in world frame.
   *
   * @return The base position in world frame (x,y,z).
   */
  virtual std::optional<Position> basePosW() const {
    GENERIC_LOG_STREAM(ERROR, "basePosW() not implemented");
    return std::nullopt;
  }
  /**
   * @brief Get base orientation in world frame.
   *
   * @return The base orientation in world frame (w,x,y,z).
   */
  virtual std::optional<Quaternion> baseQuatW() const {
    GENERIC_LOG_STREAM(ERROR, "baseQuatW() not implemented");
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of linear base velocity in base frame.
   *
   * Called once during initialization (usually non real-time).
   */
  virtual bool initBaseLinVelB() {
    GENERIC_LOG_STREAM(ERROR, "initBaseLinVelB() not implemented");
    return false;
  }
  /**
   * @brief Initialize data source of angular base velocity in base frame.
   *
   * Called once during initialization (usually non real-time).
   */
  virtual bool initBaseAngVelB() {
    GENERIC_LOG_STREAM(ERROR, "initBaseAngVelB() not implemented");
    return false;
  }
  /**
   * @brief Get linear base velocity rotated in base frame.
   *
   * @return The linear base velocity in base frame (vx, vy, vz).
   */
  virtual std::optional<LinearVelocity> baseLinVelB() const {
    GENERIC_LOG_STREAM(ERROR, "baseLinVelB() not implemented");
    return std::nullopt;
  }
  /**
   * @brief Get angular base velocity rotated in base frame.
   *
   * @return The angular base velocity in base frame (ωx, ωy, ωz).
   */
  virtual std::optional<AngularVelocity> baseAngVelB() const {
    GENERIC_LOG_STREAM(ERROR, "baseAngVelB() not implemented");
    return std::nullopt;
  }
  /**
   * @brief Initialize data sources for joint position data.
   *
   * Called once during initialization (non real-time).
   *
   * @param joint_name The name of the joint.
   */
  virtual bool initJointPosition(const std::string& joint_name) {
    GENERIC_LOG_STREAM(ERROR, "initJointPosition() not implemented for joint: " << joint_name);
    return false;
  }
  /**
   * @brief Initialize data sources for joint velocity data.
   *
   * Called once during initialization (non real-time).
   *
   * @param joint_name The name of the joint.
   */
  virtual bool initJointVelocity(const std::string& joint_name) {
    GENERIC_LOG_STREAM(ERROR, "initJointVelocity() not implemented for joint: " << joint_name);
    return false;
  }
  /**
   * @brief Initialize data sources for joint effort data.
   *
   * Called once during initialization (non real-time).
   *
   * @param joint_name The name of the joint.
   */
  virtual bool initJointEffort(const std::string& joint_name) {
    GENERIC_LOG_STREAM(ERROR, "initJointEffort() not implemented for joint: " << joint_name);
    return false;
  }
  /**
   * @brief Initialize data sinks for joint output data.
   *
   * Called once during initialization (non real-time).
   *
   * @param joint_name The name of the joint.
   */
  virtual bool initJointOutput(const std::string& joint_name) {
    GENERIC_LOG_STREAM(ERROR, "initJointOutput() not implemented for joint: " << joint_name);
    return false;
  }
  /**
   * @brief Get joint position.
   *
   * @param joint_name The name of the joint.
   * @return The position of the joint.
   */
  virtual std::optional<double> jointPosition(const std::string& joint_name) const {
    GENERIC_LOG_STREAM(ERROR, "jointPosition() not implemented for joint: " << joint_name);
    return std::nullopt;
  }
  /**
   * @brief Get joint velocity.
   *
   * @param joint_name The name of the joint.
   * @return The velocity of the joint.
   */
  virtual std::optional<double> jointVelocity(const std::string& joint_name) const {
    GENERIC_LOG_STREAM(ERROR, "jointVelocity() not implemented for joint: " << joint_name);
    return std::nullopt;
  }
  /**
   * @brief Get joint effort.
   *
   * @param joint_name The name of the joint.
   * @return The effort of the joint.
   */
  virtual std::optional<double> jointEffort(const std::string& joint_name) const {
    GENERIC_LOG_STREAM(ERROR, "jointEffort() not implemented for joint: " << joint_name);
    return std::nullopt;
  }
  /**
   * @brief Set joint position.
   *
   * The following control law is assumed:
   * u = joint_effort + p_gain * (joint_position - joint_position_measured) + d_gain *
   * (joint_velocity - joint_velocity_measured)
   *
   * @param joint_name The name of the joint.
   * @param position The joint position to be set.
   */
  virtual bool setJointPosition(const std::string& joint_name, double position) {
    GENERIC_LOG_STREAM(ERROR, "setJointPosition() not implemented for joint: "
                                  << joint_name << ", position: " << position);
    return false;
  }
  /**
   * @brief Set joint velocity.
   *
   * The following control law is assumed:
   * u = joint_effort + p_gain * (joint_position - joint_position_measured) + d_gain *
   * (joint_velocity - joint_velocity_measured)
   *
   * @param joint_name The name of the joint.
   * @param velocity The joint velocity to be set.
   */
  virtual bool setJointVelocity(const std::string& joint_name, double velocity) {
    GENERIC_LOG_STREAM(ERROR, "setJointVelocity() not implemented for joint: "
                                  << joint_name << ", velocity: " << velocity);
    return false;
  }
  /**
   * @brief Set joint effort.
   *
   * The following control law is assumed:
   * u = joint_effort + p_gain * (joint_position - joint_position_measured) + d_gain *
   * (joint_velocity - joint_velocity_measured)
   *
   * @param joint_name The name of the joint.
   * @param effort The joint effort to be set.
   */
  virtual bool setJointEffort(const std::string& joint_name, double effort) {
    GENERIC_LOG_STREAM(ERROR, "setJointEffort() not implemented for joint: "
                                  << joint_name << ", effort: " << effort);
    return false;
  }
  /**
   * @brief Set joint p-gain.
   *
   * The following control law is assumed:
   * u = joint_effort + p_gain * (joint_position - joint_position_measured) + d_gain *
   * (joint_velocity - joint_velocity_measured)
   *
   * @param joint_name The name of the joint.
   * @param p_gain The p-gain to be set.
   */
  virtual bool setJointPGain(const std::string& joint_name, double p_gain) {
    GENERIC_LOG_STREAM(ERROR, "setJointPGain() not implemented for joint: "
                                  << joint_name << ", p_gain: " << p_gain);
    return false;
  }
  /**
   * @brief Set joint d-gain.
   *
   * The following control law is assumed:
   * u = joint_effort + p_gain * (joint_position - joint_position_measured) + d_gain *
   * (joint_velocity - joint_velocity_measured)
   *
   * @param joint_name The name of the joint.
   * @param d_gain The d-gain to be set.
   */
  virtual bool setJointDGain(const std::string& joint_name, double d_gain) {
    GENERIC_LOG_STREAM(ERROR, "setJointDGain() not implemented for joint: "
                                  << joint_name << ", d_gain: " << d_gain);
    return false;
  }
  /**
   * @brief Initialize data sources of se(2) frame velocity.
   *
   * Called once during initialization (non real-time).
   *
   * @param frame_name The name of the considered frame.
   */
  virtual bool initSe2Velocity(const std::string& frame_name) {
    GENERIC_LOG_STREAM(ERROR, "initSe2Velocity() not implemented for frame: " << frame_name);
    return false;
  }
  /**
   * @brief Set se(2) velocity reference of a specific frame.
   *
   * @param velocity The se(2) velocity to be set.
   *
   * @param frame_name The name of the considered frame.
   */
  virtual bool setSe2Velocity(const std::string& frame_name, const SE2Velocity& /*velocity*/) {
    GENERIC_LOG_STREAM(ERROR, "setSe2Velocity() not implemented for frame: " << frame_name);
    return false;
  }
  /**
   * @brief Initialize data source of IMU angular velocity.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param imu_name The name of the IMU.
   */
  virtual bool initImuAngularVelocityImu(const std::string& imu_name) {
    GENERIC_LOG_STREAM(ERROR, "initImuAngularVelocityImu() not implemented for IMU: " << imu_name);
    return false;
  }
  /**
   * @brief Initialize data source of IMU orientation.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param imu_name The name of the IMU.
   */
  virtual bool initImuOrientationW(const std::string& imu_name) {
    GENERIC_LOG_STREAM(ERROR, "initImuOrientationW() not implemented for IMU: " << imu_name);
    return false;
  }
  /**
   * @brief Get IMU angular velocity in IMU frame.
   *
   * @param imu_name The name of the IMU.
   * @return The angular velocity of the IMU in world frame (x,y,z) .
   */
  virtual std::optional<AngularVelocity> imuAngularVelocityImu(const std::string& imu_name) const {
    GENERIC_LOG_STREAM(ERROR, "imuAngularVelocityImu() not implemented for IMU: " << imu_name);
    return std::nullopt;
  }
  /**
   * @brief Get IMU orientation in world frame (w, x, y, z).
   *
   * @param imu_name The name of the IMU.
   * @return The orientation of the IMU in world frame represented as a unit quaternion (w,x,y,z).
   */
  virtual std::optional<Quaternion> imuOrientationW(const std::string& imu_name) const {
    GENERIC_LOG_STREAM(ERROR, "imuOrientationW() not implemented for IMU: " << imu_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of body position in world frame.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param body_name The name of the body.
   */
  virtual bool initBodyPositionW(const std::string& body_name) {
    GENERIC_LOG_STREAM(ERROR, "initBodyPositionW() not implemented for body: " << body_name);
    return false;
  }
  /**
   * @brief Initialize data source of body orientation in base frame (w, x, y, z).
   *
   * Called once during initialization (usually non real-time).
   *
   * @param body_name The name of the body.
   */
  virtual bool initBodyOrientationW(const std::string& body_name) {
    GENERIC_LOG_STREAM(ERROR, "initBodyOrientationW() not implemented for body: " << body_name);
    return false;
  }
  /**
   * @brief Initialize data source of body linear velocity in body frame.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param body_name The name of the body.
   */
  virtual bool initBodyLinearVelocityB(const std::string& body_name) {
    GENERIC_LOG_STREAM(ERROR, "initBodyLinearVelocityB() not implemented for body: " << body_name);
    return false;
  }
  /**
   * @brief Initialize data source of body angular velocity in body frame.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param body_name The name of the body.
   */
  virtual bool initBodyAngularVelocityB(const std::string& body_name) {
    GENERIC_LOG_STREAM(ERROR, "initBodyAngularVelocityB() not implemented for body: " << body_name);
    return false;
  }
  /**
   * @brief Get body position in world frame.
   *
   * @param body_name The name of the body.
   * @return The position of the body in world frame.
   */
  virtual std::optional<Position> bodyPositionW(const std::string& body_name) const {
    GENERIC_LOG_STREAM(ERROR, "bodyPositionW() not implemented for body: " << body_name);
    return std::nullopt;
  }
  /**
   * @brief Get body orientation in world frame (w, x, y, z).
   *
   * @param body_name The name of the body.
   * @return The orientation of the body in world frame.
   */
  virtual std::optional<Quaternion> bodyOrientationW(const std::string& body_name) const {
    GENERIC_LOG_STREAM(ERROR, "bodyOrientationW() not implemented for body: " << body_name);
    return std::nullopt;
  }
  /**
   * @brief Get body linear velocity in body frame.
   *
   * @param body_name The name of the body.
   * @return The linear velocity of the body in body frame.
   */
  virtual std::optional<LinearVelocity> bodyLinearVelocityB(const std::string& body_name) const {
    GENERIC_LOG_STREAM(ERROR, "bodyLinearVelocityB() not implemented for body: " << body_name);
    return std::nullopt;
  }
  /**
   * @brief Get body angular velocity in body frame.
   *
   * @param body_name The name of the body.
   * @return The angular velocity of the body in body frame.
   */
  virtual std::optional<AngularVelocity> bodyAngularVelocityB(const std::string& body_name) const {
    GENERIC_LOG_STREAM(ERROR, "bodyAngularVelocityB() not implemented for body: " << body_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of the heightscan.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param config The configuration of the heightscan.
   */
  virtual bool initHeightScan(const std::string& /*sensor_name*/,
                              const HeightScanConfig& /*config*/) {
    GENERIC_LOG_STREAM(ERROR, "initHeightScan() not implemented");
    return false;
  }
  /**
   * @brief Get subsampled heightscan in base frame.
   *
   * This function returns the flattened (column-major), subsampled heightscan with the pattern
   * specified in the init function. The pattern is centered around the base.
   *
   * @param base_pos_w The base position in world frame.
   * @param base_quat_w The base orientation in world frame.
   * @return A vector of heightscans.
   */
  virtual std::optional<HeightScan*> heightScan(
      const std::string& /*sensor_name*/, const std::unordered_set<std::string>& /*layer_names*/,
      const Position& /*base_pos_w*/, const Quaternion& /*base_quat_w*/) {
    GENERIC_LOG_STREAM(ERROR, "heightScan() not implemented");
    return std::nullopt;
  }
  /**
   * @brief Initialize the LiDAR range image data source.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param config The configuration of the range image.
   */
  virtual bool initRangeImage(const RangeImageConfig& /*config*/) {
    GENERIC_LOG_STREAM(ERROR, "initRangeImage() not implemented");
    return false;
  }
  /**
   * @brief Get the range image.
   *
   * This function returns the image, flattened in row-major storage order.
   *
   * @return The flattened range image.
   */
  virtual std::optional<std::span<const float>> rangeImage() {
    GENERIC_LOG_STREAM(ERROR, "rangeImage() not implemented");
    return std::nullopt;
  }
  /**
   * @brief Initialize the depth image data source.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param config The configuration of the depth image.
   */
  virtual bool initDepthImage(const DepthImageConfig& /*config*/) {
    GENERIC_LOG_STREAM(ERROR, "initDepthImage() not implemented");
    return false;
  }
  /**
   * @brief Get the depth image.
   *
   * This function returns the image, flattened in row-major storage order.
   *
   * @return The flattened depth image.
   */
  virtual std::optional<std::span<const float>> depthImage() {
    GENERIC_LOG_STREAM(ERROR, "depthImage() not implemented");
    return std::nullopt;
  }
};

}  // namespace exploy::control
