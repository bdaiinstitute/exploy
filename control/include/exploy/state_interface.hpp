// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <optional>
#include <string>
#include <unordered_set>

#include "exploy/interfaces.hpp"
#include "exploy/logging_utils.hpp"

namespace exploy::control {

/**
 * @brief Grid pattern configuration for height scanning.
 */
struct HeightScanPattern {
  Eigen::Vector2d size{};    ///< Grid size (x, y) in meters.
  double resolution{};       ///< Grid resolution (spacing between points) in meters.
  Eigen::Vector2d offset{};  ///< Grid offset (x, y) from base frame in meters.
};

// ---------------------------------------------------------------------------
// Per-function info structs.
//
// Every public RobotStateInterface method takes a dedicated info struct instead
// of loose parameters. This keeps the call sites and overrides stable when more
// information needs to be passed along in the future: a new field can be added
// to the relevant struct without changing any method signature.
// ---------------------------------------------------------------------------

/// @brief Arguments for RobotStateInterface::initBasePosW and RobotStateInterface::basePosW.
struct BasePosWInfo {
  std::string articulation_name;  ///< Articulation or rigid object the base belongs to.
};

/// @brief Arguments for RobotStateInterface::initBaseQuatW and RobotStateInterface::baseQuatW.
struct BaseQuatWInfo {
  std::string articulation_name;  ///< Articulation or rigid object the base belongs to.
};

/// @brief Arguments for RobotStateInterface::initBaseLinVelB and RobotStateInterface::baseLinVelB.
struct BaseLinVelBInfo {
  std::string articulation_name;  ///< Articulation or rigid object the base belongs to.
};

/// @brief Arguments for RobotStateInterface::initBaseAngVelB and RobotStateInterface::baseAngVelB.
struct BaseAngVelBInfo {
  std::string articulation_name;  ///< Articulation or rigid object the base belongs to.
};

/// @brief Arguments for RobotStateInterface::initJointPosition and
///        RobotStateInterface::jointPosition.
struct JointPositionInfo {
  std::string articulation_name;  ///< Articulation the joint belongs to.
  std::string joint_name;         ///< Name of the joint.
};

/// @brief Arguments for RobotStateInterface::initJointVelocity and
///        RobotStateInterface::jointVelocity.
struct JointVelocityInfo {
  std::string articulation_name;  ///< Articulation the joint belongs to.
  std::string joint_name;         ///< Name of the joint.
};

/// @brief Arguments for RobotStateInterface::initJointEffort and RobotStateInterface::jointEffort.
struct JointEffortInfo {
  std::string articulation_name;  ///< Articulation the joint belongs to.
  std::string joint_name;         ///< Name of the joint.
};

/// @brief Arguments for RobotStateInterface::initJointOutput.
struct JointOutputInfo {
  std::string articulation_name;  ///< Articulation the joint belongs to.
  std::string joint_name;         ///< Name of the joint.
};

/// @brief Arguments for RobotStateInterface::setJointPosition.
struct SetJointPositionInfo {
  std::string articulation_name;  ///< Articulation the joint belongs to.
  std::string joint_name;         ///< Name of the joint.
  double position{};              ///< Joint position to be set.
};

/// @brief Arguments for RobotStateInterface::setJointVelocity.
struct SetJointVelocityInfo {
  std::string articulation_name;  ///< Articulation the joint belongs to.
  std::string joint_name;         ///< Name of the joint.
  double velocity{};              ///< Joint velocity to be set.
};

/// @brief Arguments for RobotStateInterface::setJointEffort.
struct SetJointEffortInfo {
  std::string articulation_name;  ///< Articulation the joint belongs to.
  std::string joint_name;         ///< Name of the joint.
  double effort{};                ///< Joint effort to be set.
};

/// @brief Arguments for RobotStateInterface::setJointPGain.
struct SetJointPGainInfo {
  std::string articulation_name;  ///< Articulation the joint belongs to.
  std::string joint_name;         ///< Name of the joint.
  double p_gain{};                ///< Proportional gain to be set.
};

/// @brief Arguments for RobotStateInterface::setJointDGain.
struct SetJointDGainInfo {
  std::string articulation_name;  ///< Articulation the joint belongs to.
  std::string joint_name;         ///< Name of the joint.
  double d_gain{};                ///< Derivative gain to be set.
};

/// @brief Arguments for RobotStateInterface::initSe2Velocity and
/// RobotStateInterface::setSe2Velocity.
struct Se2VelocityInfo {
  std::string frame_name;  ///< Name of the considered frame.
  SE2Velocity velocity{};  ///< The se(2) velocity to be set (used by setSe2Velocity).
};

/// @brief Arguments for RobotStateInterface::initImuLinearVelocityImu and
///        RobotStateInterface::imuLinearVelocityImu.
struct ImuLinearVelocityImuInfo {
  std::string imu_name;  ///< Name of the IMU.
};

/// @brief Arguments for RobotStateInterface::initImuAngularVelocityImu and
///        RobotStateInterface::imuAngularVelocityImu.
struct ImuAngularVelocityImuInfo {
  std::string imu_name;  ///< Name of the IMU.
};

/// @brief Arguments for RobotStateInterface::initImuOrientationW and
///        RobotStateInterface::imuOrientationW.
struct ImuOrientationWInfo {
  std::string imu_name;  ///< Name of the IMU.
};

/// @brief Arguments for RobotStateInterface::initBodyPositionW and
///        RobotStateInterface::bodyPositionW.
struct BodyPositionWInfo {
  std::string articulation_name;  ///< Articulation or rigid object the body belongs to.
  std::string body_name;          ///< Name of the body.
};

/// @brief Arguments for RobotStateInterface::initBodyOrientationW and
///        RobotStateInterface::bodyOrientationW.
struct BodyOrientationWInfo {
  std::string articulation_name;  ///< Articulation or rigid object the body belongs to.
  std::string body_name;          ///< Name of the body.
};

/// @brief Arguments for RobotStateInterface::initBodyLinearVelocityB and
///        RobotStateInterface::bodyLinearVelocityB.
struct BodyLinearVelocityBInfo {
  std::string articulation_name;  ///< Articulation or rigid object the body belongs to.
  std::string body_name;          ///< Name of the body.
};

/// @brief Arguments for RobotStateInterface::initBodyAngularVelocityB and
///        RobotStateInterface::bodyAngularVelocityB.
struct BodyAngularVelocityBInfo {
  std::string articulation_name;  ///< Articulation or rigid object the body belongs to.
  std::string body_name;          ///< Name of the body.
};

/// @brief Arguments for RobotStateInterface::initHeightScan and RobotStateInterface::heightScan.
struct HeightScanInfo {
  std::string sensor_name;      ///< Name of the height scan sensor.
  HeightScanPattern pattern{};  ///< Grid pattern specification (used by initHeightScan).
  std::unordered_set<std::string> layer_names{};  ///< Layer names to include / retrieve.
};

/// @brief Arguments for RobotStateInterface::initSphericalImage and
///        RobotStateInterface::sphericalImage.
struct SphericalImageInfo {
  std::string sensor_name;  ///< Name of the spherical image sensor.
  int v_res{};              ///< Number of pixels in vertical direction (used by init).
  int h_res{};              ///< Number of pixels in horizontal direction (used by init).
  double v_fov_min_deg{};   ///< Minimum vertical field of view angle in degrees (used by init).
  double v_fov_max_deg{};   ///< Maximum vertical field of view angle in degrees (used by init).
  std::unordered_set<std::string> channel_names{};  ///< Channel names to include / retrieve.
};

/// @brief Arguments for RobotStateInterface::initPinholeImage and
/// RobotStateInterface::pinholeImage.
struct PinholeImageInfo {
  std::string sensor_name;  ///< Name of the pinhole image sensor.
  int width{};              ///< Image width in pixels (used by init).
  int height{};             ///< Image height in pixels (used by init).
  double fx{};              ///< Focal length in x direction in pixels (used by init).
  double fy{};              ///< Focal length in y direction in pixels (used by init).
  double cx{};              ///< Principal point x-coordinate in pixels (used by init).
  double cy{};              ///< Principal point y-coordinate in pixels (used by init).
  std::unordered_set<std::string> channel_names{};  ///< Channel names to include / retrieve.
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
   *
   * @param info Arguments including the name of the articulation or rigid object the base belongs
   * to.
   */
  virtual bool initBasePosW(const BasePosWInfo& info) {
    LOG_STREAM(ERROR,
               "initBasePosW() not implemented for articulation: " << info.articulation_name);
    return false;
  }
  /**
   * @brief Initialize data source of base orientation in world frame (w,x,y,z).
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info Arguments including the name of the articulation or rigid object the base belongs
   * to.
   */
  virtual bool initBaseQuatW(const BaseQuatWInfo& info) {
    LOG_STREAM(ERROR,
               "initBaseQuatW() not implemented for articulation: " << info.articulation_name);
    return false;
  }
  /**
   * @brief Get base position in world frame.
   *
   * @param info Arguments including the name of the articulation or rigid object the base belongs
   * to.
   * @return The base position in world frame (x,y,z).
   */
  virtual std::optional<Position> basePosW(const BasePosWInfo& info) const {
    LOG_STREAM(ERROR, "basePosW() not implemented for articulation: " << info.articulation_name);
    return std::nullopt;
  }
  /**
   * @brief Get base orientation in world frame.
   *
   * @param info Arguments including the name of the articulation or rigid object the base belongs
   * to.
   * @return The base orientation in world frame (w,x,y,z).
   */
  virtual std::optional<Quaternion> baseQuatW(const BaseQuatWInfo& info) const {
    LOG_STREAM(ERROR, "baseQuatW() not implemented for articulation: " << info.articulation_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of linear base velocity in base frame.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info Arguments including the name of the articulation or rigid object the base belongs
   * to.
   */
  virtual bool initBaseLinVelB(const BaseLinVelBInfo& info) {
    LOG_STREAM(ERROR,
               "initBaseLinVelB() not implemented for articulation: " << info.articulation_name);
    return false;
  }
  /**
   * @brief Initialize data source of angular base velocity in base frame.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info Arguments including the name of the articulation or rigid object the base belongs
   * to.
   */
  virtual bool initBaseAngVelB(const BaseAngVelBInfo& info) {
    LOG_STREAM(ERROR,
               "initBaseAngVelB() not implemented for articulation: " << info.articulation_name);
    return false;
  }
  /**
   * @brief Get linear base velocity rotated in base frame.
   *
   * @param info Arguments including the name of the articulation or rigid object the base belongs
   * to.
   * @return The linear base velocity in base frame (vx, vy, vz).
   */
  virtual std::optional<LinearVelocity> baseLinVelB(const BaseLinVelBInfo& info) const {
    LOG_STREAM(ERROR, "baseLinVelB() not implemented for articulation: " << info.articulation_name);
    return std::nullopt;
  }
  /**
   * @brief Get angular base velocity rotated in base frame.
   *
   * @param info Arguments including the name of the articulation or rigid object the base belongs
   * to.
   * @return The angular base velocity in base frame (ωx, ωy, ωz).
   */
  virtual std::optional<AngularVelocity> baseAngVelB(const BaseAngVelBInfo& info) const {
    LOG_STREAM(ERROR, "baseAngVelB() not implemented for articulation: " << info.articulation_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data sources for joint position data.
   *
   * Called once during initialization (non real-time).
   *
   * @param info Arguments including the articulation name and the joint name.
   */
  virtual bool initJointPosition(const JointPositionInfo& info) {
    LOG_STREAM(ERROR, "initJointPosition() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name);
    return false;
  }
  /**
   * @brief Initialize data sources for joint velocity data.
   *
   * Called once during initialization (non real-time).
   *
   * @param info Arguments including the articulation name and the joint name.
   */
  virtual bool initJointVelocity(const JointVelocityInfo& info) {
    LOG_STREAM(ERROR, "initJointVelocity() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name);
    return false;
  }
  /**
   * @brief Initialize data sources for joint effort data.
   *
   * Called once during initialization (non real-time).
   *
   * @param info Arguments including the articulation name and the joint name.
   */
  virtual bool initJointEffort(const JointEffortInfo& info) {
    LOG_STREAM(ERROR, "initJointEffort() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name);
    return false;
  }
  /**
   * @brief Initialize data sinks for joint output data.
   *
   * Called once during initialization (non real-time).
   *
   * @param info Arguments including the articulation name and the joint name.
   */
  virtual bool initJointOutput(const JointOutputInfo& info) {
    LOG_STREAM(ERROR, "initJointOutput() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name);
    return false;
  }
  /**
   * @brief Get joint position.
   *
   * @param info Arguments including the articulation name and the joint name.
   * @return The position of the joint.
   */
  virtual std::optional<double> jointPosition(const JointPositionInfo& info) const {
    LOG_STREAM(ERROR, "jointPosition() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name);
    return std::nullopt;
  }
  /**
   * @brief Get joint velocity.
   *
   * @param info Arguments including the articulation name and the joint name.
   * @return The velocity of the joint.
   */
  virtual std::optional<double> jointVelocity(const JointVelocityInfo& info) const {
    LOG_STREAM(ERROR, "jointVelocity() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name);
    return std::nullopt;
  }
  /**
   * @brief Get joint effort.
   *
   * @param info Arguments including the articulation name and the joint name.
   * @return The effort of the joint.
   */
  virtual std::optional<double> jointEffort(const JointEffortInfo& info) const {
    LOG_STREAM(ERROR, "jointEffort() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name);
    return std::nullopt;
  }
  /**
   * @brief Set joint position.
   *
   * The following control law is assumed:
   * u = joint_effort + p_gain * (joint_position - joint_position_measured) + d_gain *
   * (joint_velocity - joint_velocity_measured)
   *
   * @param info Arguments including the articulation name, joint name, and the joint position to be
   *             set.
   */
  virtual bool setJointPosition(const SetJointPositionInfo& info) {
    LOG_STREAM(ERROR, "setJointPosition() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name
                          << ", position: " << info.position);
    return false;
  }
  /**
   * @brief Set joint velocity.
   *
   * The following control law is assumed:
   * u = joint_effort + p_gain * (joint_position - joint_position_measured) + d_gain *
   * (joint_velocity - joint_velocity_measured)
   *
   * @param info Arguments including the articulation name, joint name, and the joint velocity to be
   *             set.
   */
  virtual bool setJointVelocity(const SetJointVelocityInfo& info) {
    LOG_STREAM(ERROR, "setJointVelocity() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name
                          << ", velocity: " << info.velocity);
    return false;
  }
  /**
   * @brief Set joint effort.
   *
   * The following control law is assumed:
   * u = joint_effort + p_gain * (joint_position - joint_position_measured) + d_gain *
   * (joint_velocity - joint_velocity_measured)
   *
   * @param info Arguments including the articulation name, joint name, and the joint effort to be
   *             set.
   */
  virtual bool setJointEffort(const SetJointEffortInfo& info) {
    LOG_STREAM(ERROR, "setJointEffort() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name
                          << ", effort: " << info.effort);
    return false;
  }
  /**
   * @brief Set joint p-gain.
   *
   * The following control law is assumed:
   * u = joint_effort + p_gain * (joint_position - joint_position_measured) + d_gain *
   * (joint_velocity - joint_velocity_measured)
   *
   * @param info Arguments including the articulation name, joint name, and the p-gain to be set.
   */
  virtual bool setJointPGain(const SetJointPGainInfo& info) {
    LOG_STREAM(ERROR, "setJointPGain() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name
                          << ", p_gain: " << info.p_gain);
    return false;
  }
  /**
   * @brief Set joint d-gain.
   *
   * The following control law is assumed:
   * u = joint_effort + p_gain * (joint_position - joint_position_measured) + d_gain *
   * (joint_velocity - joint_velocity_measured)
   *
   * @param info Arguments including the articulation name, joint name, and the d-gain to be set.
   */
  virtual bool setJointDGain(const SetJointDGainInfo& info) {
    LOG_STREAM(ERROR, "setJointDGain() not implemented for articulation: "
                          << info.articulation_name << ", joint: " << info.joint_name
                          << ", d_gain: " << info.d_gain);
    return false;
  }
  /**
   * @brief Initialize data sources of se(2) frame velocity.
   *
   * Called once during initialization (non real-time).
   *
   * @param info Arguments including the name of the considered frame.
   */
  virtual bool initSe2Velocity(const Se2VelocityInfo& info) {
    LOG_STREAM(ERROR, "initSe2Velocity() not implemented for frame: " << info.frame_name);
    return false;
  }
  /**
   * @brief Set se(2) velocity reference of a specific frame.
   *
   * @param info Arguments including the considered frame name and the se(2) velocity to be set.
   */
  virtual bool setSe2Velocity(const Se2VelocityInfo& info) {
    LOG_STREAM(ERROR, "setSe2Velocity() not implemented for frame: " << info.frame_name);
    return false;
  }
  /**
   * @brief Initialize data source of IMU linear velocity.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info Arguments including the name of the IMU.
   */
  virtual bool initImuLinearVelocityImu(const ImuLinearVelocityImuInfo& info) {
    LOG_STREAM(ERROR, "initImuLinearVelocityImu() not implemented for IMU: " << info.imu_name);
    return false;
  }
  /**
   * @brief Get IMU linear velocity in IMU frame.
   *
   * @param info Arguments including the name of the IMU.
   * @return The linear velocity of the IMU in IMU frame (vx, vy, vz).
   */
  virtual std::optional<LinearVelocity> imuLinearVelocityImu(
      const ImuLinearVelocityImuInfo& info) const {
    LOG_STREAM(ERROR, "imuLinearVelocityImu() not implemented for IMU: " << info.imu_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of IMU angular velocity.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info Arguments including the name of the IMU.
   */
  virtual bool initImuAngularVelocityImu(const ImuAngularVelocityImuInfo& info) {
    LOG_STREAM(ERROR, "initImuAngularVelocityImu() not implemented for IMU: " << info.imu_name);
    return false;
  }
  /**
   * @brief Initialize data source of IMU orientation.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info Arguments including the name of the IMU.
   */
  virtual bool initImuOrientationW(const ImuOrientationWInfo& info) {
    LOG_STREAM(ERROR, "initImuOrientationW() not implemented for IMU: " << info.imu_name);
    return false;
  }
  /**
   * @brief Get IMU angular velocity in IMU frame.
   *
   * @param info Arguments including the name of the IMU.
   * @return The angular velocity of the IMU in world frame (x,y,z) .
   */
  virtual std::optional<AngularVelocity> imuAngularVelocityImu(
      const ImuAngularVelocityImuInfo& info) const {
    LOG_STREAM(ERROR, "imuAngularVelocityImu() not implemented for IMU: " << info.imu_name);
    return std::nullopt;
  }
  /**
   * @brief Get IMU orientation in world frame (w, x, y, z).
   *
   * @param info Arguments including the name of the IMU.
   * @return The orientation of the IMU in world frame represented as a unit quaternion (w,x,y,z).
   */
  virtual std::optional<Quaternion> imuOrientationW(const ImuOrientationWInfo& info) const {
    LOG_STREAM(ERROR, "imuOrientationW() not implemented for IMU: " << info.imu_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of body position in world frame.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info Arguments including the articulation name and the body name.
   */
  virtual bool initBodyPositionW(const BodyPositionWInfo& info) {
    LOG_STREAM(ERROR, "initBodyPositionW() not implemented for articulation: "
                          << info.articulation_name << ", body: " << info.body_name);
    return false;
  }
  /**
   * @brief Initialize data source of body orientation in base frame (w, x, y, z).
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info Arguments including the articulation name and the body name.
   */
  virtual bool initBodyOrientationW(const BodyOrientationWInfo& info) {
    LOG_STREAM(ERROR, "initBodyOrientationW() not implemented for articulation: "
                          << info.articulation_name << ", body: " << info.body_name);
    return false;
  }
  /**
   * @brief Initialize data source of body linear velocity in body frame.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info Arguments including the articulation name and the body name.
   */
  virtual bool initBodyLinearVelocityB(const BodyLinearVelocityBInfo& info) {
    LOG_STREAM(ERROR, "initBodyLinearVelocityB() not implemented for articulation: "
                          << info.articulation_name << ", body: " << info.body_name);
    return false;
  }
  /**
   * @brief Initialize data source of body angular velocity in body frame.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info Arguments including the articulation name and the body name.
   */
  virtual bool initBodyAngularVelocityB(const BodyAngularVelocityBInfo& info) {
    LOG_STREAM(ERROR, "initBodyAngularVelocityB() not implemented for articulation: "
                          << info.articulation_name << ", body: " << info.body_name);
    return false;
  }
  /**
   * @brief Get body position in world frame.
   *
   * @param info Arguments including the articulation name and the body name.
   * @return The position of the body in world frame.
   */
  virtual std::optional<Position> bodyPositionW(const BodyPositionWInfo& info) const {
    LOG_STREAM(ERROR, "bodyPositionW() not implemented for articulation: "
                          << info.articulation_name << ", body: " << info.body_name);
    return std::nullopt;
  }
  /**
   * @brief Get body orientation in world frame (w, x, y, z).
   *
   * @param info Arguments including the articulation name and the body name.
   * @return The orientation of the body in world frame.
   */
  virtual std::optional<Quaternion> bodyOrientationW(const BodyOrientationWInfo& info) const {
    LOG_STREAM(ERROR, "bodyOrientationW() not implemented for articulation: "
                          << info.articulation_name << ", body: " << info.body_name);
    return std::nullopt;
  }
  /**
   * @brief Get body linear velocity in body frame.
   *
   * @param info Arguments including the articulation name and the body name.
   * @return The linear velocity of the body in body frame.
   */
  virtual std::optional<LinearVelocity> bodyLinearVelocityB(
      const BodyLinearVelocityBInfo& info) const {
    LOG_STREAM(ERROR, "bodyLinearVelocityB() not implemented for articulation: "
                          << info.articulation_name << ", body: " << info.body_name);
    return std::nullopt;
  }
  /**
   * @brief Get body angular velocity in body frame.
   *
   * @param info Arguments including the articulation name and the body name.
   * @return The angular velocity of the body in body frame.
   */
  virtual std::optional<AngularVelocity> bodyAngularVelocityB(
      const BodyAngularVelocityBInfo& info) const {
    LOG_STREAM(ERROR, "bodyAngularVelocityB() not implemented for articulation: "
                          << info.articulation_name << ", body: " << info.body_name);
    return std::nullopt;
  }
  /**
   * @brief Initialize data source of the heightscan.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info The sensor info, including the sensor name, grid pattern, and layer names.
   */
  virtual bool initHeightScan(const HeightScanInfo& /*info*/) {
    LOG_STREAM(ERROR, "initHeightScan() not implemented");
    return false;
  }
  /**
   * @brief Get subsampled heightscan in base frame.
   *
   * This function returns the flattened (column-major), subsampled heightscan with the pattern
   * specified in the init function. The pattern is centered around the base.
   *
   * @param info The sensor info, including the sensor name and layer names.
   * @param base_pos_w The current base position in world frame.
   * @param base_quat_w The current base orientation in world frame.
   * @return A vector of heightscans.
   */
  virtual std::optional<const HeightScan*> heightScan(const HeightScanInfo& /*info*/,
                                                      const Position& /*base_pos_w*/,
                                                      const Quaternion& /*base_quat_w*/) {
    LOG_STREAM(ERROR, "heightScan() not implemented");
    return std::nullopt;
  }
  /**
   * @brief Initialize the spherical image data source.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info The sensor info, including the sensor name, resolution, field of view, and channel
   *             names.
   */
  virtual bool initSphericalImage(const SphericalImageInfo& /*info*/) {
    LOG_STREAM(ERROR, "initSphericalImage() not implemented");
    return false;
  }
  /**
   * @brief Get the spherical image with multiple channels.
   *
   * This function returns the multi-channel spherical image data, with each channel
   * flattened in row-major storage order.
   *
   * @param info The sensor info, including the sensor name and the channel names to retrieve.
   * @return Pointer to the spherical image data.
   */
  virtual std::optional<const MultiChannelImage*> sphericalImage(
      const SphericalImageInfo& /*info*/) {
    LOG_STREAM(ERROR, "sphericalImage() not implemented");
    return std::nullopt;
  }
  /**
   * @brief Initialize the pinhole image data source.
   *
   * Called once during initialization (usually non real-time).
   *
   * @param info The sensor info, including the sensor name, image dimensions, intrinsics, and
   *             channel names.
   */
  virtual bool initPinholeImage(const PinholeImageInfo& /*info*/) {
    LOG_STREAM(ERROR, "initPinholeImage() not implemented");
    return false;
  }
  /**
   * @brief Get the pinhole image with multiple channels.
   *
   * This function returns the multi-channel pinhole image data, with each channel
   * flattened in row-major storage order.
   *
   * @param info The sensor info, including the sensor name and the channel names to retrieve.
   * @return Pointer to the pinhole image data.
   */
  virtual std::optional<const MultiChannelImage*> pinholeImage(const PinholeImageInfo& /*info*/) {
    LOG_STREAM(ERROR, "pinholeImage() not implemented");
    return std::nullopt;
  }
};

}  // namespace exploy::control
