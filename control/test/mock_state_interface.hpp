// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <gmock/gmock.h>
#include <unordered_set>
#include "state_interface.hpp"

namespace exploy::control {

class MockRobotStateInterface : public RobotStateInterface {
 public:
  MOCK_METHOD(bool, initBasePosW, (), (override));
  MOCK_METHOD(bool, initBaseQuatW, (), (override));
  MOCK_METHOD(bool, initBaseLinVelB, (), (override));
  MOCK_METHOD(bool, initBaseAngVelB, (), (override));
  MOCK_METHOD(std::optional<Position>, basePosW, (), (const, override));
  MOCK_METHOD(std::optional<Quaternion>, baseQuatW, (), (const, override));
  MOCK_METHOD(std::optional<LinearVelocity>, baseLinVelB, (), (const, override));
  MOCK_METHOD(std::optional<AngularVelocity>, baseAngVelB, (), (const, override));
  MOCK_METHOD(bool, initJointOutput, (const std::string& joint_name), (override));
  MOCK_METHOD(bool, initJointPosition, (const std::string& joint_name), (override));
  MOCK_METHOD(bool, initJointVelocity, (const std::string& joint_name), (override));
  MOCK_METHOD(bool, initJointEffort, (const std::string& joint_name), (override));
  MOCK_METHOD(std::optional<double>, jointPosition, (const std::string& joint_name),
              (const, override));
  MOCK_METHOD(std::optional<double>, jointVelocity, (const std::string& joint_name),
              (const, override));
  MOCK_METHOD(std::optional<double>, jointEffort, (const std::string& joint_name),
              (const, override));
  MOCK_METHOD(bool, setJointPosition, (const std::string& joint_name, double position), (override));
  MOCK_METHOD(bool, setJointVelocity, (const std::string& joint_name, double velocity), (override));
  MOCK_METHOD(bool, setJointEffort, (const std::string& joint_name, double effort), (override));
  MOCK_METHOD(bool, setJointPGain, (const std::string& joint_name, double p_gain), (override));
  MOCK_METHOD(bool, setJointDGain, (const std::string& joint_name, double d_gain), (override));
  MOCK_METHOD(bool, initSe2Velocity, (const std::string& frame_name), (override));
  MOCK_METHOD(bool, setSe2Velocity, (const std::string& frame_name, const SE2Velocity& velocity),
              (override));
  MOCK_METHOD(bool, initImuAngularVelocityImu, (const std::string& imu_name), (override));
  MOCK_METHOD(bool, initImuOrientationW, (const std::string& imu_name), (override));
  MOCK_METHOD(std::optional<AngularVelocity>, imuAngularVelocityImu, (const std::string& imu_name),
              (const, override));
  MOCK_METHOD(std::optional<Quaternion>, imuOrientationW, (const std::string& imu_name),
              (const, override));
  MOCK_METHOD(bool, initBodyOrientationW, (const std::string& body_name), (override));
  MOCK_METHOD(bool, initBodyPositionW, (const std::string& body_name), (override));
  MOCK_METHOD(bool, initBodyLinearVelocityB, (const std::string& body_name), (override));
  MOCK_METHOD(bool, initBodyAngularVelocityB, (const std::string& body_name), (override));
  MOCK_METHOD(std::optional<Position>, bodyPositionW, (const std::string& body_name),
              (const override));
  MOCK_METHOD(std::optional<Quaternion>, bodyOrientationW, (const std::string& body_name),
              (const override));
  MOCK_METHOD(std::optional<LinearVelocity>, bodyLinearVelocityB, (const std::string& body_name),
              (const override));
  MOCK_METHOD(std::optional<AngularVelocity>, bodyAngularVelocityB, (const std::string& body_name),
              (const override));
  MOCK_METHOD(bool, initHeightScan,
              (const std::string& sensor_name, const HeightScanConfig& config), (override));
  MOCK_METHOD(std::optional<HeightScan*>, heightScan,
              (const std::string& sensor_name, const std::unordered_set<std::string>& layer_names,
               const Position& base_pos_w, const Quaternion& base_quat_w),
              (override));
  MOCK_METHOD(bool, initRangeImage, (const RangeImageConfig& config), (override));
  MOCK_METHOD(std::optional<std::span<const float>>, rangeImage, (), (override));
  MOCK_METHOD(bool, initDepthImage, (const DepthImageConfig& config), (override));
  MOCK_METHOD(std::optional<std::span<const float>>, depthImage, (), (override));
};

}  // namespace exploy::control
