#pragma once

#include <optional>
#include <regex>
#include <string>

#include "command_interface.hpp"
#include "onnx_metadata.hpp"
#include "onnx_runtime.hpp"
#include "state_interface.hpp"

namespace rai::cs::control::common::onnx {

struct Input {
  virtual ~Input() = default;
  virtual bool init(RobotStateInterface& /*state*/, CommandInterface& /*command*/) { return true; }
  virtual bool read(OnnxRuntime& runtime, RobotStateInterface& state,
                    CommandInterface& command) = 0;
};

struct Output {
  virtual ~Output() = default;
  virtual bool init(RobotStateInterface& /*state*/, CommandInterface& /*command*/) { return true; }
  virtual bool write(OnnxRuntime& runtime, RobotStateInterface& state,
                     CommandInterface& command) = 0;
};

class JointPositionInput : public Input {
 public:
  JointPositionInput(const std::string& key, const std::vector<std::string>& joint_names);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  std::vector<std::string> joint_names_;
};

class JointVelocityInput : public Input {
 public:
  JointVelocityInput(const std::string& key, const std::vector<std::string>& joint_names);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  std::vector<std::string> joint_names_;
};

class BasePositionInput : public Input {
 public:
  BasePositionInput(const std::string& key);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
};

class BaseOrientationInput : public Input {
 public:
  BaseOrientationInput(const std::string& key);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
};

class BaseLinearVelocityInput : public Input {
 public:
  BaseLinearVelocityInput(const std::string& key);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
};

class BaseAngularVelocityInput : public Input {
 public:
  BaseAngularVelocityInput(const std::string& key);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
};

class IMUAngularVelocityInput : public Input {
 public:
  IMUAngularVelocityInput(const std::string& key, const std::string& imu_name);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  std::string imu_name_;
};

class IMUOrientationInput : public Input {
 public:
  IMUOrientationInput(const std::string& key, const std::string& imu_name);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  std::string imu_name_;
};

class HeightScanInput : public Input {
 public:
  HeightScanInput(const std::string& key, const std::string& sensor_name,
                  const std::unordered_set<std::string>& layer_names,
                  const metadata::HeightScanMetadata& metadata);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  std::string sensor_name_;
  std::unordered_set<std::string> layer_names_;
  metadata::HeightScanMetadata metadata_;
};

class RangeImageInput : public Input {
 public:
  RangeImageInput(const std::string& key, const metadata::RangeImageMetadata& metadata);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  metadata::RangeImageMetadata metadata_;
};

class DepthImageInput : public Input {
 public:
  DepthImageInput(const std::string& key, const metadata::DepthImageMetadata& metadata);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  metadata::DepthImageMetadata metadata_;
};

class BodyOrientationInput : public Input {
 public:
  BodyOrientationInput(const std::string& key, const std::string& body_name);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  std::string body_name_;
};

class BodyPositionInput : public Input {
 public:
  BodyPositionInput(const std::string& key, const std::string& body_name);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  std::string body_name_;
};

class CommandSE3PoseInput : public Input {
 public:
  CommandSE3PoseInput(const std::string& key, const std::string& command_name);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  std::string command_name_;
};

class CommandSE2VelocityInput : public Input {
 public:
  CommandSE2VelocityInput(const std::string& key, const std::string& command_name,
                          const metadata::SE2VelocityCommandMetadata& metadata);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  std::string command_name_;
  metadata::SE2VelocityCommandMetadata metadata_;
};

class CommandBooleanInput : public Input {
 public:
  CommandBooleanInput(const std::string& key, const std::string& command_name);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  std::string command_name_;
};

class CommandFloatInput : public Input {
 public:
  CommandFloatInput(const std::string& key, const std::string& command_name);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
  std::string command_name_;
};

class StepCountInput : public Input {
 public:
  StepCountInput(const std::string& key);
  bool read(OnnxRuntime& runtime, RobotStateInterface& state,
            CommandInterface& command) override;

 private:
  std::string key_;
};

class JointTargetOutput : public Output {
 public:
  JointTargetOutput(const std::string& pos_key, const std::string& vel_key,
                    const std::string& eff_key, const metadata::JointOutputMetadata& metadata);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool write(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string pos_key_;
  std::string vel_key_;
  std::string eff_key_;
  metadata::JointOutputMetadata metadata_;
};

class SE2VelocityOutput : public Output {
 public:
  SE2VelocityOutput(const std::string& key, const metadata::Se2VelocityOutputMetadata& metadata);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool write(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;
  metadata::Se2VelocityOutputMetadata metadata_;
};

class MemoryOutput : public Output {
 public:
  MemoryOutput(const std::string& key);
  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool write(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;
};

}  // namespace rai::cs::control::common::onnx
