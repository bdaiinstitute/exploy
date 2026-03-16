// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <optional>
#include <regex>
#include <string>

#include "command_interface.hpp"
#include "metadata.hpp"
#include "onnx_runtime.hpp"
#include "state_interface.hpp"

namespace exploy::control {

/**
 * @brief Abstract base class for input components that read data into ONNX model inputs.
 *
 * Input components read robot state, sensor data, or commands and populate ONNX runtime
 * input tensors. Subclasses implement specific data sources (joints, IMU, cameras, etc.).
 */
struct Input {
  virtual ~Input() = default;

  /**
   * @brief Initialize the input component (non-real-time).
   *
   * Called once during controller setup to register data sources and configure parameters.
   *
   * @param state Robot state interface for reading sensor and state data.
   * @param command Command interface for reading external commands.
   * @return true if initialization succeeded, false otherwise.
   */
  virtual bool init(RobotStateInterface& /*state*/, CommandInterface& /*command*/) { return true; }

  /**
   * @brief Read data from robot/command interface into ONNX input buffer (real-time).
   *
   * Called every control loop to update the ONNX model inputs with current data.
   *
   * @param runtime ONNX runtime containing input/output buffers.
   * @param state Robot state interface for reading sensor and state data.
   * @param command Command interface for reading external commands.
   * @return true if read succeeded, false if data unavailable.
   */
  virtual bool read(OnnxRuntime& runtime, RobotStateInterface& state,
                    CommandInterface& command) = 0;
};

/**
 * @brief Abstract base class for output components that write ONNX model outputs.
 *
 * Output components read ONNX runtime output tensors and write commands to robot
 * control interfaces. Subclasses implement specific output targets (joint targets,
 * velocity commands, memory buffers, etc.).
 */
struct Output {
  virtual ~Output() = default;

  /**
   * @brief Initialize the output component (non-real-time).
   *
   * Called once during controller setup to configure output targets.
   *
   * @param state Robot state interface for accessing robot configuration.
   * @param command Command interface for configuring output targets.
   * @return true if initialization succeeded, false otherwise.
   */
  virtual bool init(RobotStateInterface& /*state*/, CommandInterface& /*command*/) { return true; }

  /**
   * @brief Write ONNX output buffer to robot/command interface (real-time).
   *
   * Called every control loop after ONNX inference to apply model outputs.
   *
   * @param runtime ONNX runtime containing input/output buffers.
   * @param state Robot state interface for writing state commands.
   * @param command Command interface for writing control commands.
   * @return true if write succeeded, false otherwise.
   */
  virtual bool write(OnnxRuntime& runtime, RobotStateInterface& state,
                     CommandInterface& command) = 0;
};

/**
 * @brief Input component that reads joint positions.
 *
 * Reads position values for specified joints from the robot state interface and
 * copies them to the ONNX input buffer. Joint order in the buffer matches the
 * order specified in joint_names.
 */
class JointPositionInput : public Input {
 public:
  /**
   * @brief Construct a joint position input component.
   *
   * @param key ONNX input tensor name (e.g., "robot.joints.pos").
   * @param joint_names Vector of joint names to read, in buffer order.
   */
  JointPositionInput(const std::string& key, const std::vector<std::string>& joint_names);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                       ///< ONNX input tensor name.
  std::vector<std::string> joint_names_;  ///< Joint names to read.
};

/**
 * @brief Input component that reads joint velocities.
 *
 * Reads velocity values for specified joints from the robot state interface and
 * copies them to the ONNX input buffer. Joint order in the buffer matches the
 * order specified in joint_names.
 */
class JointVelocityInput : public Input {
 public:
  /**
   * @brief Construct a joint velocity input component.
   *
   * @param key ONNX input tensor name (e.g., "robot.joints.vel").
   * @param joint_names Vector of joint names to read, in buffer order.
   */
  JointVelocityInput(const std::string& key, const std::vector<std::string>& joint_names);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                       ///< ONNX input tensor name.
  std::vector<std::string> joint_names_;  ///< Joint names to read.
};

/**
 * @brief Input component that reads robot base position in world frame.
 *
 * Reads the robot base position as a 3D vector (x, y, z) in world coordinates
 * and copies it to the ONNX input buffer.
 */
class BasePositionInput : public Input {
 public:
  /**
   * @brief Construct a base position input component.
   *
   * @param key ONNX input tensor name (e.g., "robot.base.pos").
   */
  BasePositionInput(const std::string& key);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;  ///< ONNX input tensor name.
};

/**
 * @brief Input component that reads robot base orientation in world frame.
 *
 * Reads the robot base orientation as a quaternion (w, x, y, z) in world coordinates
 * and copies it to the ONNX input buffer.
 */
class BaseOrientationInput : public Input {
 public:
  /**
   * @brief Construct a base orientation input component.
   *
   * @param key ONNX input tensor name (e.g., "robot.base.quat").
   */
  BaseOrientationInput(const std::string& key);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;  ///< ONNX input tensor name.
};

/**
 * @brief Input component that reads robot base linear velocity.
 *
 * Reads the robot base linear velocity as a 3D vector (vx, vy, vz) expressed
 * in the base frame and copies it to the ONNX input buffer.
 */
class BaseLinearVelocityInput : public Input {
 public:
  /**
   * @brief Construct a base linear velocity input component.
   *
   * @param key ONNX input tensor name (e.g., "robot.base.lin_vel").
   */
  BaseLinearVelocityInput(const std::string& key);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;  ///< ONNX input tensor name.
};

/**
 * @brief Input component that reads robot base angular velocity.
 *
 * Reads the robot base angular velocity as a 3D vector (wx, wy, wz) expressed
 * in the base frame and copies it to the ONNX input buffer.
 */
class BaseAngularVelocityInput : public Input {
 public:
  /**
   * @brief Construct a base angular velocity input component.
   *
   * @param key ONNX input tensor name (e.g., "robot.base.ang_vel").
   */
  BaseAngularVelocityInput(const std::string& key);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;  ///< ONNX input tensor name.
};

/**
 * @brief Input component that reads angular velocity from a specific IMU sensor.
 *
 * Reads angular velocity measurements from a named IMU sensor and copies the
 * 3D vector (wx, wy, wz) to the ONNX input buffer.
 */
class IMUAngularVelocityInput : public Input {
 public:
  /**
   * @brief Construct an IMU angular velocity input component.
   *
   * @param key ONNX input tensor name (e.g., "sensors.imu.ang_vel").
   * @param imu_name Name of the IMU sensor to read from.
   */
  IMUAngularVelocityInput(const std::string& key, const std::string& imu_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;       ///< ONNX input tensor name.
  std::string imu_name_;  ///< IMU sensor name.
};

/**
 * @brief Input component that reads orientation from a specific IMU sensor.
 *
 * Reads orientation measurements from a named IMU sensor and copies the
 * quaternion (w, x, y, z) to the ONNX input buffer.
 */
class IMUOrientationInput : public Input {
 public:
  /**
   * @brief Construct an IMU orientation input component.
   *
   * @param key ONNX input tensor name (e.g., "sensors.imu.quat").
   * @param imu_name Name of the IMU sensor to read from.
   */
  IMUOrientationInput(const std::string& key, const std::string& imu_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;       ///< ONNX input tensor name.
  std::string imu_name_;  ///< IMU sensor name.
};

/**
 * @brief Input component that reads terrain height scan data with multiple layers.
 *
 * Reads height scan data from a named sensor with configurable layers (e.g., height,
 * normals, color). Each layer is written to a separate ONNX input buffer with the
 * naming pattern "key.layer_name".
 */
class HeightScanInput : public Input {
 public:
  /**
   * @brief Construct a height scan input component.
   *
   * @param key ONNX input tensor base name (e.g., "sensors.height_scan").
   * @param sensor_name Name of the height scan sensor to read from.
   * @param layer_names Set of layer names to include (each creates a buffer "key.layer_name").
   * @param metadata Height scan metadata containing grid pattern, resolution, and size.
   */
  HeightScanInput(const std::string& key, const std::string& sensor_name,
                  const std::unordered_set<std::string>& layer_names,
                  const metadata::HeightScanMetadata& metadata);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                              ///< ONNX input tensor base name.
  std::string sensor_name_;                      ///< Height scan sensor name.
  std::unordered_set<std::string> layer_names_;  ///< Layer names to read.
  metadata::HeightScanMetadata metadata_;        ///< Height scan configuration.
};

/**
 * @brief Input component that reads range/depth image from LiDAR sensor.
 *
 * Reads range image data (distances) from a LiDAR sensor and copies it to the
 * ONNX input buffer. Configuration includes resolution, field of view, and
 * sentinel value for unobserved points.
 */
class RangeImageInput : public Input {
 public:
  /**
   * @brief Construct a range image input component.
   *
   * @param key ONNX input tensor name (e.g., "sensors.lidar.range").
   * @param metadata Range image metadata containing resolution, FOV, and sentinel value.
   */
  RangeImageInput(const std::string& key, const metadata::RangeImageMetadata& metadata);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                        ///< ONNX input tensor name.
  metadata::RangeImageMetadata metadata_;  ///< Range image configuration.
};

/**
 * @brief Input component that reads depth image from camera sensor.
 *
 * Reads depth image data from a camera sensor and copies it to the ONNX input
 * buffer. Configuration includes image dimensions and camera intrinsic parameters.
 */
class DepthImageInput : public Input {
 public:
  /**
   * @brief Construct a depth image input component.
   *
   * @param key ONNX input tensor name (e.g., "sensors.camera.depth").
   * @param metadata Depth image metadata containing width, height, and camera intrinsics.
   */
  DepthImageInput(const std::string& key, const metadata::DepthImageMetadata& metadata);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                        ///< ONNX input tensor name.
  metadata::DepthImageMetadata metadata_;  ///< Depth image configuration.
};

/**
 * @brief Input component that reads orientation of a specific rigid body.
 *
 * Reads the orientation of a named rigid body in world frame as a quaternion
 * (w, x, y, z) and copies it to the ONNX input buffer.
 */
class BodyOrientationInput : public Input {
 public:
  /**
   * @brief Construct a body orientation input component.
   *
   * @param key ONNX input tensor name (e.g., "robot.body.quat").
   * @param body_name Name of the rigid body to read orientation from.
   */
  BodyOrientationInput(const std::string& key, const std::string& body_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;        ///< ONNX input tensor name.
  std::string body_name_;  ///< Rigid body name.
};

/**
 * @brief Input component that reads position of a specific rigid body.
 *
 * Reads the position of a named rigid body in world frame as a 3D vector
 * (x, y, z) and copies it to the ONNX input buffer.
 */
class BodyPositionInput : public Input {
 public:
  /**
   * @brief Construct a body position input component.
   *
   * @param key ONNX input tensor name (e.g., "robot.body.pos").
   * @param body_name Name of the rigid body to read position from.
   */
  BodyPositionInput(const std::string& key, const std::string& body_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;        ///< ONNX input tensor name.
  std::string body_name_;  ///< Rigid body name.
};

/**
 * @brief Input component that reads commanded SE(3) pose.
 *
 * Reads an external SE(3) pose command (position + orientation) and copies it
 * to the ONNX input buffer as 7 values: [x, y, z, qw, qx, qy, qz].
 */
class CommandSE3PoseInput : public Input {
 public:
  /**
   * @brief Construct a commanded SE(3) pose input component.
   *
   * @param key ONNX input tensor name (e.g., "commands.target_pose").
   * @param command_name Name of the SE(3) pose command to read.
   */
  CommandSE3PoseInput(const std::string& key, const std::string& command_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;           ///< ONNX input tensor name.
  std::string command_name_;  ///< SE(3) pose command name.
};

/**
 * @brief Input component that reads commanded planar (SE(2)) velocity.
 *
 * Reads an external planar velocity command (vx, vy, wz) and copies it to the
 * ONNX input buffer. Supports optional velocity range constraints.
 */
class CommandSE2VelocityInput : public Input {
 public:
  /**
   * @brief Construct a commanded SE(2) velocity input component.
   *
   * @param key ONNX input tensor name (e.g., "commands.base_velocity").
   * @param command_name Name of the SE(2) velocity command to read.
   * @param metadata SE(2) velocity metadata containing optional velocity ranges.
   */
  CommandSE2VelocityInput(const std::string& key, const std::string& command_name,
                          const metadata::SE2VelocityCommandMetadata& metadata);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                                ///< ONNX input tensor name.
  std::string command_name_;                       ///< SE(2) velocity command name.
  metadata::SE2VelocityCommandMetadata metadata_;  ///< Velocity command configuration.
};

/**
 * @brief Input component that reads a boolean command value.
 *
 * Reads an external boolean command (e.g., enable/disable flag) and copies it
 * to the ONNX input buffer as a scalar value.
 */
class CommandBooleanInput : public Input {
 public:
  /**
   * @brief Construct a boolean command input component.
   *
   * @param key ONNX input tensor name (e.g., "commands.enable_mode").
   * @param command_name Name of the boolean command to read.
   */
  CommandBooleanInput(const std::string& key, const std::string& command_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;           ///< ONNX input tensor name.
  std::string command_name_;  ///< Boolean command name.
};

/**
 * @brief Input component that reads a floating-point command value.
 *
 * Reads an external float command (e.g., speed scale, gain parameter) and
 * copies it to the ONNX input buffer as a scalar value.
 */
class CommandFloatInput : public Input {
 public:
  /**
   * @brief Construct a float command input component.
   *
   * @param key ONNX input tensor name (e.g., "commands.speed_scale").
   * @param command_name Name of the float command to read.
   * @param metadata Float command metadata containing optional value range.
   */
  CommandFloatInput(const std::string& key, const std::string& command_name,
                    const metadata::FloatCommandMetadata& metadata);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                          ///< ONNX input tensor name.
  std::string command_name_;                 ///< Float command name.
  metadata::FloatCommandMetadata metadata_;  ///< Float command configuration.
};

/**
 * @brief Input component that maintains and increments a step counter.
 *
 * Provides a monotonically increasing step count to the ONNX model for tracking
 * policy iterations. The counter is incremented on each read() call.
 */
class StepCountInput : public Input {
 public:
  /**
   * @brief Construct a step count input component.
   *
   * @param key ONNX input tensor name (typically "ctx.step_count").
   */
  StepCountInput(const std::string& key);

  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;  ///< ONNX input tensor name.
};

/**
 * @brief Output component that writes joint target positions, velocities, and efforts.
 *
 * Reads ONNX output buffers for joint targets and writes them to the robot control
 * interface along with PD controller gains (stiffness and damping) specified in metadata.
 */
class JointTargetOutput : public Output {
 public:
  /**
   * @brief Construct a joint target output component.
   *
   * @param pos_key ONNX output tensor name for target positions.
   * @param vel_key ONNX output tensor name for target velocities.
   * @param eff_key ONNX output tensor name for target efforts (feedforward torques).
   * @param metadata Joint output metadata containing joint names, stiffness, and damping.
   */
  JointTargetOutput(const std::string& pos_key, const std::string& vel_key,
                    const std::string& eff_key, const metadata::JointOutputMetadata& metadata);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool write(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string pos_key_;                     ///< ONNX output tensor name for positions.
  std::string vel_key_;                     ///< ONNX output tensor name for velocities.
  std::string eff_key_;                     ///< ONNX output tensor name for efforts.
  metadata::JointOutputMetadata metadata_;  ///< Joint output configuration.
};

/**
 * @brief Output component that writes planar (SE(2)) velocity commands.
 *
 * Reads ONNX output buffer containing planar velocity (vx, vy, wz) and writes
 * it to a target frame on the robot control interface.
 */
class SE2VelocityOutput : public Output {
 public:
  /**
   * @brief Construct an SE(2) velocity output component.
   *
   * @param key ONNX output tensor name (expects 3 values: vx, vy, wz).
   * @param metadata SE(2) velocity output metadata containing target frame name.
   */
  SE2VelocityOutput(const std::string& key, const metadata::Se2VelocityOutputMetadata& metadata);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool write(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                               ///< ONNX output tensor name.
  metadata::Se2VelocityOutputMetadata metadata_;  ///< Velocity output configuration.
};

/**
 * @brief Output component that maintains memory state for recurrent policies.
 *
 * Copies ONNX output buffer "memory.{key}.out" to input buffer "memory.{key}.in"
 * to maintain hidden state across control steps for stateful policies (e.g., LSTMs).
 */
class MemoryOutput : public Output {
 public:
  /**
   * @brief Construct a memory output component.
   *
   * @param key Base name for memory tensors (creates "memory.{key}.in" and "memory.{key}.out").
   */
  MemoryOutput(const std::string& key);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool write(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;  ///< Base name for memory tensors.
};

}  // namespace exploy::control
