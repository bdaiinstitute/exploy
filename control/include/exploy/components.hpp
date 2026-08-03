// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "exploy/command_interface.hpp"
#include "exploy/metadata.hpp"
#include "exploy/onnx_runtime.hpp"
#include "exploy/state_interface.hpp"

namespace exploy::control {

/**
 * @brief Abstract base class for input components that read data into ONNX model inputs.
 *
 * Input components read robot state, sensor data, or commands and populate ONNX runtime
 * input tensors. Subclasses implement specific data sources (joints, IMU, cameras, etc.).
 */
struct Input {
  /**
   * @brief Construct an input component.
   *
   * @param name Human-readable component type name (used in logs).
   * @param tensor_names ONNX input tensor name(s) this component populates. Used to verify
   * that every ONNX input tensor is served by exactly one component.
   */
  Input(std::string name, std::unordered_set<std::string> tensor_names)
      : name_(std::move(name)), tensor_names_(std::move(tensor_names)) {}
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

  /**
   * @brief Get the name of this component.
   * @return Name of this component.
   */
  virtual const std::string& getName() const { return name_; }

  /**
   * @brief Get the ONNX input tensor name(s) this component populates.
   *
   * Used to verify that every ONNX input tensor is served by exactly one component.
   *
   * @return Names of the ONNX tensors this component serves.
   */
  const std::unordered_set<std::string>& tensorNames() const { return tensor_names_; }

 private:
  std::string name_;                              ///< Name of this component.
  std::unordered_set<std::string> tensor_names_;  ///< ONNX tensor names served by this component.
};

/**
 * @brief Abstract base class for output components that write ONNX model outputs.
 *
 * Output components read ONNX runtime output tensors and write commands to robot
 * control interfaces. Subclasses implement specific output targets (joint targets,
 * velocity commands, memory buffers, etc.).
 */
struct Output {
  /**
   * @brief Construct an output component.
   *
   * @param name Human-readable component type name (used in logs).
   * @param tensor_names ONNX output tensor name(s) this component serves. Used to verify
   * that every ONNX output tensor is served by exactly one component.
   */
  Output(std::string name, std::unordered_set<std::string> tensor_names)
      : name_(std::move(name)), tensor_names_(std::move(tensor_names)) {}
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

  /**
   * @brief Get the name of this component.
   * @return Name of this component.
   */
  virtual const std::string& getName() const { return name_; }

  /**
   * @brief Get the ONNX tensor name(s) this component is responsible for.
   *
   * Used to verify that every ONNX tensor is served by exactly one component. The set may include
   * any ONNX tensor names the component owns (for example MemoryOutput owns both an input and an
   * output tensor), not just output tensors.
   *
   * @return Names of the ONNX tensors this component serves.
   */
  const std::unordered_set<std::string>& tensorNames() const { return tensor_names_; }

 private:
  std::string name_;                              ///< Name of this component.
  std::unordered_set<std::string> tensor_names_;  ///< ONNX tensor names served by this component.
};

/**
 * @brief Abstract base class for read-only observer components.
 *
 * Observer components inspect ONNX runtime buffers without modifying them, e.g. for
 * debugging or visualization. Read-only access is enforced by the type system.
 * Observers are created by matchers via `Matcher::createObservers()` /
 * `GroupMatcher::createObservers()` and run once per control step after inference.
 */
struct Observer {
  explicit Observer(const std::string& name) : name_(name) {}
  virtual ~Observer() = default;

  /**
   * @brief Initialize the observer component (non-real-time).
   *
   * Called once during controller setup.
   *
   * @return true if initialization succeeded, false otherwise.
   */
  virtual bool init() { return true; }

  /**
   * @brief Inspect ONNX buffers without modifying them (real-time).
   *
   * @param runtime ONNX runtime containing input/output buffers (read-only).
   */
  virtual void observe(const OnnxRuntime& runtime) const = 0;

  /**
   * @brief Get the name of this component.
   * @return Name of this component.
   */
  virtual const std::string& getName() const { return name_; }

 private:
  std::string name_;  ///< Name of this component.
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
   * @param articulation_name Name of the articulation the joints belong to.
   * @param joint_names Vector of joint names to read, in buffer order.
   */
  JointPositionInput(const std::string& key, const std::string& articulation_name,
                     const std::vector<std::string>& joint_names);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                       ///< ONNX input tensor name.
  std::string articulation_name_;         ///< Articulation name.
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
   * @param articulation_name Name of the articulation the joints belong to.
   * @param joint_names Vector of joint names to read, in buffer order.
   */
  JointVelocityInput(const std::string& key, const std::string& articulation_name,
                     const std::vector<std::string>& joint_names);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                       ///< ONNX input tensor name.
  std::string articulation_name_;         ///< Articulation name.
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
   * @param articulation_name Name of the articulation.
   */
  BasePositionInput(const std::string& key, const std::string& articulation_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                ///< ONNX input tensor name.
  std::string articulation_name_;  ///< Articulation name.
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
   * @param articulation_name Name of the articulation.
   */
  BaseOrientationInput(const std::string& key, const std::string& articulation_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                ///< ONNX input tensor name.
  std::string articulation_name_;  ///< Articulation name.
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
   * @param articulation_name Name of the articulation.
   */
  BaseLinearVelocityInput(const std::string& key, const std::string& articulation_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                ///< ONNX input tensor name.
  std::string articulation_name_;  ///< Articulation name.
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
   * @param articulation_name Name of the articulation.
   */
  BaseAngularVelocityInput(const std::string& key, const std::string& articulation_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                ///< ONNX input tensor name.
  std::string articulation_name_;  ///< Articulation name.
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
 * @brief Input component that reads linear velocity from a specific IMU sensor.
 *
 * Reads linear velocity measurements from a named IMU sensor and copies the
 * 3D vector (vx, vy, vz) to the ONNX input buffer.
 */
class IMULinearVelocityInput : public Input {
 public:
  /**
   * @brief Construct an IMU linear velocity input component.
   *
   * @param key ONNX input tensor name (e.g., "sensor.imu.imu_pelvis.lin_vel_b_rt_w_in_b").
   * @param imu_name Name of the IMU sensor to read from.
   */
  IMULinearVelocityInput(const std::string& key, const std::string& imu_name);

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
  std::string key_;                ///< ONNX input tensor base name.
  std::string articulation_name_;  ///< Articulation providing the base frame for the scan.
  HeightScanInfo scan_info_;       ///< Cached scan info (sensor, pattern, layers).
};

/**
 * @brief Input component that reads spherical image data with multiple channels.
 *
 * Reads spherical image data from a named sensor with configurable channels (e.g., range,
 * risk). Each channel is written to a separate ONNX input buffer with the
 * naming pattern "key.channel_name".
 */
class SphericalImageInput : public Input {
 public:
  /**
   * @brief Construct a spherical image input component.
   *
   * @param key ONNX input tensor base name (e.g., "sensor.spherical_image.lidar1").
   * @param sensor_name Name of the spherical image sensor to read from.
   * @param channel_names Set of channel names to include (each creates a buffer
   * "key.channel_name").
   * @param metadata Spherical image metadata containing resolution, FOV, and sentinel value.
   */
  SphericalImageInput(const std::string& key, const std::string& sensor_name,
                      const std::unordered_set<std::string>& channel_names,
                      const metadata::SphericalImageMetadata& metadata);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;          ///< ONNX input tensor base name.
  SphericalImageInfo info_;  ///< Cached spherical image info (sensor, intrinsics, channels).
};

/**
 * @brief Input component that reads pinhole camera image data with multiple channels.
 *
 * Reads pinhole image data from a named sensor with configurable channels (e.g., depth,
 * risk). Each channel is written to a separate ONNX input buffer with the
 * naming pattern "key.channel_name".
 */
class PinholeImageInput : public Input {
 public:
  /**
   * @brief Construct a pinhole image input component.
   *
   * @param key ONNX input tensor base name (e.g., "sensor.pinhole_image.cam1").
   * @param sensor_name Name of the pinhole image sensor to read from.
   * @param channel_names Set of channel names to include (each creates a buffer
   * "key.channel_name").
   * @param metadata Pinhole image metadata containing width, height, and camera intrinsics.
   */
  PinholeImageInput(const std::string& key, const std::string& sensor_name,
                    const std::unordered_set<std::string>& channel_names,
                    const metadata::PinholeImageMetadata& metadata);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;        ///< ONNX input tensor base name.
  PinholeImageInfo info_;  ///< Cached pinhole image info (sensor, intrinsics, channels).
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
   * @param articulation_name Name of the articulation the body belongs to.
   * @param body_name Name of the rigid body to read orientation from.
   */
  BodyOrientationInput(const std::string& key, const std::string& articulation_name,
                       const std::string& body_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                ///< ONNX input tensor name.
  std::string articulation_name_;  ///< Articulation name.
  std::string body_name_;          ///< Rigid body name.
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
   * @param articulation_name Name of the articulation the body belongs to.
   * @param body_name Name of the rigid body to read position from.
   */
  BodyPositionInput(const std::string& key, const std::string& articulation_name,
                    const std::string& body_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                ///< ONNX input tensor name.
  std::string articulation_name_;  ///< Articulation name.
  std::string body_name_;          ///< Rigid body name.
};

/**
 * @brief Input component that reads linear velocity of a specific rigid body.
 *
 * Reads the linear velocity of a named rigid body as a 3D vector (vx, vy, vz)
 * and copies it to the ONNX input buffer.
 */
class BodyLinearVelocityInput : public Input {
 public:
  /**
   * @brief Construct a body linear velocity input component.
   *
   * @param key ONNX input tensor name (e.g., "obj.object.box.lin_vel_b_rt_w_in_b").
   * @param articulation_name Name of the articulation the body belongs to.
   * @param body_name Name of the rigid body to read linear velocity from.
   */
  BodyLinearVelocityInput(const std::string& key, const std::string& articulation_name,
                          const std::string& body_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                ///< ONNX input tensor name.
  std::string articulation_name_;  ///< Articulation name.
  std::string body_name_;          ///< Rigid body name.
};

/**
 * @brief Input component that reads angular velocity of a specific rigid body.
 *
 * Reads the angular velocity of a named rigid body as a 3D vector (wx, wy, wz)
 * and copies it to the ONNX input buffer.
 */
class BodyAngularVelocityInput : public Input {
 public:
  /**
   * @brief Construct a body angular velocity input component.
   *
   * @param key ONNX input tensor name (e.g., "obj.object.box.ang_vel_b_rt_w_in_b").
   * @param articulation_name Name of the articulation the body belongs to.
   * @param body_name Name of the rigid body to read angular velocity from.
   */
  BodyAngularVelocityInput(const std::string& key, const std::string& articulation_name,
                           const std::string& body_name);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                ///< ONNX input tensor name.
  std::string articulation_name_;  ///< Articulation name.
  std::string body_name_;          ///< Rigid body name.
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
 * @brief Input component that reads commanded joint positions.
 *
 * Reads the commanded position for each joint specified in the metadata by calling
 * CommandInterface::jointPosition() once per joint, then copies the values to the
 * ONNX input buffer in the order defined by the metadata joint names.
 */
class CommandJointPositionInput : public Input {
 public:
  /**
   * @brief Construct a joint position command input component.
   *
   * @param key ONNX input tensor name (e.g., "cmd.joint_pos.arm").
   * @param command_name Name of the joint position command to read.
   * @param metadata Metadata specifying the ordered list of joint names.
   */
  CommandJointPositionInput(const std::string& key, const std::string& command_name,
                            const metadata::JointPositionCommandMetadata& metadata);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool read(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string key_;                                  ///< ONNX input tensor name.
  std::string command_name_;                         ///< Joint position command name.
  metadata::JointPositionCommandMetadata metadata_;  ///< Command configuration.
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
   * @param articulation_name Name of the articulation to write joint targets to.
   * @param metadata Joint output metadata containing joint names, stiffness, and damping.
   */
  JointTargetOutput(const std::string& pos_key, const std::string& vel_key,
                    const std::string& eff_key, const std::string& articulation_name,
                    const metadata::JointOutputMetadata& metadata);

  bool init(RobotStateInterface& state, CommandInterface& command) override;
  bool write(OnnxRuntime& runtime, RobotStateInterface& state, CommandInterface& command) override;

 private:
  std::string pos_key_;                     ///< ONNX output tensor name for positions.
  std::string vel_key_;                     ///< ONNX output tensor name for velocities.
  std::string eff_key_;                     ///< ONNX output tensor name for efforts.
  std::string articulation_name_;           ///< Articulation name.
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
