<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Controller Tutorial

This tutorial walks through deploying an exported ONNX policy on a robot using
`exploy::control`. Rather than calling the ONNX model directly, exploy wraps the
inference loop with three composable interfaces — robot state, commands, and data
collection — so the same `OnnxRLController` class works identically on hardware,
in a simulation, or in a unit test.

By the end of this tutorial you will know how to:

1. Implement `RobotStateInterface` to bridge your hardware or simulator.
2. Implement `CommandInterface` to feed velocity targets or other commands.
3. Implement `DataCollectionInterface` to record telemetry.
4. Load the ONNX model and run the closed-loop control cycle.

> **Note:** A complete, runnable version of the code in this tutorial is available in
> [`examples/controller/`](https://github.com/rai-opensource/exploy/tree/main/examples/controller/).
> This tutorial explains the details behind that example step by step.
>
> **See also:** If you haven't exported your model yet, start with the
> [Exporter Tutorial](../exporter/exporter_tutorial.md).

## Architecture Overview

`OnnxRLController::update()` performs one full cycle:

1. Reads observations from `RobotStateInterface` and `CommandInterface`.
2. Runs the ONNX policy.
3. Writes joint targets back through `RobotStateInterface`.
4. Calls `DataCollectionInterface::collectData()`.

All three interfaces are pure-virtual, so you wire in your own implementations at
construction time. The controller never depends on any particular middleware.

## Prerequisites

The tutorial uses the following headers:

```cpp
#include "controller.hpp"           // OnnxRLController
#include "state_interface.hpp"      // RobotStateInterface
#include "command_interface.hpp"    // CommandInterface
#include "data_collection_interface.hpp"  // DataCollectionInterface
#include "logging_interface.hpp"    // setLogger / StdoutLogger
#include "interfaces.hpp"           // SE2Velocity, Position, Quaternion, …
```

All headers live in `control/` and are pulled in by the `exploy_control` CMake target.

---

## Step 1 — Implement `RobotStateInterface`

`RobotStateInterface` is the bridge between the controller and your robot or
simulator. It has two responsibilities:

- **Read** sensor data (joint positions/velocities, IMU, base pose, body poses, …)
  so the ONNX policy can compute observations.
- **Write** joint targets (position, velocity, or effort) that the policy produces
  back to the actuator layer.

Each sensor source follows a two-step pattern: an `init*()` method is called once
during `controller.init()` to allocate resources or subscribe to topics, and a
getter method is called every cycle during `controller.update()` to return the
current value.

### Hardware implementation

A hardware implementation typically reads from a real-time communication layer
(EtherCAT, CAN, ROS 2 topics, etc.). Below is a minimal skeleton showing the
structure; fill in the platform-specific code in each method body:

```cpp
#include "state_interface.hpp"

class MyHardwareStateInterface : public exploy::control::RobotStateInterface {
 public:
  // ── Initialisation (called once, non-real-time) ─────────────────────────

  bool initImuAngularVelocityImu(const std::string& /*imu_name*/) override {
    // Subscribe to IMU topic or open device handle.
    return true;
  }

  bool initJointPosition(const std::string& joint_name) override {
    // Map joint_name to its encoder channel.
    encoder_channels_[joint_name] = lookupChannel(joint_name);
    return true;
  }

  bool initJointOutput(const std::string& joint_name) override {
    // Map joint_name to its actuator channel.
    actuator_channels_[joint_name] = lookupActuator(joint_name);
    return true;
  }

  // ── Getters (called every cycle, real-time) ──────────────────────────────

  std::optional<exploy::control::AngularVelocity> imuAngularVelocityImu(const std::string& imu_name) const override {
    // Read from IMU driver.
    return imu_driver_.angularVelocity(imu_name);
  }

  std::optional<double> jointPosition(const std::string& joint_name) const override {
    auto it = encoder_channels_.find(joint_name);
    if (it == encoder_channels_.end()) return std::nullopt;
    return encoder_driver_.read(it->second);
  }

  // ── Setters (called every cycle, real-time) ──────────────────────────────

  bool setJointPosition(const std::string& joint_name, double position) override {
    auto it = actuator_channels_.find(joint_name);
    if (it == actuator_channels_.end()) return false;
    actuator_driver_.write(it->second, position);
    return true;
  }

 private:
  ImuDriver imu_driver_;
  EncoderDriver encoder_driver_;
  std::unordered_map<std::string, int> encoder_channels_;
  std::unordered_map<std::string, int> actuator_channels_;
};
```

The base class defaults log an error and return `false` / `std::nullopt` for
every unimplemented method, which will cause `controller.init()` to fail with
a descriptive message if a required sensor is missing.

### Simulation implementation

A simulator implementation reads from the physics engine instead of hardware
drivers. The pattern is identical — `init*()` registers the articulation handle
and getters/setters query it:

```cpp
class MySimStateInterface : public exploy::control::RobotStateInterface {
 public:
  explicit MySimStateInterface(SimArticulation& articulation)
      : articulation_(articulation) {}

  bool initJointPosition(const std::string& joint_name) override {
    return articulation_.hasJoint(joint_name);
  }

  std::optional<double> jointPosition(const std::string& joint_name) const override {
    return articulation_.getJointPosition(joint_name);
  }

  bool setJointPosition(const std::string& joint_name, double position) override {
    articulation_.setJointPositionTarget(joint_name, position);
    return true;
  }

  // … implement remaining sensors your model requires …

 private:
  SimArticulation& articulation_;
};
```

---

## Step 2 — Implement `CommandInterface`

`CommandInterface` supplies high-level commands to the policy — typically a desired
SE2 velocity from a joystick, a navigation planner, or a supervisory controller.

Like `RobotStateInterface`, it uses the same `init*()` / getter pattern:

```cpp
#include "command_interface.hpp"

class MyJoystickCommandInterface : public exploy::control::CommandInterface {
 public:
  bool initSe2Velocity(const std::string& /*command_name*/,
                       const exploy::control::SE2VelocityConfig& /*cfg*/) override {
    // Subscribe to joystick topic or open device.
    return joystick_.open("/dev/input/js0");
  }

  std::optional<exploy::control::SE2Velocity> se2Velocity(
      const std::string& /*command_name*/) override {
    // Read latest joystick state; apply any scaling specified in cfg.ranges.
    auto [vx, vy, omega] = joystick_.readAxes();
    return exploy::control::SE2Velocity{vx, vy, omega};
  }

 private:
  Joystick joystick_;
};
```

---

## Step 3 — Implement `DataCollectionInterface`

`DataCollectionInterface` records telemetry produced by the controller. It has two
responsibilities:

- `registerDataSource()` is called once during `controller.init()` to announce a
  named data channel and bind it to a live memory span or scalar.
- `collectData(time_us)` is called every cycle during `controller.update()` to
  snapshot the currently bound values at the given timestamp.

```cpp
// data_collection_interface.hpp (simplified)
class DataCollectionInterface {
 public:
  virtual bool registerDataSource(const std::string& prefix,
                                  std::span<const float> data) { return false; }
  virtual bool registerDataSource(const std::string& prefix,
                                  std::span<const double> data) { return false; }
  virtual bool registerDataSource(const std::string& prefix,
                                  const double& scalar) { return false; }
  virtual bool collectData(uint64_t time_us) = 0;
};
```

Common implementations write telemetry to a file (MCAP, HDF5), publish it to a
middleware topic (ROS 2, LCM), or forward it to a time-series database.

### No-op placeholder

If you don't need data collection yet, use a no-op implementation:

```cpp
class NoOpDataCollection : public exploy::control::DataCollectionInterface {
 public:
  bool registerDataSource(const std::string&, std::span<const double>) override { return true; }
  bool registerDataSource(const std::string&, std::span<const float>)  override { return true; }
  bool registerDataSource(const std::string&, const double&)           override { return true; }
  bool collectData(uint64_t) override { return true; }
};
```

---

## Step 4 — Create and Configure the Controller

With the three interfaces ready, instantiate `OnnxRLController` and load the model:

```cpp
#include "controller.hpp"
#include "logging_interface.hpp"

// 1. Set up a logger (optional — defaults to StdoutLogger if omitted).
exploy::control::StdoutLogger logger;
exploy::control::setLogger(&logger);

// 2. Instantiate the three interfaces.
MyHardwareStateInterface state;
MyJoystickCommandInterface command;
McapDataCollection data_collection("run_001.mcap");

// 3. Create the controller.
exploy::control::OnnxRLController controller(state, command, data_collection);

// 4. Load the ONNX model. This parses metadata and wires up all matchers.
if (!controller.create("/path/to/policy.onnx")) {
    std::cerr << "Failed to load ONNX model\n";
    return 1;
}

// 5. Initialise all components (calls init*() on state/command/data interfaces).
//    Pass true to enable data collection during the run.
if (!controller.init(/*enable_data_collection=*/true)) {
    std::cerr << "Controller initialisation failed\n";
    return 1;
}
```

`controller.create()` reads the model metadata to determine the update rate and
maps every ONNX tensor to a concrete `Input` or `Output` using the built-in
matchers. `controller.init()` then calls the corresponding `init*()` methods on
your state and command interfaces, one per matched tensor.

---

## Step 5 — Run the Control Loop

After initialisation, call `controller.update(time_us)` once per control cycle.
The controller reads the current timestamp, runs inference, and dispatches joint
targets:

```cpp
const double update_rate_hz = static_cast<double>(controller.context().updateRate());
const std::chrono::duration<double> dt(
    update_rate_hz > 0.0 ? 1.0 / update_rate_hz : 0.0);
const uint64_t dt_us = static_cast<uint64_t>(dt.count() * 1e6);

uint64_t time_us = 0;
int failures = 0;

for (int cycle = 0; cycle < num_cycles; ++cycle) {
    auto cycle_start = std::chrono::steady_clock::now();

    // Single control step: read state → infer → write targets → log.
    if (!controller.update(time_us)) {
        std::cerr << "Cycle " << cycle << " failed\n";
        ++failures;
    }

    time_us += dt_us;

    // Sleep to maintain the target control rate.
    if (update_rate_hz > 0.0) {
        auto elapsed = std::chrono::steady_clock::now() - cycle_start;
        if (elapsed < dt)
            std::this_thread::sleep_for(dt - elapsed);
    }
}
```

`controller.context().updateRate()` returns the update rate (in Hz) that was
baked into the ONNX metadata at export time, so the loop frequency automatically
matches the training configuration.

---

## Advanced: Custom Matchers

The built-in matchers cover the standard IsaacLab tensor naming conventions.
If your model uses a different naming scheme you can register additional matchers
before calling `controller.create()`:

```cpp
#include "matcher.hpp"

class CustomBodyPositionMatcher : public exploy::control::Matcher {
 public:
  bool matches(const exploy::control::Match& maybe_match) override {
    // Match tensors of the form obj.<articulation>.<body>.pos_b_rt_w_in_w.
    std::smatch m;
    static const std::regex pattern(
        R"(obj\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\.pos_b_rt_w_in_w)");
    if (std::regex_match(maybe_match.name, m, pattern) && m.size() > 2) {
      found_matches_[m[2].str()] = maybe_match;
      return true;
    }
    return false;
  }

  std::vector<std::unique_ptr<exploy::control::Input>> createInputs() const override {
    std::vector<std::unique_ptr<exploy::control::Input>> inputs;
    for (const auto& [body_name, found_match] : found_matches_)
      inputs.push_back(
          std::make_unique<exploy::control::BodyPositionInput>(found_match.name, body_name));
    return inputs;
  }
};

// Register before create():
controller.context().registerMatcher(std::make_unique<CustomBodyPositionMatcher>());
```

Custom matchers are evaluated after the built-in ones, so you only need to handle
tensors that the defaults don't cover.

---

## Advanced: Custom Logging Backend

By default the controller logs to stdout. To redirect log messages to your own
logging system, implement `LoggingInterface` and call `setLogger()`:

```cpp
#include "logging_interface.hpp"

// Example: forward to spdlog.
class SpdlogAdapter : public exploy::control::LoggingInterface {
 public:
  explicit SpdlogAdapter(std::shared_ptr<spdlog::logger> logger)
      : logger_(std::move(logger)) {}

  void log(Level level, std::string_view message) override {
    switch (level) {
      case Level::Error: logger_->error("{}", message); break;
      case Level::Warn:  logger_->warn("{}",  message); break;
      case Level::Info:  logger_->info("{}",  message); break;
    }
  }

 private:
  std::shared_ptr<spdlog::logger> logger_;
};

// Example: forward to ROS 2.
class Ros2LoggingAdapter : public exploy::control::LoggingInterface {
 public:
  void log(Level level, std::string_view message) override {
    switch (level) {
      case Level::Error: RCLCPP_ERROR(rclcpp::get_logger("exploy"), "%s", message.data()); break;
      case Level::Warn:  RCLCPP_WARN (rclcpp::get_logger("exploy"), "%s", message.data()); break;
      case Level::Info:  RCLCPP_INFO (rclcpp::get_logger("exploy"), "%s", message.data()); break;
    }
  }
};

// Activate before creating the controller:
SpdlogAdapter adapter(spdlog::default_logger());
exploy::control::setLogger(&adapter);
```

`setLogger()` stores a raw pointer, so the adapter must outlive the controller.
Pass `nullptr` to revert to the built-in stdout logger.

---

## Advanced: Custom Commands

The built-in `CommandInterface` covers standard command types (SE2 velocity,
SE3 pose, boolean selectors, float values, joint positions). If your policy
consumes a tensor that doesn't match any of those — for example an arbitrary
vector produced by a higher-level planner — you can feed it in by registering
a custom `Matcher` and `Input` pair.

### How it works

When `controller.create()` loads the ONNX model it iterates over every input
and output tensor and calls `matches()` on each registered matcher. If your
matcher claims a tensor, `createInputs()` is later called to instantiate the
component that will populate it every cycle.

The pattern therefore involves three pieces:

| Piece | Responsibility |
|-------|---------------|
| Your data-source class | Owns the data and exposes `init` / `read` methods; independent of `CommandInterface` |
| `CustomInput : public Input` | Bridges your data source to an ONNX input buffer |
| `CustomMatcher : public Matcher` | Recognises the tensor by name, stores the match, and constructs `CustomInput` instances |

### Example — feeding an arbitrary vector command

Suppose your model has an input tensor named `custom.planner.output` of shape
`[1, N]` that should be filled from a motion planner. The naming convention
`custom.<type>.<name>` groups tensors by type and identifies each by name.
The built-in matchers ignore tensors with the `custom.*` prefix, so you need a
custom matcher.

#### Step 1 — define an interface for your data source

```cpp
// Your own header — no dependency on exploy.
class PlannerInterface {
 public:
  // Called once during init; return false to abort controller init.
  virtual bool initPlannerOutput(const std::string& output_name) = 0;

  // Called every cycle; return std::nullopt when data is not yet available.
  virtual std::optional<std::vector<double>>
      plannerOutput(const std::string& output_name) const = 0;
};
```

#### Step 2 — implement `Input`

```cpp
#include "exploy/components.hpp"  // Input, OnnxRuntime
#include "exploy/interfaces.hpp"

class PlannerInput : public exploy::control::Input {
 public:
  PlannerInput(const std::string& tensor_key,
               const std::string& output_name,
               PlannerInterface& planner)
      : tensor_key_(tensor_key),
        output_name_(output_name),
        planner_(planner) {}

  // Called once by controller.init() — non-real-time.
  bool init(exploy::control::RobotStateInterface& /*state*/,
            exploy::control::CommandInterface& /*command*/) override {
    return planner_.initPlannerOutput(output_name_);
  }

  // Called every cycle by controller.update() — real-time.
  bool read(exploy::control::OnnxRuntime& runtime,
            exploy::control::RobotStateInterface& /*state*/,
            exploy::control::CommandInterface& /*command*/) override {
    auto maybe_data = planner_.plannerOutput(output_name_);
    if (!maybe_data) return false;

    auto maybe_buffer = runtime.inputBuffer<float>(tensor_key_);
    if (!maybe_buffer) return false;

    if (maybe_buffer->size() != maybe_data->size()) return false;
    std::copy(maybe_data->begin(), maybe_data->end(), maybe_buffer->begin());
    return true;
  }

 private:
  std::string tensor_key_;
  std::string output_name_;
  PlannerInterface& planner_;
};
```

#### Step 3 — implement `Matcher`

The matcher uses the `custom.<type>.<name>` pattern. The regex captures the
type segment (group 1) and the name segment (group 2). Here the matcher
only handles the `planner` type; extend `matches()` to cover additional types
as needed.

```cpp
#include "exploy/matcher.hpp"
#include <regex>

class PlannerMatcher : public exploy::control::Matcher {
 public:
  explicit PlannerMatcher(PlannerInterface& planner) : planner_(planner) {}

  // Called by controller.create() for every tensor in the ONNX model.
  bool matches(const exploy::control::Match& maybe_match) override {
    // Matches tensors of the form custom.<type>.<name>.
    static const std::regex kPattern(R"(custom\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+))");
    std::smatch m;
    if (std::regex_match(maybe_match.name, m, kPattern) && m.size() > 2) {
      const std::string& type = m[1].str();  // e.g. "planner"
      const std::string& name = m[2].str();  // e.g. "output"
      if (type == "planner") {
        found_matches_[name] = maybe_match;
        return true;
      }
    }
    return false;
  }

  // Called after create() to instantiate inputs for every claimed tensor.
  std::vector<std::unique_ptr<exploy::control::Input>> createInputs() const override {
    std::vector<std::unique_ptr<exploy::control::Input>> inputs;
    for (const auto& [name, match] : found_matches_) {
      inputs.push_back(
          std::make_unique<PlannerInput>(match.name, name, planner_));
    }
    return inputs;
  }

 private:
  PlannerInterface& planner_;
  // found_matches_ is inherited from exploy::control::Matcher
};
```

#### Step 4 — register the matcher before `controller.create()`

```cpp
MyMotionPlanner planner;           // implements PlannerInterface

exploy::control::OnnxRLController controller(state, command, data_collection);

// Must be registered before create() — matchers are evaluated inside create().
controller.context().registerMatcher(std::make_unique<PlannerMatcher>(planner));

controller.create("/path/to/policy.onnx");
controller.init(/*enable_data_collection=*/false);
```

After this, every tensor matching `custom.planner.<name>` is claimed by your
matcher. For each such tensor `read()` is called once per `controller.update()`
cycle, immediately before ONNX inference.

### Key constraints

- **Register before `create()`** — matchers are only consulted during model
  loading. Registering after `create()` has no effect.
- **Buffer element type** — use `runtime.inputBuffer<float>(key)` for `float`
  tensors. The ONNX runtime buffer type must match the tensor's element type in
  the model.
- **Return value** — returning `false` from `read()` causes `controller.update()`
  to return `false` for that cycle, signalling a failed update to the caller.
- **Custom matchers are evaluated after built-ins** — if a tensor name matches a
  built-in matcher it is claimed first and your matcher will not see it.
