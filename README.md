# Exploy

EXport and dePLOY Reinforcement Learning policies.

The core idea lies in a "self-contained" export approach:
Rather than exporting only the neural network policy, this tool captures the entire environment
logic—including observation generation and action processing—into a single ONNX file.
By tracing Torch operations from the simulation environment, the exporter embeds the computational
layers required to transform raw robot state interfaces into policy inputs and policy outputs into
executable commands.

By encapsulating the environment's computation graph within the model file itself,
this library minimizes operational effort and maximizes confidence that a policy will behave
identically in simulation and on physical hardware.

## Features

- **Environment Exporting**: Export RL environments and policies from
  simulation frameworks in a self-contained ONNX file.
- **C++ Controller with ONNX Runtime Integration**: Deploy trained policies using ONNX Runtime
  for real-time policy execution
- **Multi-Framework Support**: Built-in support for IsaacLab with extensible
  framework integration

## Project Structure

- `control/`: C++ controller library with ONNX Runtime integration
- `exporter/`: Python exporter package for policy and environment export
- `examples/`: Usage examples for supported frameworks
- `docs/`: Documentation source files

## Getting Started

### Prerequisites

- [Pixi](https://pixi.sh) installed on your system

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/bdaiinstitute/exploy.git
   cd exploy
   ```

2. **Initialize the environment and install dependencies**:

   ```bash
   pixi install
   ```

### Building the Project

1. **Configure and Build C++ Library**:

   ```bash
   pixi run -e controller configure
   pixi run -e controller build
   ```

2. **Run tests**:

   ```bash
   # Python tests
   pixi run -e core test

   # C++ tests
   pixi run -e controller test
   ```

### Usage Examples

#### Exporting a Policy from IsaacLab

Run the example export

```bash
pixi run export-isaaclab
```

```python
import exploy.exporter.core as exporter
from exploy.exporter.frameworks.isaaclab.env import IsaacLabExportableEnvironment

# Create an exportable environment from a ManagerBasedRLEnv
exportable_env = IsaacLabExportableEnvironment(env)

# Export the policy
exporter.export_environment_as_onnx(
    env=exportable_env,
    actor=actor,
    normalizer=normalizer,
    path=onnx_export_dir,
    filename=onnx_export_file,
    verbose=False,
)

# Create a session wrapper to run inference
session_wrapper = exporter.SessionWrapper(
    onnx_folder=onnx_export_dir,
    onnx_file_name=onnx_export_file,
    policy=policy,
    optimize=True,
)

# Evaluate.
with torch.inference_mode():
    export_ok, _ = exporter.evaluate(
        env=exportable_env,
        context_manager=exportable_env.context_manager(),
        session_wrapper=session_wrapper,
        num_steps=200,
        verbose=True,
        pause_on_failure=True,
    )

assert export_ok
```

#### Using the C++ Controller

```cpp
#include "controller.hpp"
#include "state_interface.hpp"
#include "command_interface.hpp"
#include "data_collection_interface.hpp"

// Implement the required interfaces for your robot
class MyRobotState : public exploy::control::RobotStateInterface {
  // Implement joint state, base state, sensor methods...
};

class MyRobotCommand : public exploy::control::CommandInterface {
  // Implement SE2 velocity, SE3 pose, and other command methods...
};

class MyDataCollection : public exploy::control::DataCollectionInterface {
  // Implement data logging methods...
};

int main() {
  // Create interface instances
  MyRobotState state;
  MyRobotCommand command;
  MyDataCollection data_collection;

  // Create the controller
  exploy::control::OnnxRLController controller(state, command, data_collection);

  // Load the ONNX model
  if (!controller.create("/path/to/policy.onnx")) {
    return -1;
  }

  // Initialize the controller
  if (!controller.init(false)) {
    return -1;
  }

  // Get the update rate from the model metadata (Hz)
  int update_rate_hz = controller.context().updateRate();
  uint64_t period_us = 1000000 / update_rate_hz;

  // Control loop
  uint64_t next_update_us = getCurrentTimeMicroseconds();
  while (running) {
    uint64_t now_us = getCurrentTimeMicroseconds();

    // Wait until next update time
    if (now_us < next_update_us) {
      sleepUntil(next_update_us);
      now_us = next_update_us;
    }

    // Run inference and update commands
    if (!controller.update(now_us)) {
      // Handle error
      break;
    }

    // Schedule next update
    next_update_us += period_us;
  }

  return 0;
}
```

## Versioning

This project uses semantic versioning (MAJOR.MINOR.PATCH). The current version is specified in `pixi.toml`.

Releases are published using GitHub Releases. Version tags follow the format `vX.Y.Z`.

## Limitations

- IsaacLab integration requires NVIDIA GPU with CUDA support
- C++ controller is designed for real-time systems but performance depends
  on hardware
- ONNX model execution time varies based on policy complexity

## Dependencies

All dependencies are managed through Pixi and specified in `pixi.toml` with version constraints.

### Core Dependencies

- **Python**: 3.11.x
- **C++ Build Tools**: CMake (3.24), Ninja, GCC/G++
- **C++ Libraries**: ONNX Runtime (>=1.15), Eigen (>=3.4), fmt (>=9.1), nlohmann_json (>=3.11)
- **Python Libraries**: PyTorch, onnxscript
- **Testing**: GoogleTest, pytest

See `pixi.toml` for complete dependency specifications with version ranges.

## Development

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. To set up pre-commit hooks:

```bash
# Install pre-commit hooks
pixi run -e python pre-commit install

# Run pre-commit on all files (optional)
pixi run -e python pre-commit run --all-files
```

### Available Tasks

Specified in `pixi.toml`:

**Python tasks** (use `-e core` environment):

- `pixi run -e core test`: Run Python tests with pytest
- `pixi run -e core lint-python`: Check Python code with ruff
- `pixi run -e core format-python`: Format Python code with ruff

**C++ tasks** (use `-e controller` environment):

- `pixi run -e controller configure`: Run CMake configuration
- `pixi run -e controller build`: Build the C++ library
- `pixi run -e controller test`: Run C++ tests with CTest
- `pixi run -e controller format-cpp`: Format C++ code with clang-format

## Maintenance and Support

This project is under **light maintenance**. No feature development is
guaranteed, but if you have bug reports and/or pull requests that fix bugs,
expect an RAI maintainer to respond within a few weeks.

## Contributing

We welcome your contributions!

To contribute:

- Report bugs and suggest improvements by opening an issue.
- Submit pull requests for code changes or documentation updates.
- Follow the project's code style and testing guidelines (see comments and existing code for reference).
- Ensure your changes pass all tests and pre-commit checks.

All contributions are reviewed by project maintainers before merging. Thank you for helping improve this project!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.

Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute
