# Exploy

EXport and dePLOY Reinforcement Learning policies.

## Features

- **Environment Exporting**: Export RL environments and policies from
  simulation frameworks
- **ONNX Runtime Integration**: Deploy trained policies using ONNX Runtime
  for efficient inference
- **Multi-Framework Support**: Built-in support for IsaacLab with extensible
  framework integration
- **C++ Controller**: High-performance C++ controller for real-time policy
  execution

## Project Structure

- `control/`: C++ controller library with ONNX Runtime integration
- `exporter/`: Python exporter package for policy and environment export
- `examples/`: Usage examples for IsaacLab and ROS2 Control
- `docs/`: Documentation source files
- `cmake/`: CMake modules for building the project

## Getting Started

### Prerequisites

- [Pixi](https://pixi.sh) installed on your system
- Linux x86_64 (tested on Ubuntu 22.04 with glibc 2.35)
- Git

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

3. **Setup dependencies** (includes IsaacLab if using that environment):

   ```bash
   pixi run setup
   ```

### Building the Project

The project uses CMake and Ninja, managed by Pixi.

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

```python
from exporter.exporter import Exporter
from exporter_frameworks.isaaclab import IsaacLabExportableEnvironment

# Create an exportable environment
env = IsaacLabExportableEnvironment(
    task_name="Isaac-Velocity-Flat-Anymal-D-v0",
    num_envs=1
)

# Export the policy
exporter = Exporter(env, policy_module)
exporter.export("my_policy.onnx")
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

#### Running IsaacLab Examples

```bash
pixi run -e isaaclab python examples/exporter_scripts/export_isaaclab.py
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
- **C++ Libraries**: ONNX Runtime (>=1.15), Eigen (>=3.4), fmt (>=9.1),
  nlohmann_json (>=3.11)
- **Python Libraries**: PyTorch (via IsaacLab), onnxscript,
  pybind11 (>=2.10)
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
- `pixi run -e core lint`: Check Python code with ruff
- `pixi run -e core format`: Format Python code with ruff

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

We welcome bug fixes and improvements! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Reporting issues
- Submitting pull requests
- Code style requirements
- Testing requirements

All contributions require review and approval from project owners before merging.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.

Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute
