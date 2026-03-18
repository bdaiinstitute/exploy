# Exploy

EXport and dePLOY Reinforcement Learning policies.

The core idea lies in a "self-contained" export approach:
Rather than exporting only the neural network policy, this tool captures the entire environment
logic, including observation generation and action processing, into a single ONNX file.
By tracing Torch operations from the simulation environment, the exporter embeds the computational
layers required to transform raw robot state interfaces into policy inputs and policy outputs into
executable commands.

By encapsulating the environment's computation graph within the model file itself,
this library minimizes operational effort and maximizes confidence that a policy will behave
identically in simulation and on physical hardware.

Authors: Dario Bellicoso, Annika Wollschläger

## Features

- **Environment Exporting**: Export RL environments and policies from
  simulation frameworks in a self-contained ONNX file.
- **C++ Controller with ONNX Runtime Integration**: Deploy trained policies using ONNX Runtime
  for real-time policy execution
- **Multi-Framework Support**: Built-in support for IsaacLab with extensible
  framework integration

## Project Structure

- `control/`: C++ controller library with ONNX Runtime integration
- `python/exploy/`: Python exporter package for policy and environment export
- `examples/exporter_scripts/`: Usage examples for supported frameworks
- `examples/controller/`: Usage examples for control development
- `docs/`: Documentation source files

## Documentation

Exploy's documentation is available at [bdaiinstitute.github.io/exploy](https://bdaiinstitute.github.io/exploy).
To get started with Exploy's core concepts, refer to the following guides:

- [**Exporter**](docs/tutorial/exporter/exporter_tutorial.md) — Step-by-step guide to
  exporting an RL environment and policy to a self-contained ONNX file using `exploy.exporter.core`.
- [**Controller**](docs/tutorial/controller/controller_tutorial.md) — Step-by-step guide to
  deploying a trained policy on a robot using the C++ controller with ONNX Runtime integration.

## Getting Started

### Prerequisites

We use [Pixi](https://pixi.sh) to build this repository. See the
[Pixi installation guide](https://pixi.prefix.dev/latest/installation/) for setup instructions.

### Installation

#### Clone the repository

   ```bash
   git clone https://github.com/bdaiinstitute/exploy.git
   cd exploy
   ```

#### Install the Python exporter as a pip package

  Exploy is split into two packages: the Exporter package developed in Python and the Controller package developed in C++.
  To use the exporter in your project, install it with `pip`:

   ```bash
   pip install -e .
   ```

   The command above installs the core implementation of the exporter. This repository also provides
   support for exporter integrations with environments developed in `IsaacLab`. You can install
   the additional dependency with:

   ```bash
   pip install -e .[isaaclab]
   ```

#### Integrate into your CMake project

   The easiest way to consume the library is via `add_subdirectory`:

   ```cmake
   cmake_minimum_required(VERSION 3.20)
   project(my_robot_controller LANGUAGES CXX)

   set(CMAKE_CXX_STANDARD 20)
   set(CMAKE_CXX_STANDARD_REQUIRED ON)

   # Path to the exploy project root
   set(EXPLOY_ROOT_DIR "/path/to/exploy" CACHE PATH "Exploy root directory")

   # Pull in the exploy shared library (disables its tests in your build)
   set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
   add_subdirectory(
       "${EXPLOY_ROOT_DIR}/control"
       "${CMAKE_CURRENT_BINARY_DIR}/exploy_control"
       EXCLUDE_FROM_ALL
   )

   add_executable(my_controller main.cpp)
   target_link_libraries(my_controller PRIVATE exploy)
   ```

  See [`examples/controller/`](https://github.com/bdaiinstitute/exploy/blob/main/examples/controller) for a complete working example.

#### Initialize the environment and install dependencies

   ```bash
   pixi install
   ```

### Building the Project

#### Configure and Build C++ Library

   ```bash
   pixi run -e controller configure
   pixi run -e controller build
   ```

#### Run tests

   ```bash
   # Python tests
   pixi run -e core test

   # C++ tests
   pixi run -e controller test
   ```

### Use Exploy

Start with the [documentation](#documentation) to learn the core workflow for exporting and
deploying a task. If you have an NVIDIA GPU, you can also run an IsaacLab exporter example with
`pixi run -e isaaclab export-isaaclab`.

## Versioning

This project uses semantic versioning (MAJOR.MINOR.PATCH). The current version is specified in `pixi.toml`.

Releases are published using GitHub Releases. Version tags follow the format `vX.Y.Z`.

## Dependencies

All dependencies are managed through Pixi and specified in `pixi.toml` with version constraints.

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

Tasks are run through Pixi and defined in `pixi.toml`. Common tasks include:

#### Core exporter tasks

- `pixi run -e core test`: Run Python tests with pytest

#### Python code quality tasks

- `pixi run -e core lint-python`: Check Python code with ruff
- `pixi run -e core format-python`: Format Python code with ruff

#### C++ tasks

- `pixi run -e controller configure`: Run CMake configuration
- `pixi run -e controller build`: Build the C++ library
- `pixi run -e controller test`: Run C++ tests with CTest
- `pixi run -e controller format-cpp`: Format C++ code with clang-format

#### Example task

- `pixi run -e isaaclab export-isaaclab`: Export a policy for an existing IsaacLab task

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

## Citation

If you use this work in your research or project, please consider citing it using the 'Cite this repository'
button in the sidebar, or using:

```bibtex
@misc{exploy2026,
  author = {Dario Bellicoso, Annika Wollschläger},
  title = {Exploy: EXport and dePLOY Reinforcement Learning policies},
  month = {March},
  year = {2026},
  url = {https://github.com/bdaiinstitute/exploy}
}
```
