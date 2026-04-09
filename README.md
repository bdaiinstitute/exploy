# Exploy

> [!IMPORTANT]
> This repository has moved from [`bdaiinstitute/exploy`](https://github.com/bdaiinstitute/exploy) to [`rai-opensource/exploy`](https://github.com/rai-opensource/exploy).

**EX**port and de**PLOY** reinforcement learning policies.

Exploy is a library that packages complex reinforcement learning environment logic and policies into a single,
deployable computational graph.

## Motivation

A common approach in robotics reinforcement learning is training policies in simulation and deploying
them on physical hardware.
These two domains are typically quite different, since simulation environments are defined in Python
while real robots operate using C++ / ROS.
To bridge this gap, a standard method for exchanging policies is exporting them as ONNX files.
This provides a language-agnostic format that can be loaded and executed across different software stacks.

However, this is only part of the story.
Most policy exchange solutions only include the structure and weights of the neural network within the ONNX file.
They map raw observations to raw actions but ignore the complex surrounding logic required for observation
generation and action processing.

Exploy solves this by embedding the exact computational layers required to execute the entire environment logic.
It captures the complete pipeline that maps high-level robot states into policy observations and post-processes
the network's output into real actuation signals.

This is achieved by tracing PyTorch operations directly from the simulation environment.
By encapsulating the environment's entire computation graph within the model file, Exploy minimizes operational effort
and maximizes confidence that your policy will behave identically in simulation and on the physical robot.

**Authors**: Dario Bellicoso, Annika Wollschläger

## Features

- **Environment Exporting**:
  Export RL environments and policies from simulation frameworks in a self-contained ONNX file.
- **C++ Controller with ONNX Runtime Integration**:
  Deploy trained policies using ONNX Runtime for real-time policy execution.
- **Multi-Framework Support**:
  Built-in support for [IsaacLab][isaaclab], with an architecture designed for extensible framework integration.

[isaaclab]: https://github.com/isaac-sim/IsaacLab

## Project Structure

```text
exploy/
├── control/               # C++ controller library with ONNX Runtime integration
├── docs/                  # Documentation source files
├── examples/
│   ├── controller/        # Usage examples for control development
│   └── exporter_scripts/  # Usage examples for supported frameworks
├── python/exploy/         # Python exporter package for policy and environment
└── ros/                   # ROS integration packages
```

## Documentation

Exploy's documentation is available at [bdaiinstitute.github.io/exploy][docs].
To get started with the core concepts, refer to our step-by-step guides:

- [**Exporter**][tutorial_exporter]:
  Learn to export an RL environment and policy to a self-contained ONNX file using  `exploy.exporter.core`.
- [**Controller**][tutorial_controller]:
  Learn to deploy a trained policy on a robot using the C++ controller with ONNX Runtime integration.

[docs]: https://bdaiinstitute.github.io/exploy
[tutorial_exporter]: https://bdaiinstitute.github.io/exploy/tutorial/exporter/exporter_tutorial.html
[tutorial_controller]: https://bdaiinstitute.github.io/exploy/tutorial/controller/controller_tutorial.html

## Installation and Usage

### Python Exporter

To consume the Python package in your own project,
the recommended approach is to install it directly from the git repository using `pip`:

```bash
pip install git+https://github.com/bdaiinstitute/exploy.git
```

To install with additional dependencies for `IsaacLab` integration:

```bash
pip install "exploy[isaaclab]@git+https://github.com/bdaiinstitute/exploy.git"
```

### C++ Controller Library

To consume the C++ controller library, use plain CMake commands to build and install it on your system (or to a custom prefix):

```bash
git clone https://github.com/bdaiinstitute/exploy.git
cd exploy
cmake -S control/ -B build/ -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build build/ --parallel
cmake --install build/
```

Once installed, you can integrate it into your downstream CMake project:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_robot_controller LANGUAGES CXX)

# Find the installed exploy package.
find_package(exploy CONFIG REQUIRED)

# Link against the exploy library.
add_executable(my_controller main.cpp)
target_link_libraries(my_controller PRIVATE exploy::exploy)
```

See the [`examples/controller/`](./examples/controller) directory for a complete working example.

### ROS Integration

If you are deploying your policy within a ROS ecosystem, Exploy provides resources to easily load it into your workspace.
You can use the `exploy_vendor` wrapper package located in the [`ros/`](./ros) folder.

Simply copy or symlink the package into your colcon workspace's `src` directory, and then build it with:

```bash
colcon build --packages-up-to exploy_vendor
```

This ensures the Exploy C++ controller library is built and exposed correctly as a dependency for your other ROS packages.

### Running Examples

> [!NOTE]
> This example assumes you have an NVIDIA GPU available.

You can export an ONNX policy for an existing [IsaacLab][isaaclab] tasks:

```bash
pixi run export-isaaclab
```

Then, you can run the [C++ example][cpp_example] that loads and executes the exported policy:

```bash
pixi run run-cpp-example
```

## Development

### Using Official Pixi Environment

This project uses [Pixi][pixi] to deliver a ready-to-use, reproducible, and fully isolated development environment.
It supports both C++ and Python development, with all required dependencies preinstalled and configured.
The repository also includes a VS Code setup that integrates seamlessly with the Pixi environment,
enabling features such as IntelliSense and debugging for both languages out of the box.

See the [Pixi installation guide][pixi_setup] for initial setup instructions.

[pixi]: https://pixi.sh
[pixi_setup]: https://pixi.prefix.dev/latest/installation/

#### 1. Clone the repository

```bash
git clone https://github.com/bdaiinstitute/exploy.git
cd exploy
```

#### 2. Install the environment and build the project

```bash
pixi install
pixi run build
```

#### 3. Open the project in VSCode

After running this command, you can open the project folder in VS Code.
The editor will automatically detect the Pixi environment.

- C++ autocompletion should work out of the box.
- By default, the Python environment is configured for the `exploy.exporter.core` subpackage.

If you are working on a machine with an NVIDIA GPU, you can switch to the environment
for the `exploy.exporter.frameworks.isaaclab` component:

- Click the Python icon in the left sidebar.
- In the `Environment Managers` panel, locate `Pixi`.
- Select the `exploy/isaaclab` entry and set it as the project environment.

#### 4. Available Tasks

We define a variety of development tasks in `pixi.toml`.

You can inspect all available tasks at any time by running:

```text
$ pixi task list
Tasks that can run on this machine:
-----------------------------------
# <...>

Task    Description

build   Build all C++ code.
check   Check all code without making changes.
clean   Clean all build artifacts and caches.
format  Format all code.
lint    Lint all code.
test    Run all tests.
```

To simplify development, the main tasks wrap both Python and C++ operations.
Most of these main tasks also come with language-specific variants by appending `-cpp` or `-python`.

[cpp_example]: ./examples/controller

#### 5. Linting and Formatting

We use pre-commit hooks to ensure code quality.

Either run the same check performed in our CI pipeline locally before pushing your changes:

```bash
pixi run pre-commit-ci
```

or install the pre-commit hooks to run automatically on every commit:

```bash
pixi run -e ci pre-commit install
```

### Alternative method

Specifically for Python development, you might prefer to install `exploy` in editable mode directly within your setup.

Run the following command with a `pip` that points to your desired Python environment:

```bash
# Install only the core component
pip install -e .

# Install the isaaclab component
pip install -e ".[isaaclab]"
```

Using this method, you are responsible for ensuring the consistency of the dependencies in your environment,
which is not guaranteed to be the same as the one used in our CI.

## Maintenance and Contributing

This project is under **light maintenance**.

No feature development is guaranteed, but RAI maintainers aim to respond to bug reports and pull requests within a few weeks.

We welcome your contributions! To contribute:

- Report bugs and suggest improvements by opening an issue.
- Submit pull requests for code changes or documentation updates.
- Follow the project's code style and testing guidelines.
- Ensure your changes pass all tests and pre-commit checks before requesting a review.

## Versioning

- This project uses [semantic versioning 2.0][semver2].
- Releases are published using GitHub Releases with tags formatted as `vM.m.p`.
- The current version is specified in `pixi.toml` and need manual propagation to the C++, Python, and ROS components.
- All Python dependencies are declared in [`pyproject.toml`][pyproject].
- All C++ dependencies are declared in the main [`CMakeLists.txt`][cmakelists].
- The `pixi.toml` file and its lock file can be used as source of truth for a fully reproducible development environment
  with exact versions tested in CI.

[semver2]: https://semver.org/
[pyproject]: ./pyproject.toml
[cmakelists]: ./control/CMakeLists.txt

## License

This project is licensed under the MIT License. See the [LICENSE][license] file for details.

[license]: LICENSE

```text
Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute.
```

## Citation

If you use this work in your research or project, please consider citing it using the 'Cite this repository'
button in the sidebar, or using the following BibTeX entry:

```bibtex
@misc{exploy2026,
  author = {Dario Bellicoso, Annika Wollschläger},
  title = {Exploy: EXport and dePLOY Reinforcement Learning policies},
  month = {March},
  year = {2026},
  url = {https://github.com/bdaiinstitute/exploy}
}
```
