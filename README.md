# Exporter

A development environment for a C++ library with Python bindings, integrated with ONNX Runtime and PyTorch, using [Pixi](https://pixi.sh).

## Project Structure

- `include/`, `src/`: C++ library and Pybind11 bindings.
- `python/`: Python package source.
- `examples/`:
    - `isaaclab/`: Python examples for IsaacLab integration.
    - `ros2_control/`: C++ examples for ROS2 Control integration.

## Getting Started

### Prerequisites

- [Pixi](https://pixi.sh) installed on your system.

### Environment Setup

Initialize the environment and install dependencies:

```bash
pixi install
```

### Building the Project

The project uses CMake and Ninja, managed by Pixi.

1. **Configure and Build C++ Library**:
   ```bash
   pixi run build
   ```
   This builds the C++ library and the Python module (`_core`) into `python/my_package/`.

2. **Install Python Package in Editable Mode**:
   ```bash
   pixi run install-python
   ```

### Running Examples

#### IsaacLab (Python)
```bash
pixi run python examples/isaaclab/example_robot.py
```

#### ROS2 Control (C++)
The C++ example can be built separately or integrated into your ROS2 workspace. To run the standalone example:
```bash
# Note: standalone example logic may require manual build steps in examples/ros2_control
```

## Dependencies

- **C++**: ONNX Runtime, Pybind11.
- **Python**: PyTorch, ONNX Runtime.
- **Build Tools**: CMake, Ninja, GCC/G++.

## Development Tasks

Specified in `pixi.toml`:
- `pixi run configure`: Run CMake configuration.
- `pixi run build`: Build the project.
- `pixi run test`: Run tests using CTest.
