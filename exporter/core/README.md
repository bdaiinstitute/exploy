# exporter-core

Framework-agnostic ONNX environment exporter core.

## Overview

`exporter-core` provides the foundational components for exporting reinforcement learning environments to ONNX format. It defines abstract interfaces and core utilities that are framework-independent.

## Features

- Abstract `ExportableEnvironment` interface for environment integration
- Component abstractions (Input, Output, Memory, Connection, Group)
- ONNX export functionality via `Exporter` and `onnx_export`
- Session wrapper for ONNX Runtime inference
- Evaluation utilities for comparing PyTorch vs ONNX execution
- Tensor manipulation and context management utilities

## Installation

### From GitHub

```bash
pip install git+https://github.com/YOUR_ORG/export.git#subdirectory=exporter/core
```

### For Development

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/export.git
cd export

# Install in editable mode
pip install -e exporter/core
```

## Dependencies

- Python >= 3.11
- torch
- onnx
- onnxruntime
- numpy

## Usage

The core package provides abstract interfaces. To use with a specific framework (e.g., IsaacLab), install the corresponding framework-specific package:

```bash
pip install git+https://github.com/YOUR_ORG/export.git#subdirectory=packages/exporter-isaaclab
```

## Package Structure

- `exporter.core` - Core export functionality and abstract interfaces
- `exporter.utils` - Utility functions for tensor operations, ONNX manipulation, and paths

## License

See LICENSE file in the root directory.
