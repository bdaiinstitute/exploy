# exporter-isaaclab

IsaacLab-specific ONNX environment exporter implementation.

## Overview

`exporter-isaaclab` provides a concrete implementation of the exporter-core interfaces for [NVIDIA IsaacLab](https://github.com/isaac-sim/IsaacLab) environments. It enables exporting trained RL policies to ONNX format for deployment.

## Features

- `IsaacLabExportableEnvironment` - Integration with IsaacLab's `ManagerBasedRLEnv`
- IsaacLab-specific data sources (ArticulationDataSource, RigidObjectDataSource)
- Input/Output/Memory handlers for IsaacLab observations and actions
- Utilities for articulation and rigid body data extraction
- Training-integrated ONNX export workflows

## Installation

### From GitHub

```bash
# Install exporter-core (automatically included as dependency)
pip install git+https://github.com/YOUR_ORG/export.git#subdirectory=exporter/frameworks/isaaclab
```

### For Development

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/export.git
cd export

# Install IsaacLab first (see IsaacLab documentation)
# Then install exporter-isaaclab in editable mode
pip install -e exporter/frameworks/isaaclab
```

## Dependencies

- Python >= 3.11
- exporter-core (installed automatically)
- torch
- numpy
- gymnasium == 1.2.0
- tqdm
- isaacsim[all,extscache] == 5.1.0 (includes isaaclab)

## Usage

```python
from isaaclab.envs import ManagerBasedRLEnv
from exporter.isaaclab.env import IsaacLabExportableEnvironment
from exporter.core.exporter import onnx_export

# Create your IsaacLab environment
env = ManagerBasedRLEnv(cfg)

# Wrap it for export
exportable_env = IsaacLabExportableEnvironment(env, env_id=0)

# Export to ONNX
onnx_export(
    exportable_env,
    output_path="policy.onnx",
    # ... other parameters
)
```

See `examples/exporter_scripts/export_isaaclab.py` for a complete example.

## Package Structure

- `exporter.isaaclab` - IsaacLab-specific implementation
  - `env.py` - IsaacLabExportableEnvironment
  - `inputs.py` - Input handlers for observations
  - `outputs.py` - Output handlers for actions
  - `memory.py` - Memory handlers for recurrent policies
  - `articulation_data.py` - Articulation data extraction
  - `rigid_object_data.py` - Rigid body data extraction
  - `utils.py` - Helper utilities

## License

See LICENSE file in the root directory.
