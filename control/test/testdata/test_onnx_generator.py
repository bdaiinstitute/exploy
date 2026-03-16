# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import json
import logging
import os
import sys

import onnx
import torch

# Suppress torch logging warnings
logging.getLogger("torch.onnx").setLevel(logging.ERROR)

# ========== Constants ==========

INPUT_NAMES = [
    # joints
    "obj.robot1.joints.pos",
    "obj.robot1.joints.vel",
    # base
    "obj.robot1.base_name.pos_b_rt_w_in_w",
    "obj.robot1.base_name.w_Q_b",
    "obj.robot1.base_name.lin_vel_b_rt_w_in_b",
    "obj.robot1.base_name.ang_vel_b_rt_w_in_b",
    # commands
    "cmd.se2_velocity.vel",
    "cmd.se2_velocity.vel_with_range",
    "cmd.se3_pose.pose",
    "cmd.boolean.selector",
    "cmd.float.value",
    # IMU
    "sensor.imu.torso.w_Q_b",
    "sensor.imu.pelvis.ang_vel_b_rt_w_in_b",
    # sensors
    "sensor.ray_caster.one.height",
    "sensor.ray_caster.two.height",
    "sensor.range_image.one",
    "sensor.ray_caster.trail.height",
    "sensor.ray_caster.trail.r",
    "sensor.ray_caster.trail.g",
    "sensor.ray_caster.trail.b",
    "sensor.depth_image.one",
    # body
    "obj.box1.bodies.box.pos_b_rt_w_in_w",
    "obj.box1.bodies.box.w_Q_b",
    # memory
    "memory.output.joint_targets.jt1.pos.in",
    # step count
    "ctx.step_count",
    # custom extensible data for testing
    "custom.extensible_data",
]

OUTPUT_NAMES = [
    "output.joint_targets.jt1.pos",
    "output.joint_targets.jt1.vel",
    "output.joint_targets.jt1.effort",
    "output.se2_velocity.vel",
    "actions",
    "memory.output.joint_targets.jt1.pos.out",
]


# ========== Model Definition ==========


class FullTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        joint_pos,
        joint_vel,
        base_pos_in_w,
        world_Q_base,
        lin_vel_base_in_base,
        ang_vel_base_in_base,
        cmd_se2_vel,
        cmd_se2_vel_with_range,
        cmd_se3_pose,
        cmd_boolean,
        cmd_float,
        imu_data_quat,
        imu_data_ang_vel,
        heightscan,
        another_heightscan,
        range_image,
        trail_scan_height,
        trail_scan_r,
        trail_scan_g,
        trail_scan_b,
        depth_image,
        body_pos,
        body_quat,
        memory,
        step_count,
        custom_extensible_data,
    ):
        # Collect all inputs (excluding self) and ensure they are not optimized away
        inputs = [v for k, v in locals().items() if k != "self"]

        # Identity operations to keep them in the graph
        processed = [i * 1.0 for i in inputs]

        concatenated = torch.cat(processed, dim=1)

        joint_pos_targets = concatenated[:, 0:2]
        joint_vel_targets = concatenated[:, 2:4]
        joint_effort_targets = concatenated[:, 4:6]
        se2_base_velocity_target = processed[6]  # cmd_se2_vel
        actions = concatenated[:, 6:8]
        joint_targets_mem = concatenated[:, 8:10]

        return (
            joint_pos_targets,
            joint_vel_targets,
            joint_effort_targets,
            se2_base_velocity_target,
            actions,
            joint_targets_mem,
        )


# ========== Metadata Definitions ==========


def get_output_metadata() -> dict:
    """Returns metadata for model outputs."""
    return {
        "output.joint_targets.jt1": {
            "names": ["j1", "j2"],
            "stiffness": [1.0, 2.0],
            "damping": [0.1, 0.2],
        },
        "output.se2_velocity.vel": {
            "target_frame": "base_frame",
        },
    }


def get_sensor_metadata() -> dict:
    """Returns metadata for sensor inputs."""
    return {
        "sensor.ray_caster.one": {
            "pattern_type": "grid_pattern",
            "resolution": 0.1,
            "size_x": 1.6,
            "size_y": 1.0,
            "offset_x": 1.0,
            "offset_y": 0.0,
        },
        "sensor.ray_caster.two": {
            "pattern_type": "grid_pattern",
            "resolution": 0.1,
            "size_x": 1.6,
            "size_y": 1.0,
        },
        "sensor.range_image.one": {
            "pattern_type": "lidar_pattern",
            "v_res": 128,
            "h_res": 1024,
            "v_fov_min_deg": -45.0,
            "v_fov_max_deg": 45.0,
            "unobserved_value": -2.0,
        },
        "sensor.depth_image.one": {
            "pattern_type": "grid_pattern",
            "height": 1,
            "width": 1,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.5,
            "cy": 0.5,
        },
        "sensor.ray_caster.trail": {
            "pattern_type": "grid_pattern",
            "resolution": 0.1,
            "size_x": 1.6,
            "size_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        "custom.extensible_data": {
            "data_type": "extensible",
            "description": "Custom extensible data for testing user extensions",
            "dimension": 3,
        },
    }


def get_command_metadata() -> dict:
    """Returns metadata for command inputs."""
    return {
        "cmd.se2_velocity.vel": {},
        "cmd.se2_velocity.vel_with_range": {
            "ranges": {
                "lin_vel_x": [-1.5, 1.5],
                "lin_vel_y": [-0.75, 0.75],
                "ang_vel_z": [-2.5, 2.5],
            },
        },
        "cmd.float.value": {},
    }


def get_articulation_metadata() -> dict:
    """Returns metadata for articulation inputs."""
    return {
        "obj.robot1.joints": {
            "joint_names": ["j1", "j2", "j3"],
        },
    }


def get_env_metadata() -> dict:
    """Returns metadata for environment configuration."""
    return {
        "exploy_version": "0.1.0",
        "update_rate": 10.0,
        "base_names": {"robot1": "base_name"},
    }


# ========== Helper Functions ==========


def create_dummy_inputs() -> tuple:
    """Creates dummy input tensors for ONNX export."""
    # Joint inputs
    joint_pos = torch.rand((1, 3), dtype=torch.float32)
    joint_vel = torch.rand((1, 3), dtype=torch.float32)

    # Base state inputs
    base_pos = torch.rand((1, 3), dtype=torch.float32)
    base_orientation = torch.rand((1, 4), dtype=torch.float32)
    base_lin_vel = torch.rand((1, 3), dtype=torch.float32)
    base_ang_vel = torch.rand((1, 3), dtype=torch.float32)

    # Command inputs
    se2_velocity_command = torch.rand((1, 3), dtype=torch.float32)
    se3_pose_command = torch.rand((1, 7), dtype=torch.float32)
    boolean_command = torch.tensor([[True]], dtype=torch.bool)
    float_command = torch.tensor([[3.14]], dtype=torch.float32)

    # IMU inputs
    imu_data_quat = torch.rand((1, 4), dtype=torch.float32)
    imu_data_ang_vel = torch.rand((1, 3), dtype=torch.float32)

    # Sensor inputs
    heightscan = torch.rand((1, 4), dtype=torch.float32)
    range_image = torch.rand((1, 4), dtype=torch.float32)
    depth_image = torch.rand((1, 4), dtype=torch.float32)
    trail_scan = torch.rand((1, 8), dtype=torch.float32)

    # Body inputs
    body_pos = torch.rand((1, 3), dtype=torch.float32)
    body_quat = torch.rand((1, 4), dtype=torch.float32)

    # Memory and state
    memory = torch.rand((1, 2), dtype=torch.float32)
    step_count = torch.tensor([[42]], dtype=torch.int32)

    # Custom extensible data
    custom_extensible_data = torch.rand((1, 3), dtype=torch.float32)

    return (
        # joints
        joint_pos,
        joint_vel,
        # base
        base_pos,
        base_orientation,
        base_lin_vel,
        base_ang_vel,
        # commands
        se2_velocity_command,
        se2_velocity_command,
        se3_pose_command,
        boolean_command,
        float_command,
        # IMU
        imu_data_quat,
        imu_data_ang_vel,
        # sensors
        heightscan,
        heightscan,
        range_image,
        trail_scan,  # height
        trail_scan,  # r
        trail_scan,  # g
        trail_scan,  # b
        depth_image,
        # body
        body_pos,
        body_quat,
        # memory
        memory,
        # step count
        step_count,
        # custom extensible data
        custom_extensible_data,
    )


def add_metadata(path: str, metadata: dict):
    """Adds metadata properties to an ONNX model."""
    onnx_model = onnx.load(path)
    for key, val in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = json.dumps(val)
    onnx.save(onnx_model, path)


def export_model(data_dir: str):
    """Exports the test model to ONNX format with metadata."""
    output_path = os.path.join(data_dir, "test_export.onnx")

    model = FullTestModel()
    model.eval()  # Set to evaluation mode before export
    dummy_inputs = create_dummy_inputs()

    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
    )

    # Combine all metadata
    all_metadata = (
        get_env_metadata()
        | get_output_metadata()
        | get_command_metadata()
        | get_sensor_metadata()
        | get_articulation_metadata()
    )

    add_metadata(output_path, all_metadata)


class SimpleTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, float_input, int_input, bool_input):
        # Simple pass-through model that just forwards inputs to outputs
        float_output = float_input * 2.0  # Simple transformation
        int_output = int_input + 1  # Simple transformation
        bool_output = torch.logical_not(bool_input)  # Simple transformation

        return float_output, int_output, bool_output


def export_simple_model(data_dir: str):
    """Export the simple test model to ONNX format with metadata."""
    output_path_simple = os.path.join(data_dir, "test_simple.onnx")
    simple_model = SimpleTestModel()
    simple_model.eval()  # Set to evaluation mode before export

    # Create test inputs with different types
    float_input = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float32)
    int_input = torch.tensor([[10, 20, 30]], dtype=torch.int32)
    bool_input = torch.tensor([[True, False, True]], dtype=torch.bool)

    torch.onnx.export(
        simple_model,
        (float_input, int_input, bool_input),
        output_path_simple,
        input_names=["float_input", "int_input", "bool_input"],
        output_names=["float_output", "int_output", "bool_output"],
    )

    # Add simple metadata to the simple test model
    simple_metadata = {
        "model_version": "1.0",
        "model_type": "simple_test",
    }

    add_metadata(output_path_simple, simple_metadata)


# ========== Main ==========


def main():
    """Main entry point for generating test ONNX model."""
    if len(sys.argv) > 1:
        arg_path = os.path.abspath(sys.argv[1])
        data_dir = arg_path if os.path.isdir(arg_path) else os.path.dirname(arg_path)
    else:
        data_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(data_dir, exist_ok=True)

    export_simple_model(data_dir)
    export_model(data_dir)


if __name__ == "__main__":
    main()
