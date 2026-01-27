# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import json
import os
import sys

import torch

import onnx


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        joint_pos,
        joint_vel,
        pos_base_in_w,
        world_Q_base,
        lin_vel_base_in_base,
        ang_vel_base_in_base,
        cmd_se2_vel,
        cmd_se2_vel_with_range,
        cmd_se3_pose,
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
        body_quat,
        memory,
    ):
        # use all inputs and ensure they are not optimized away
        inputs = [
            joint_pos,
            joint_vel,
            pos_base_in_w,
            world_Q_base,
            lin_vel_base_in_base,
            ang_vel_base_in_base,
            cmd_se2_vel,
            cmd_se2_vel_with_range,
            cmd_se3_pose,
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
            body_quat,
            memory,
        ]
        
        # Identity operations to keep them in the graph
        processed = [i * 1.0 for i in inputs]
        
        concatenated = torch.cat(processed, dim=1)
        
        joint_pos_targets = concatenated[:, 0:2]
        joint_vel_targets = concatenated[:, 2:4]
        joint_effort_targets = concatenated[:, 4:6]
        se2_base_velocity_target = processed[6] # cmd_se2_vel
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


def add_metadata(path: str, metadata: dict):
    onnx_model = onnx.load(path)
    for key, val in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = json.dumps(val)
    onnx.save(onnx_model, path)


def main():
    # Define dummy inputs
    joint_pos = torch.rand((1, 3), dtype=torch.float32)
    joint_vel = torch.rand((1, 3), dtype=torch.float32)
    pos_base_in_w = torch.rand((1, 3), dtype=torch.float32)
    world_Q_base = torch.rand((1, 4), dtype=torch.float32)
    lin_vel_base_in_base = torch.rand((1, 3), dtype=torch.float32)
    ang_vel_base_in_base = torch.rand((1, 3), dtype=torch.float32)
    se2_velocity_command = torch.rand((1, 3), dtype=torch.float32)
    se3_pose_command = torch.rand((1, 7), dtype=torch.float32)
    imu_data_quat = torch.rand((1, 4), dtype=torch.float32)
    imu_data_ang_vel = torch.rand((1, 3), dtype=torch.float32)
    heightscan = torch.rand((1, 4), dtype=torch.float32)
    range_image = torch.rand((1, 4), dtype=torch.float32)
    depth_image = torch.rand((1, 4), dtype=torch.float32)
    trail_scan = torch.rand((1, 8), dtype=torch.float32)
    body_quat = torch.rand((1, 4), dtype=torch.float32)
    memory = torch.rand((1, 2), dtype=torch.float32)

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Store model
    output_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(data_dir, "test.onnx")
    model = TestModel()
    torch.onnx.export(
        model,
        (
            joint_pos,
            joint_vel,
            pos_base_in_w,
            world_Q_base,
            lin_vel_base_in_base,
            ang_vel_base_in_base,
            se2_velocity_command,
            se2_velocity_command,
            se3_pose_command,
            imu_data_quat,
            imu_data_ang_vel,
            heightscan,
            heightscan,
            range_image,
            trail_scan,  # height
            trail_scan,  # r
            trail_scan,  # g
            trail_scan,  # b
            depth_image,
            body_quat,
            memory,
        ),
        output_path,
        input_names=[
            "joint_pos",
            "joint_vel",
            "pos_base_in_w",
            "world_Q_base",
            "lin_vel_base_in_base",
            "ang_vel_base_in_base",
            "command.se2_vel",
            "command.se2_vel_with_range",
            "command.se3_pose",
            "imu_data.quat.imu1",
            "imu_data.ang_vel.imu1",
            "sensor.height_scanner",
            "sensor.another_height_scanner",
            "sensor.lidar_scanner",
            "sensor.trail_scanner.height",
            "sensor.trail_scanner.r",
            "sensor.trail_scanner.g",
            "sensor.trail_scanner.b",
            "sensor.depth_image",
            "body.hand.quat",
            "memory.output.joint_targets.pos.in",
        ],
        output_names=[
            "output.joint_targets.pos",
            "output.joint_targets.vel",
            "output.joint_targets.effort",
            "output.se2_base_velocity",
            "actions",
            "memory.output.joint_targets.pos.out",
        ],
    )

    output_metadata = {
        "output.joint_targets": {
            "type": "joint_targets",
            "names": ["j1", "j2"],
            "stiffness": [1.0, 2.0],
            "damping": [0.1, 0.2],
        },
        "output.se2_base_velocity": {
            "type": "se2_velocity",
            "target_frame": "base_frame",
        },
    }

    sensor_metadata = {
        "sensor.height_scanner": {
            "type": "ray_caster",
            "pattern_type": "grid_pattern",
            "resolution": 0.1,
            "size_x": 1.6,
            "size_y": 1.0,
            "offset_x": 1.0,
            "offset_y": 0.0,
        },
        "sensor.another_height_scanner": {
            "type": "ray_caster",
            "pattern_type": "grid_pattern",
            "resolution": 0.1,
            "size_x": 1.6,
            "size_y": 1.0,
        },
        "sensor.lidar_scanner": {
            "type": "lidar_range_image",
            "pattern_type": "lidar_pattern",
            "v_res": 128,
            "h_res": 1024,
            "v_fov_min_deg": -45.0,
            "v_fov_max_deg": 45.0,
            "unobserved_value": -2.0,
        },
        "sensor.trail_scanner": {
            "type": "trail_ray_caster",
            "pattern_type": "grid_pattern",
            "resolution": 0.1,
            "size_x": 1.6,
            "size_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        "sensor.depth_image": {
            "type": "depth_image",
            "pattern_type": "grid_pattern",
            "height": 1,
            "width": 1,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.5,
            "cy": 0.5,
        },
    }

    command_metadata = {
        "command.se3_pose": {"type": "se3_pose"},
        "command.se2_vel": {"type": "se2_velocity"},
        "command.se2_vel_with_range": {
            "type": "se2_velocity",
            "ranges": {
                "lin_vel_x": [-1.5, 1.5],
                "lin_vel_y": [-0.75, 0.75],
                "ang_vel_z": [-2.5, 2.5],
            },
        },
    }

    add_metadata(
        output_path,
        {
            "policy_dt": 0.1,
            "decimation": 2.0,
            "outputs": output_metadata,
            "commands": command_metadata,
            "sensors": sensor_metadata,
        },
    )

    # Remove as soon as we remove joint_names support.
    onnx_model = onnx.load(output_path)
    meta = onnx_model.metadata_props.add()
    meta.key = "joint_names"
    meta.value = "j1,j2,j3"
    onnx.save(onnx_model, output_path)


if __name__ == "__main__":
    main()
