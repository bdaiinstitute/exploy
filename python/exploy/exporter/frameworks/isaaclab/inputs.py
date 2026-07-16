# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CommandManager
from isaaclab.sensors import RayCaster, SensorBase
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg, PatternBaseCfg

from exploy.exporter.core.context_manager import ContextManager, Group, Input
from exploy.exporter.frameworks.isaaclab import utils
from exploy.exporter.frameworks.isaaclab.derived_tensors import (
    body_link_ang_vel_b,
    body_link_lin_vel_b,
)

OBJ_PREFIX = "obj"
SENSOR_PREFIX = "sensor"
CMD_PREFIX = "cmd"


def add_commands(
    command_manager: CommandManager,
    context_manager: ContextManager,
):
    """Add command inputs from the command manager to the context manager.

    This function processes all active command terms in the command manager and adds them
    as inputs to the context manager. Currently supports UniformVelocityCommand.

    Args:
        command_manager: The IsaacLab command manager containing active command terms.
        context_manager: The context manager to add command inputs to.
    """
    for command_name in command_manager.active_terms:
        command = command_manager.get_term(name=command_name)

        if isinstance(command, UniformVelocityCommand):
            # Capture command_name in closure to avoid B023
            def make_command_getter(cmd_name: str):
                def inner_getter() -> torch.Tensor:
                    term: UniformVelocityCommand = command_manager.get_term(name=cmd_name)
                    return term.vel_command_b

                return inner_getter

            getter = make_command_getter(command_name)

            onnx_input = Input(
                name=f"{CMD_PREFIX}.se2_velocity.{command_name}",
                get_from_env_cb=getter,
                metadata={
                    "ranges": {
                        "lin_vel_x": command.cfg.ranges.lin_vel_x,
                        "lin_vel_y": command.cfg.ranges.lin_vel_y,
                        "ang_vel_z": command.cfg.ranges.ang_vel_z,
                    },
                },
            )

            context_manager.add_component(onnx_input)


def add_body_pos(
    articulations: dict[str, Articulation],
    context_manager: ContextManager,
):
    """Add body position inputs for all articulations, skipping root bodies.

    For each articulation, this function adds inputs for the position
    of each body belonging to that articulation.

    Args:
        articulations: Dictionary mapping object names to Articulation instances.
        context_manager: The context manager to add body pose inputs to.
    """

    # Add inputs for all body positions in world frame
    for obj_name, articulation in articulations.items():
        root_name = articulation.body_names[0]
        for body_name in articulation.data.body_names:
            if body_name == root_name:
                continue
            body_ids, _ = articulation.find_bodies(body_name)
            assert len(body_ids) == 1, (
                f"Body name {body_name} is not unique in articulation {obj_name}. "
                f"Found body IDs: {body_ids}"
            )
            body_idx = body_ids[0]
            pos_b_rt_w_in_w = Input(
                name=f"{OBJ_PREFIX}.{obj_name}.{body_name}.pos_b_rt_w_in_w",
                get_from_env_cb=lambda art=articulation, idx=body_idx: art.data.body_pos_w[:, idx],
            )
            context_manager.add_component(pos_b_rt_w_in_w)


def add_body_quat(
    articulations: dict[str, Articulation],
    context_manager: ContextManager,
):
    """Add body orientation inputs for all articulations, skipping root bodies.

    For each articulation, this function adds inputs for the quaternion
    of each body belonging to that articulation.

    Args:
        articulations: Dictionary mapping object names to Articulation instances.
        context_manager: The context manager to add body pose inputs to.
    """
    # Add inputs for all body quaternions in world frame
    for obj_name, articulation in articulations.items():
        root_name = articulation.body_names[0]
        for body_name in articulation.data.body_names:
            if body_name == root_name:
                continue
            body_ids, _ = articulation.find_bodies(body_name)
            assert len(body_ids) == 1, (
                f"Body name {body_name} is not unique in articulation {obj_name}. "
                f"Found body IDs: {body_ids}"
            )
            body_idx = body_ids[0]
            w_Q_b = Input(
                name=f"{OBJ_PREFIX}.{obj_name}.{body_name}.w_Q_b",
                get_from_env_cb=lambda art=articulation, idx=body_idx: art.data.body_quat_w[:, idx],
            )
            context_manager.add_component(w_Q_b)


def add_body_pos_and_quat(
    articulations: dict[str, Articulation],
    context_manager: ContextManager,
):
    """Add body position and orientation inputs for all articulations, skipping root bodies.

    For each articulation, this function adds inputs for the position and quaternion
    of each body belonging to that articulation.

    Args:
        articulations: Dictionary mapping object names to Articulation instances.
        context_manager: The context manager to add body pose inputs to.
    """
    # Add inputs for all body positions and quaternions in world frame
    add_body_pos(
        articulations=articulations,
        context_manager=context_manager,
    )
    add_body_quat(
        articulations=articulations,
        context_manager=context_manager,
    )


def add_body_vel(
    articulations: dict[str, Articulation],
    context_manager: ContextManager,
):
    """Add body velocity inputs for all articulations, skipping root bodies.

    For each articulation, this function adds inputs for the body linear and angular
    velocities relative to the world/inertial frame expressed in their local frame.

    Stock Isaac Lab only exposes per-body velocities in the world frame, so the body-frame
    quantities are derived from the world-frame link velocities and the body orientations. They
    are materialized as stored per-body leaves against the `ArticulationDataSource` when
    `IsaacLabExportableEnvironment` swaps it in, so the input callbacks and observation functions
    read the exact stored objects during tracing, making these inputs ONNX graph inputs.

    During evaluation, after the original articulation data has been restored, the input callbacks
    fall back to computing the same quantities from the live world-frame data.

    Args:
        articulations: Dictionary mapping object names to Articulation instances.
        context_manager: The context manager to add body velocity inputs to.
    """
    # Add inputs for all body velocities expressed in their local (body) frame.
    for obj_name, articulation in articulations.items():
        root_name = articulation.body_names[0]
        for body_name in articulation.data.body_names:
            if body_name == root_name:
                continue
            body_ids, _ = articulation.find_bodies(body_name)
            assert len(body_ids) == 1, (
                f"Body name {body_name} is not unique in articulation {obj_name}. Found body IDs: {body_ids}"
            )
            body_idx = body_ids[0]
            input_name_prefix = f"{OBJ_PREFIX}.{obj_name}.{body_name}"
            lin_vel_b_rt_w_in_b = Input(
                name=f"{input_name_prefix}.lin_vel_b_rt_w_in_b",
                get_from_env_cb=lambda art=articulation, idx=body_idx: body_link_lin_vel_b(
                    art.data
                )[:, idx],
            )
            ang_vel_b_rt_w_in_b = Input(
                name=f"{input_name_prefix}.ang_vel_b_rt_w_in_b",
                get_from_env_cb=lambda art=articulation, idx=body_idx: body_link_ang_vel_b(
                    art.data
                )[:, idx],
            )
            context_manager.add_component(lin_vel_b_rt_w_in_b)
            context_manager.add_component(ang_vel_b_rt_w_in_b)


def add_base_vel(
    articulations: dict[str, Articulation],
    context_manager: ContextManager,
):
    """Add base velocity inputs for all articulations.

    For each articulation, this function adds inputs for the base linear and angular
    velocities in the base frame.

    Args:
        articulations: Dictionary mapping object names to Articulation instances.
        context_manager: The context manager to add base velocity inputs to.
    """
    for obj_name, articulation in articulations.items():
        input_name_prefix = f"{OBJ_PREFIX}.{obj_name}.{articulation.body_names[0]}"

        # Add base linear and angular velocities expressed in base frame.
        lin_vel_b_rt_w_in_b = Input(
            name=f"{input_name_prefix}.lin_vel_b_rt_w_in_b",
            get_from_env_cb=lambda art=articulation: art.data.root_lin_vel_b,
        )
        ang_vel_b_rt_w_in_b = Input(
            name=f"{input_name_prefix}.ang_vel_b_rt_w_in_b",
            get_from_env_cb=lambda art=articulation: art.data.root_ang_vel_b,
        )
        context_manager.add_component(lin_vel_b_rt_w_in_b)
        context_manager.add_component(ang_vel_b_rt_w_in_b)


def add_base_pose(
    articulations: dict[str, Articulation],
    context_manager: ContextManager,
):
    """Add base pose inputs for all articulations.

    For each articulation, this function adds inputs for the base position and orientation
    in the world frame.

    Args:
        articulations: Dictionary mapping object names to Articulation instances.
        context_manager: The context manager to add base pose inputs to.
    """
    for obj_name, articulation in articulations.items():
        input_name_prefix = f"{OBJ_PREFIX}.{obj_name}.{articulation.body_names[0]}"

        # Add base position and orientation
        pos_b_rt_w_in_w = Input(
            name=f"{input_name_prefix}.pos_b_rt_w_in_w",
            get_from_env_cb=lambda art=articulation: art.data.root_pos_w,
        )
        w_Q_b = Input(
            name=f"{input_name_prefix}.w_Q_b",
            get_from_env_cb=lambda art=articulation: art.data.root_quat_w,
        )
        context_manager.add_component(pos_b_rt_w_in_w)
        context_manager.add_component(w_Q_b)


def add_joint_pos_and_vel(
    articulations: dict[str, Articulation],
    context_manager: ContextManager,
):
    """Add joint position and velocity inputs for all articulations.

    For each articulation, this function creates a group containing joint positions
    and velocities, along with metadata about joint names.

    Args:
        articulations: Dictionary mapping object names to Articulation instances.
        context_manager: The context manager to add joint state inputs to.
    """
    for obj_name, articulation in articulations.items():
        input_name_prefix = f"{OBJ_PREFIX}.{obj_name}"

        joint_group = Group(
            name=f"{input_name_prefix}.joints",
            items=[
                Input(
                    name="pos",
                    get_from_env_cb=lambda art=articulation: art.data.joint_pos,
                ),
                Input(
                    name="vel",
                    get_from_env_cb=lambda art=articulation: art.data.joint_vel,
                ),
            ],
            metadata={
                "joint_names": articulation.joint_names,
            },
        )

        context_manager.add_group(joint_group)


def add_sensor_inputs(
    sensors: dict[str, SensorBase],
    context_manager: ContextManager,
):
    """Add sensor inputs to the context manager.

    This function processes all sensors and adds their data as inputs to the context manager.
    Currently supports RayCaster sensors with GridPatternCfg.

    Args:
        sensors: Dictionary mapping sensor names to SensorBase instances.
        context_manager: The context manager to add sensor inputs to.
    """
    for sensor_name_in_source in sensors:
        sensor: SensorBase = sensors[sensor_name_in_source]

        if type(sensor) is RayCaster:
            pattern_cfg: PatternBaseCfg = sensor.cfg.pattern_cfg
            if isinstance(pattern_cfg, GridPatternCfg):
                context_manager.add_group(
                    Group(
                        name=f"{SENSOR_PREFIX}.ray_caster.{sensor_name_in_source}",
                        metadata={
                            "pattern_type": "grid_pattern",
                            "articulation_name": utils.prim_path_to_articulation_name(
                                prim_path=sensor.cfg.prim_path
                            ),
                            "offset_x": sensor.cfg.offset.pos[0],
                            "offset_y": sensor.cfg.offset.pos[1],
                            "resolution": pattern_cfg.resolution,
                            "size_x": pattern_cfg.size[0],
                            "size_y": pattern_cfg.size[1],
                        },
                        items=[
                            Input(
                                name="height",
                                get_from_env_cb=lambda s=sensor: s._data.ray_hits_w[..., 2],
                            ),
                        ],
                    )
                )
