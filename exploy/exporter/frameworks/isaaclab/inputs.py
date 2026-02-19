# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CommandManager
from isaaclab.sensors import RayCaster, SensorBase
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg

from exploy.exporter.core.components import Connection
from exploy.exporter.core.context_manager import ContextManager, Group, Input


def add_commands(source: CommandManager, context_manager: ContextManager):
    cmd_prefix = "cmd"

    for command_name in source.active_terms:
        command = source.get_term(name=command_name)

        if isinstance(command, UniformVelocityCommand):
            # Capture command_name in closure to avoid B023
            def make_command_getter(cmd_name: str):
                def inner_getter() -> torch.Tensor:
                    term: UniformVelocityCommand = source.get_term(name=cmd_name)
                    return term.vel_command_b

                return inner_getter

            getter = make_command_getter(command_name)

            onnx_input = Input(
                name=f"{cmd_prefix}.se2_velocity.{command_name}",
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


def add_articulation_data(
    articulations: dict[str, Articulation],
    context_manager: ContextManager,
):
    obj_prefix = "obj"

    for obj_name, articulation in articulations.items():
        input_name_prefix = f"{obj_prefix}.{obj_name}"

        # Add inputs for all body positions and quaternions in world frame
        for i, body_name in enumerate(articulation.data.body_names):
            pos_b_rt_w_in_w = Input(
                name=f"{input_name_prefix}.{body_name}.pos_b_rt_w_in_w",
                get_from_env_cb=lambda art=articulation, idx=i: art.data.body_pos_w[:, idx],
            )
            w_Q_b = Input(
                name=f"{input_name_prefix}.{body_name}.w_Q_b",
                get_from_env_cb=lambda art=articulation, idx=i: art.data.body_quat_w[:, idx],
            )
            context_manager.add_component(pos_b_rt_w_in_w)
            context_manager.add_component(w_Q_b)

        # Add base orientation and velocities
        base_lin_vel_b_rt_w_in_b = Input(
            name=f"{input_name_prefix}.base.lin_vel_b_rt_w_in_b",
            get_from_env_cb=lambda art=articulation: art.data.root_lin_vel_b,
        )
        base_ang_vel_b_rt_w_in_b = Input(
            name=f"{input_name_prefix}.base.ang_vel_b_rt_w_in_b",
            get_from_env_cb=lambda art=articulation: art.data.root_ang_vel_b,
        )
        context_manager.add_component(base_lin_vel_b_rt_w_in_b)
        context_manager.add_component(base_ang_vel_b_rt_w_in_b)

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
    sensor_prefix = "sensor"
    for sensor_name_in_source in sensors:
        sensor: SensorBase = sensors[sensor_name_in_source]

        if isinstance(sensor, RayCaster):
            pattern_cfg: GridPatternCfg = sensor.cfg.pattern_cfg
            assert isinstance(pattern_cfg, GridPatternCfg), (
                "Currently only GridPatternCfg is supported for ray caster sensors."
            )
            context_manager.add_group(
                Group(
                    name=f"{sensor_prefix}.ray_caster.{sensor_name_in_source}",
                    metadata={
                        "pattern_type": "grid_pattern",
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
