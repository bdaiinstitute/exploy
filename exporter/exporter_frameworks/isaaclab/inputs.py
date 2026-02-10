# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch
from exporter_frameworks.isaaclab.utils import (
    make_getter_body_pos_w,
    make_getter_body_quat_w,
)
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CommandManager
from isaaclab.sensors import RayCaster, SensorBase
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg

from exporter import Connection, ContextManager, Group, Input


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

        onnx_inputs = [
            Input(
                name=f"{input_name_prefix}.base.pos_b_rt_w_in_w",
                get_from_env_cb=make_getter_body_pos_w(i_body=0, articulation=articulation),
            ),
            Input(
                name=f"{input_name_prefix}.base.w_Q_b",
                get_from_env_cb=make_getter_body_quat_w(i_body=0, articulation=articulation),
            ),
            Input(
                name=f"{input_name_prefix}.base.lin_vel_b_rt_w_in_b",
                get_from_env_cb=lambda art=articulation: art.data.root_lin_vel_b,
            ),
            Input(
                name=f"{input_name_prefix}.base.ang_vel_b_rt_w_in_b",
                get_from_env_cb=lambda art=articulation: art.data.root_ang_vel_b,
            ),
        ]

        for onnx_input in onnx_inputs:
            context_manager.add_component(onnx_input)

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
    articulation: Articulation,
    sensors: dict[str, SensorBase],
    context_manager: ContextManager,
):
    sensor_prefix = "sensor"
    for sensor_name_in_source in sensors:
        sensor: SensorBase = sensors[sensor_name_in_source]

        if isinstance(sensor, RayCaster):
            # TODO: add support for all ray caster sensors.
            assert isinstance(sensor, RayCaster), "Currently only RayCaster sensors are supported."

            # TODO: add support for all pattern types.
            pattern_cfg: GridPatternCfg = sensor.cfg.pattern_cfg
            assert isinstance(pattern_cfg, GridPatternCfg), (
                "Currently only PatternBaseCfg is supported for ray caster sensors."
            )

            # Prepare an empty metadata dict.
            context_manager.add_component(
                Input(
                    name=f"{sensor_prefix}.ray_caster.{sensor_name_in_source}.height",
                    get_from_env_cb=lambda s=sensor: s._data.ray_hits_w,
                    metadata={
                        "pattern_type": "grid_pattern",
                        "offset_x": sensor.cfg.offset.pos[0],
                        "offset_y": sensor.cfg.offset.pos[1],
                        "resolution": pattern_cfg.resolution,
                        "size_x": pattern_cfg.size[0],
                        "size_y": pattern_cfg.size[1],
                    },
                )
            )

            def setter(val: torch.Tensor, sensor_name: str = sensor_name_in_source):
                sensor: RayCaster = sensors[sensor_name]
                sensor._data.pos_w[:] = val

            context_manager.add_component(
                Connection(
                    name=f"{sensor_prefix}.ray_caster.{sensor_name_in_source}.sensor_pos",
                    getter=make_getter_body_pos_w(i_body=0, articulation=articulation),
                    setter=setter,
                )
            )
        else:
            continue
