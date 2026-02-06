# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch
from exporter_frameworks.isaaclab.utils import make_getter_body_pos_w, make_getter_body_quat_w
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CommandManager
from isaaclab.sensors import RayCaster, RayCasterCamera, SensorBase

from exporter import Connection, ContextManager, Input


def add_commands(source: CommandManager, context_manager: ContextManager):
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
                name=f"command.{command_name}",
                get_from_env_cb=getter,
                metadata={"type": "se2_velocity"},
            )

            context_manager.add_component(onnx_input)


def add_articulation_data(
    articulation: Articulation,
    context_manager: ContextManager,
):
    input_name_prefix = "articulation"
    root_body_name = articulation.body_names[0]

    onnx_inputs = [
        Input(
            name=f"{input_name_prefix}.bodies.{root_body_name}.body_pos_in_w",
            get_from_env_cb=make_getter_body_pos_w(i_body=0, articulation=articulation),
        ),
        Input(
            name=f"{input_name_prefix}.bodies.{root_body_name}.world_Q_body",
            get_from_env_cb=make_getter_body_quat_w(i_body=0, articulation=articulation),
        ),
        Input(
            name=f"{input_name_prefix}.bodies.{root_body_name}.lin_vel_body_in_body",
            get_from_env_cb=lambda: articulation.data.root_lin_vel_b,
        ),
        Input(
            name=f"{input_name_prefix}.bodies.{root_body_name}.ang_vel_body_in_body",
            get_from_env_cb=lambda: articulation.data.root_ang_vel_b,
        ),
        Input(
            name=f"{input_name_prefix}.joints.pos",
            get_from_env_cb=lambda: articulation.data.joint_pos,
        ),
        Input(
            name=f"{input_name_prefix}.joints.vel",
            get_from_env_cb=lambda: articulation.data.joint_vel,
        ),
    ]

    for onnx_input in onnx_inputs:
        context_manager.add_component(onnx_input)


def add_sensor_inputs(
    articulation: Articulation,
    sensors: dict[str, SensorBase],
    context_manager: ContextManager,
):
    for sensor_name_in_source in sensors:
        sensor: SensorBase = sensors[sensor_name_in_source]
        sensor_key: str = f"sensor.{sensor_name_in_source}"
        if isinstance(sensor, RayCaster):
            # Prepare an empty metadata dict.
            context_manager.add_component(
                Input(
                    name=sensor_key,
                    get_from_env_cb=lambda s=sensor: s._data.ray_hits_w,
                    metadata={
                        "type": "ray_caster",
                        "offset_x": sensor.cfg.offset.pos[0],
                        "offset_y": sensor.cfg.offset.pos[1],
                    },
                )
            )

            def setter(val: torch.Tensor, sensor_name: str = sensor_name_in_source):
                sensor: RayCaster = sensors[sensor_name]
                sensor._data.pos_w[:] = val

            context_manager.add_component(
                Connection(
                    name=f"{sensor_key}.pos",
                    getter=make_getter_body_pos_w(i_body=0, articulation=articulation),
                    setter=setter,
                )
            )
        elif isinstance(sensor, RayCasterCamera):
            context_manager.add_component(
                Input(
                    name=sensor_key,
                    get_from_env_cb=lambda s=sensor: s._data.output["distance_to_image_plane"],
                    metadata={
                        "type": "depth_image",
                        "offset_x": sensor.cfg.offset.pos[0],
                        "offset_y": sensor.cfg.offset.pos[1],
                    },
                )
            )
        else:
            continue
