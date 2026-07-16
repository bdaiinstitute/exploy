# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from mjlab.sensor import RayCastSensor
from mjlab.tasks.velocity.mdp import UniformVelocityCommand

from exploy.exporter.core.context_manager import ContextManager, Group, Input
from exploy.exporter.frameworks.mjlab.derived_tensors import (
    body_link_ang_vel_b,
    body_link_lin_vel_b,
)

OBJ_PREFIX = "obj"
SENSOR_PREFIX = "sensor"
CMD_PREFIX = "cmd"


def add_base_com_vel(entities: dict, context_manager: ContextManager) -> None:
    """Add root COM linear and angular velocity inputs for all entities.

    Reads from ``entity.data.root_com_lin_vel_b`` and ``root_com_ang_vel_b``,
    which go through the ``EntityDataSource`` proxy and are therefore traced
    as ONNX graph inputs.

    Args:
        entities: Dict mapping entity name to Entity objects.
        context_manager: The context manager to add inputs to.
    """
    for obj_name, entity in entities.items():
        root_body_name = entity.root_body.name.split("/")[-1]
        prefix = f"{OBJ_PREFIX}.{obj_name}.{root_body_name}"
        context_manager.add_component(
            Input(
                name=f"{prefix}.lin_vel_b_rt_w_in_b",
                get_from_env_cb=lambda _entity=entity: _entity.data.root_com_lin_vel_b,
            )
        )
        context_manager.add_component(
            Input(
                name=f"{prefix}.ang_vel_b_rt_w_in_b",
                get_from_env_cb=lambda _entity=entity: _entity.data.root_com_ang_vel_b,
            )
        )


def add_body_pos_and_quat(entities: dict, context_manager: ContextManager) -> None:
    """Add body position and orientation inputs for all entities.

    Reads from ``entity.data.body_link_pos_w`` and ``body_link_quat_w`` sliced
    per body, going through the proxy so the slice is an ONNX graph input.

    Args:
        entities: Dict mapping entity name to Entity objects.
        context_manager: The context manager to add inputs to.
    """
    for obj_name, entity in entities.items():
        for i, body_name in enumerate(entity.body_names):
            context_manager.add_component(
                Input(
                    name=f"{OBJ_PREFIX}.{obj_name}.{body_name}.pos_b_rt_w_in_w",
                    get_from_env_cb=lambda _entity=entity, idx=i: _entity.data.body_link_pos_w[
                        :, idx
                    ],
                )
            )
            context_manager.add_component(
                Input(
                    name=f"{OBJ_PREFIX}.{obj_name}.{body_name}.w_Q_b",
                    get_from_env_cb=lambda _entity=entity, idx=i: _entity.data.body_link_quat_w[
                        :, idx
                    ],
                )
            )


def add_body_vel(entities: dict, context_manager: ContextManager) -> None:
    """Add body velocity inputs for all entities, skipping root bodies.

    For each entity, this adds the linear and angular velocities of every non-root body relative
    to the world/inertial frame, expressed in that body's own (actor/link) frame.

    mjlab only stores per-body velocities in the world frame, so the body-frame quantities are
    derived from the world-frame link velocities and the body orientations. They are materialized
    as stored per-body leaves against the ``EntityDataSource`` when
    ``MjlabExportableEnvironment`` swaps it in, so the input callbacks and observation functions
    read the exact stored objects during tracing, making these inputs ONNX graph inputs.

    During evaluation, after the original entity data has been restored, the input callbacks fall
    back to computing the same quantities from the live world-frame data.

    Args:
        entities: Dict mapping entity name to Entity objects.
        context_manager: The context manager to add inputs to.
    """
    for obj_name, entity in entities.items():
        root_body_name = entity.root_body.name.split("/")[-1]
        for i, body_name in enumerate(entity.body_names):
            if body_name == root_body_name:
                continue
            prefix = f"{OBJ_PREFIX}.{obj_name}.{body_name}"
            context_manager.add_component(
                Input(
                    name=f"{prefix}.lin_vel_b_rt_w_in_b",
                    get_from_env_cb=lambda _entity=entity, idx=i: body_link_lin_vel_b(_entity.data)[
                        :, idx
                    ],
                )
            )
            context_manager.add_component(
                Input(
                    name=f"{prefix}.ang_vel_b_rt_w_in_b",
                    get_from_env_cb=lambda _entity=entity, idx=i: body_link_ang_vel_b(_entity.data)[
                        :, idx
                    ],
                )
            )


def add_joint_pos_and_vel(entities: dict, context_manager: ContextManager) -> None:
    """Add joint position and velocity inputs for all entities.

    Args:
        entities: Dict mapping entity name to Entity objects.
        context_manager: The context manager to add inputs to.
    """
    for obj_name, entity in entities.items():
        joint_group = Group(
            name=f"{OBJ_PREFIX}.{obj_name}.joints",
            items=[
                Input(
                    name="pos",
                    get_from_env_cb=lambda _entity=entity: _entity.data.joint_pos,
                ),
                Input(
                    name="vel",
                    get_from_env_cb=lambda _entity=entity: _entity.data.joint_vel,
                ),
            ],
            metadata={
                "joint_names": entity.joint_names,
            },
        )
        context_manager.add_group(joint_group)


def add_commands(
    command_manager,
    context_manager: ContextManager,
) -> None:
    """Add command inputs from the command manager to the context manager.

    Iterates all active command terms and adds them as inputs. Currently supports
    ``UniformVelocityCommand``.

    Args:
        command_manager: The MjLab command manager containing active command terms.
        context_manager: The context manager to add command inputs to.
    """
    for command_name in command_manager.active_terms:
        command = command_manager.get_term(name=command_name)

        if isinstance(command, UniformVelocityCommand):

            def make_getter(cmd_name: str):
                def getter() -> torch.Tensor:
                    return command_manager.get_term(name=cmd_name).command

                return getter

            context_manager.add_component(
                Input(
                    name=f"{CMD_PREFIX}.se2_velocity.{command_name}",
                    get_from_env_cb=make_getter(command_name),
                    metadata={
                        "ranges": {
                            "lin_vel_x": command.cfg.ranges.lin_vel_x,
                            "lin_vel_y": command.cfg.ranges.lin_vel_y,
                            "ang_vel_z": command.cfg.ranges.ang_vel_z,
                        },
                    },
                )
            )


def add_sensor_inputs(
    sensors: dict,
    context_manager: ContextManager,
) -> None:
    """Add sensor inputs to the context manager.

    Iterates all sensors and adds their data as inputs. Currently supports
    ``RayCastSensor``.

    Args:
        sensors: Dictionary mapping sensor names to sensor objects.
        context_manager: The context manager to add sensor inputs to.
    """
    for sensor_name, sensor in sensors.items():
        if type(sensor) is RayCastSensor:
            pattern_cfg = sensor.cfg.pattern
            frames = sensor.cfg.frame
            frame = frames[0] if isinstance(frames, tuple) else frames
            metadata = {
                "pattern_type": "grid_pattern",
                "articulation_name": frame.entity,
                "offset_x": sensor.cfg.offset.pos[0],
                "offset_y": sensor.cfg.offset.pos[1],
                "resolution": pattern_cfg.resolution,
                "size_x": pattern_cfg.size[0],
                "size_y": pattern_cfg.size[1],
            }
            context_manager.add_group(
                Group(
                    name=f"{SENSOR_PREFIX}.ray_caster.{sensor_name}",
                    metadata=metadata,
                    items=[
                        Input(
                            name="height",
                            get_from_env_cb=lambda _sensor=sensor: _sensor._data.hit_pos_w[..., 2],
                        ),
                    ],
                )
            )
