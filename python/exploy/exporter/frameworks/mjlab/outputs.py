# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import functools

from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.managers.action_manager import ActionManager

from exploy.exporter.core.context_manager import ContextManager, Group, Output
from exploy.exporter.frameworks.mjlab.utils import get_entity_actuator_gains

OUTPUT_PREFIX = "output"


def add_outputs(
    action_manager: ActionManager,
    context_manager: ContextManager,
) -> None:
    """Add joint target outputs for all action terms in the action manager.

    Iterates all active action terms and adds their joint target outputs.
    Currently supports ``BaseAction`` (and all its subclasses such as
    ``JointPositionAction``, ``JointVelocityAction``, ``JointEffortAction``).

    Args:
        action_manager: The MjLab action manager containing active action terms.
        context_manager: The context manager to add components to.
    """
    for action_term_name in action_manager.active_terms:
        action_term = action_manager.get_term(action_term_name)

        if not isinstance(action_term, BaseAction):
            continue

        robot = action_term._entity
        joint_names = list(robot.joint_names)
        joint_ids = list(range(len(joint_names)))

        actuator_gains = get_entity_actuator_gains(robot)

        def get_pos(entity, ids):
            return entity.data.joint_pos_target[..., ids]

        def get_vel(entity, ids):
            return entity.data.joint_vel_target[..., ids]

        def get_effort(entity, ids):
            return entity.data.joint_effort_target[..., ids]

        context_manager.add_group(
            Group(
                name=f"{OUTPUT_PREFIX}.joint_targets.{action_term.cfg.entity_name}.{action_term_name}",
                metadata={
                    "type": "joint_targets",
                    "names": joint_names,
                    "stiffness": [
                        actuator_gains.get(n, {}).get("stiffness", 0.0) for n in joint_names
                    ],
                    "damping": [actuator_gains.get(n, {}).get("damping", 0.0) for n in joint_names],
                },
                items=[
                    Output(
                        name="pos",
                        get_from_env_cb=functools.partial(get_pos, robot, joint_ids.copy()),
                    ),
                    Output(
                        name="vel",
                        get_from_env_cb=functools.partial(get_vel, robot, joint_ids.copy()),
                    ),
                    Output(
                        name="effort",
                        get_from_env_cb=functools.partial(get_effort, robot, joint_ids.copy()),
                    ),
                ],
            )
        )
