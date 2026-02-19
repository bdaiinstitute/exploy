# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import functools

from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ActionManager
from isaaclab.envs.mdp.actions import JointAction, JointActionCfg

from exploy.exporter.core.context_manager import ContextManager, Group, Output
from exploy.exporter.frameworks.isaaclab.utils import get_articulation_actuator_gains


def add_outputs(
    action_manager: ActionManager,
    articulation: Articulation,
    context_manager: ContextManager,
):
    for active_term_name in action_manager.active_terms:
        action_term = action_manager.get_term(active_term_name)

        if isinstance(action_term, JointAction):
            cfg: JointActionCfg = action_term.cfg
            joint_names_expr = cfg.joint_names
            joint_ids, joint_names = articulation.find_joints(joint_names_expr)

            # Make getter functions for joint states.
            def get_joint_pos_target(articulation: Articulation, joint_ids: list[int]):
                return articulation.data.joint_pos_target[..., joint_ids]

            def get_joint_vel_target(articulation: Articulation, joint_ids: list[int]):
                return articulation.data.joint_vel_target[..., joint_ids]

            def get_joint_eff_target(articulation: Articulation, joint_ids: list[int]):
                return articulation.data.joint_effort_target[..., joint_ids]

            # Update metadata.
            actuator_gains = get_articulation_actuator_gains(articulation=articulation)

            onnx_joint_outputs = Group(
                name=f"output.joint_targets.{active_term_name}",
                metadata={
                    "type": "joint_targets",
                    "names": joint_names,
                    "stiffness": [actuator_gains[name]["stiffness"] for name in joint_names],
                    "damping": [actuator_gains[name]["damping"] for name in joint_names],
                },
                items=[
                    Output(
                        name="pos",
                        get_from_env_cb=functools.partial(
                            get_joint_pos_target, articulation, joint_ids.copy()
                        ),
                        metadata=None,
                    ),
                    Output(
                        name="vel",
                        get_from_env_cb=functools.partial(
                            get_joint_vel_target, articulation, joint_ids.copy()
                        ),
                        metadata=None,
                    ),
                    Output(
                        name="effort",
                        get_from_env_cb=functools.partial(
                            get_joint_eff_target, articulation, joint_ids.copy()
                        ),
                        metadata=None,
                    ),
                ],
            )

            context_manager.add_group(group=onnx_joint_outputs)
