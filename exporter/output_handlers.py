# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import functools
import torch

from exporter import ExportEnvIdsType
# from rai.core.utils.managers import get_articulation_actuator_gains

from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ActionManager
from isaaclab.envs.mdp.actions import JointAction, JointActionCfg


class ActionHandler:
    """Manage action terms in an action manager and generate outputs.

    This class handles an `ActionManager` and iterates through action terms. For each term,
    it stores a dictionary containing:
        - the action term name
        - a callable returning the processed action output

    It also stores metadata, for each action term, in the form of a dictionary with information
    related to that action term.
    """

    def __init__(self, action_manager: ActionManager, articulation: Articulation, env_id: ExportEnvIdsType):
        self._data = {}
        self._metadata = {}

        self.add_from_action_manager(action_manager=action_manager, articulation=articulation, env_id=env_id)

    def add_from_action_manager(
        self,
        action_manager: ActionManager,
        articulation: Articulation,
        env_id: ExportEnvIdsType,
    ):
        for active_term_name in action_manager.active_terms:
            action_term = action_manager.get_term(active_term_name)
            data_key = f"output.{active_term_name}"

            if isinstance(action_term, JointAction):
                cfg: JointActionCfg = action_term.cfg
                joint_names_expr = cfg.joint_names
                joint_ids, joint_names = articulation.find_joints(joint_names_expr)

                def export_env_ids_type_indexing(asset: torch.Tensor, env_id: ExportEnvIdsType, joint_ids: list[int]):
                    if isinstance(env_id, int):
                        return asset[env_id, joint_ids]
                    else:
                        return asset[env_id][:, joint_ids]

                # Make getter functions for joint states.
                def get_joint_pos_target(articulation: Articulation, env_id: ExportEnvIdsType, joint_ids: list[int]):
                    return export_env_ids_type_indexing(articulation.data.joint_pos_target, env_id, joint_ids)

                def get_joint_vel_target(articulation: Articulation, env_id: ExportEnvIdsType, joint_ids: list[int]):
                    return export_env_ids_type_indexing(articulation.data.joint_vel_target, env_id, joint_ids)

                def get_joint_eff_target(articulation: Articulation, env_id: ExportEnvIdsType, joint_ids: list[int]):
                    return export_env_ids_type_indexing(articulation.data.joint_effort_target, env_id, joint_ids)

                # Update data dictionary.
                self._data[f"{data_key}.pos"] = functools.partial(
                    get_joint_pos_target, articulation, env_id, joint_ids.copy()
                )
                self._data[f"{data_key}.vel"] = functools.partial(
                    get_joint_vel_target, articulation, env_id, joint_ids.copy()
                )
                self._data[f"{data_key}.effort"] = functools.partial(
                    get_joint_eff_target, articulation, env_id, joint_ids.copy()
                )

                # Update metadata.
                # actuator_gains = get_articulation_actuator_gains(articulation=articulation)
                actuator_gains = {name: {"stiffness": 0.0, "damping": 0.0} for name in joint_names}
                self._metadata[data_key] = {
                    "type": "joint_targets",
                    "names": joint_names,
                    "stiffness": [actuator_gains[name]["stiffness"] for name in joint_names],
                    "damping": [actuator_gains[name]["damping"] for name in joint_names],
                }
            elif type(action_term).__name__ == "VelocityPreTrainedPolicyAction":
                self._data[f"{data_key}"] = lambda: action_term.processed_actions
                self._metadata[data_key] = {
                    "type": "se2_velocity",
                    "target_frame": "base",
                }
            else:
                raise RuntimeError(f"Got unhandled action term type: {type(action_term)}")

    def metadata(self) -> dict:
        return self._metadata

    def data(self) -> dict:
        return self._data

    def names(self) -> list[str]:
        return list(self._data.keys())

    def values(self) -> list[callable]:
        return [func() for func in self._data.values()]
