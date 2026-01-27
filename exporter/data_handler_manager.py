# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch

from exporter import ExportEnvIdsType

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv

from .data_handlers import (
    ArticulationDataHandler,
    CommandDataHandler,
    IMUDataHandler,
    RigidObjectDataHandler,
    SensorDataHandler,
)
from .memory_handlers import get_memory_handler_from_env
from .output_handlers import ActionHandler


class DataHandlerManager:

    # NOTE: Some terms aren't DataHandlers (e.g. MemoryHandler, ActionHandler) but are included here for convenience.
    # They follow a similar interface. However, they should be refactored at some point to inherit from DataHandler.

    _env: ManagerBasedRLEnv

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        env_id: ExportEnvIdsType,
        robot_name: str = "robot",
        policy_group_name: str = "policy",
        device: str = "cpu",
    ):
        self._env = env.unwrapped if hasattr(env, "unwrapped") else env
        self._env_id = env_id
        self._robot_name = robot_name
        self._policy_group_name = policy_group_name
        self._device = device

        # Get the articulation
        self._articulation: Articulation = self._env.scene[self._robot_name]

        # Initialize data handlers
        self.art_dh = ArticulationDataHandler(
            articulation=self._articulation,
            env_id=self._env_id,
            device=self._device,
        )
        self.command_dh = CommandDataHandler(
            command_manager=self._env.command_manager,
            env_id=self._env_id,
            device=self._device,
        )
        self.rigid_object_dh = RigidObjectDataHandler(
            rigid_body_data=self._env.scene.rigid_objects,
            env_id=self._env_id,
            device=self._device,
        )
        self.imu_dh = IMUDataHandler(
            sensors=self._env.scene.sensors,
            articulation=self._articulation,
            env_id=self._env_id,
            device=self._device,
        )
        self.sensor_dh = SensorDataHandler(
            sensors=self._env.scene.sensors,
            env_id=self._env_id,
            device=self._device,
        )
        self.memory_dh = get_memory_handler_from_env(  # NOTE: MemoryHandler is not a DataHandler
            env=self._env,
            env_id=self._env_id,
            device=self._device,
        )
        self.action_dh = ActionHandler(  # NOTE: ActionHandler is not a DataHandler + device is not used
            action_manager=self._env.action_manager,
            articulation=self._articulation,
            env_id=self._env_id,
        )

        # Add any dependent body state to the articulation data handler.
        self.art_dh.add_body_states(
            articulation=self._articulation,
            body_names_expression=self.sensor_dh.get_dependent_body_names(),
        )

    @property
    def env_id(self) -> int:
        return self._env_id

    def set_from_source(self):
        """Set dh data from the source environment"""
        self.art_dh.set_from_source(source=self._articulation)
        self.command_dh.set_from_source(source=self._env.command_manager)
        self.rigid_object_dh.set_from_source(source_dict=self._env.scene.rigid_objects)
        self.imu_dh.set_from_source(
            sensors=self._env.scene.sensors,
            articulation=self._articulation,
        )
        self.sensor_dh.set_from_source(source=self._env.scene.sensors)
        self.memory_dh.set_from_source()

    def set_to_source(self):
        """Set dh data to the source environment"""
        self.art_dh.set_to_source(self._articulation)
        self.command_dh.set_to_source(self._env.command_manager)
        self.rigid_object_dh.set_to_source(source_dict=self._env.scene.rigid_objects)
        self.imu_dh.set_to_source(
            sensors=self._env.scene.sensors,
            articulation=self._articulation,
        )
        self.sensor_dh.set_to_source(source=self._env.scene.sensors, articulations=self._env.scene.articulations)
        self.memory_dh.set_to_source()

    def set_command_data_to_source(self, command_data: dict[str, torch.Tensor]):
        """Set command data to the source environment.
        By setting the commands to source they can be visualized in the environment.

        Args:
            command_data (dict[str, torch.Tensor]): The command inputs to set.
        """
        self.command_dh.set_from_data(command_data)
        self.command_dh.set_to_source(self._env.command_manager)
