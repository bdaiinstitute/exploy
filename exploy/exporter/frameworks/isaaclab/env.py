# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Any

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import RayCaster

from exploy.exporter.frameworks.isaaclab.articulation_data import ArticulationDataSource
from exploy.exporter.frameworks.isaaclab.rigid_object_data import RigidObjectDataSource
from exploy.exporter.frameworks.manager_based.env import ExportableEnvironment
from exploy.exporter.frameworks.manager_based.raycaster_data import RayCasterDataSource
from exploy.exporter.frameworks.manager_based.utils import get_observation_names


class IsaacLabExportableEnvironment(ExportableEnvironment):
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        policy_obs_group_name: str = "policy",
    ):
        super().__init__(env=env, policy_obs_group_name=policy_obs_group_name)

        # Replace articulation data.
        self._art_data_list = []
        for articulation in self._env.scene.articulations.values():
            self._art_data_list.append(articulation._data)
            articulation._data = ArticulationDataSource(articulation=articulation)

        # Replace rigid object data.
        self._rigid_object_list = []
        for rigid_object in self._env.scene.rigid_objects.values():
            self._rigid_object_list.append(rigid_object._data)
            rigid_object._data = RigidObjectDataSource(rigid_object)

        # Replace raycaster sensor data.
        self._raycaster_data_list = []
        for sensor_name, sensor in self._env.scene.sensors.items():
            if type(sensor) is RayCaster:
                self._raycaster_data_list.append((sensor_name, sensor._data))
                sensor._data = RayCasterDataSource(sensor, self._env.scene.articulations)

        assert self.validate()

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        try:
            for i_articulation, articulation in enumerate(self._env.scene.articulations.values()):
                articulation._data = self._art_data_list[i_articulation]

            for i_rigid_object, rigid_object in enumerate(self._env.scene.rigid_objects.values()):
                rigid_object._data = self._rigid_object_list[i_rigid_object]

            for sensor_name, original_data in self._raycaster_data_list:
                self._env.scene.sensors[sensor_name]._data = original_data
        except AttributeError:
            # In case the environment is already cleaned up by IsaacLab, we might get attribute errors
            # when trying to restore the original data sources. We can ignore these errors since the
            # environment is being cleaned up anyway.
            return

    def _get_robots_dict(self) -> dict[str, Any]:
        return self._env.scene.articulations

    def _get_physics_dt(self) -> float:
        return self._env.cfg.sim.dt

    def _get_scene_cfg_attr(self) -> str:
        return "cfg"

    def _compute_observations_kwargs(self) -> dict[str, Any]:
        return {}

    def get_observation_names(self) -> list[str]:
        return get_observation_names(
            observation_manager=self._env.observation_manager,
            group_name=self._policy_obs_group_name,
        )
