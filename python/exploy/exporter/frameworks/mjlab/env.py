# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import json
from collections.abc import Callable

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import RayCastSensor

from exploy.exporter.core.exportable_environment import ExportableEnvironment
from exploy.exporter.frameworks.mjlab.entity_data import EntityDataSource
from exploy.exporter.frameworks.mjlab.raycaster_data import RayCasterDataSource
from exploy.exporter.frameworks.mjlab.utils import get_observation_names


class _ONNXObservationCompute:
    """Drop-in replacement for ``ObservationManager.compute`` that fires ``update`` first.

    MjLab computes observations at the very end of ``ManagerBasedRlEnv.step`` (after
    ``sim.forward()``, event application, and ``sim.sense()``). The network outputs must
    be written into the environment immediately before that computation so that the
    observations reflect the latest state. This wrapper calls ``update`` and then
    delegates to the original ``compute``, preserving its signature and return value.

    A dedicated class (rather than a closure) gives a stable type identity so callers can
    detect an already-installed wrapper via ``isinstance`` and avoid double-wrapping.
    """

    def __init__(self, wrapped_compute: Callable[..., object], update: Callable[[], None]):
        self.wrapped_compute = wrapped_compute
        self._update = update

    def __call__(self, *args, **kwargs):
        self._update()
        return self.wrapped_compute(*args, **kwargs)


class MjlabExportableEnvironment(ExportableEnvironment):
    """Wraps a MjLab ``ManagerBasedRlEnv`` for ONNX export.

    Replaces each entity's ``_data`` with an ONNX-traceable ``EntityDataSource``
    proxy so that all observation computations use managed tensors that become
    ONNX graph inputs.  Only entity data is replaced — scene sensors and other
    state are left intact.

    After export, ``cleanup()`` restores the original entity data objects.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        policy_obs_group_name: str = "actor",
    ):
        super().__init__()
        self._env = env
        self._policy_obs_group_name = policy_obs_group_name

        # Replace entity data with ONNX-traceable proxies.
        self._entity_data_list: list = []
        for entity in self._env.scene.entities.values():
            self._entity_data_list.append(entity._data)
            entity._data = EntityDataSource(entity=entity)

        # Replace raycaster sensor data with ONNX-traceable proxies.
        self._raycaster_data_list: list[tuple] = []
        for sensor_name, sensor in self._env.scene.sensors.items():
            if isinstance(sensor, RayCastSensor):
                self._raycaster_data_list.append((sensor_name, sensor._data))
                sensor._data = RayCasterDataSource(sensor, self._env.scene.entities)

        assert self.validate()

    def __del__(self):
        self.cleanup()

    # ── ExportableEnvironment interface ───────────────────────────────────────

    @property
    def env(self) -> ManagerBasedRlEnv:
        return self._env

    @property
    def decimation(self) -> int:
        return self._env.cfg.decimation

    def prepare_export(self) -> None:
        self._env.command_manager.compute(dt=0.0)

    def cleanup(self) -> None:
        """Restore original entity data, raycaster data, and evaluation hooks."""
        for i, entity in enumerate(self._env.scene.entities.values()):
            if i < len(self._entity_data_list):
                entity._data = self._entity_data_list[i]
        for sensor_name, original_data in self._raycaster_data_list:
            self._env.scene.sensors[sensor_name]._data = original_data

        obs_manager = self._env.observation_manager
        if isinstance(obs_manager.compute, _ONNXObservationCompute):
            obs_manager.compute = obs_manager.compute.wrapped_compute
        self._env.scene._sensors.pop("onnx", None)

    def compute_observations(self) -> torch.Tensor:
        obs_dict = self._env.observation_manager.compute(update_history=True)
        return obs_dict[self._policy_obs_group_name].view(1, -1)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._env.action_manager.process_action(actions)

    def apply_actions(self) -> None:
        self._env.action_manager.apply_action()

    def empty_actor_observations(self) -> torch.Tensor:
        return torch.zeros_like(
            self._env.observation_manager._obs_buffer[self._policy_obs_group_name]
        )

    def empty_actions(self) -> torch.Tensor:
        return torch.zeros_like(self._env.action_manager._action)

    def validate(self) -> bool:
        obs_manager = self._env.observation_manager

        if self._policy_obs_group_name not in obs_manager.active_terms:
            raise RuntimeError(
                f"[MjlabExportableEnvironment] Could not find observation group "
                f"'{self._policy_obs_group_name}' in observation manager. "
                f"Active terms are: {list(obs_manager.active_terms.keys())}"
            )

        if not obs_manager.group_obs_concatenate[self._policy_obs_group_name]:
            raise RuntimeError(
                "[MjlabExportableEnvironment] Observation terms must be concatenated."
            )

        for group_cfg in obs_manager._group_obs_term_cfgs.values():
            for term_cfg in group_cfg:
                if term_cfg.noise is not None:
                    raise RuntimeError(
                        "[MjlabExportableEnvironment] Found an observation term with noise "
                        "enabled. Turn off observation noise or use a play=True task config."
                    )

        return True

    def metadata(self) -> dict[str, str]:
        metadata: dict[str, str] = {}
        metadata["obs_term_names"] = json.dumps(
            self._env.observation_manager._group_obs_term_names[self._policy_obs_group_name]
        )
        metadata["decimation"] = str(self._env.cfg.decimation)
        metadata["sim_dt"] = str(self._env.physics_dt)
        metadata["update_rate"] = str(1.0 / self._env.physics_dt)
        base_names = {}
        for entity_name, entity in self._env.scene.entities.items():
            if entity.is_articulated:
                root_body = entity.root_body
                root_body_name = root_body.name.split("/")[-1]
                base_names[entity_name] = root_body_name
        metadata["base_names"] = json.dumps(base_names)
        return metadata

    def get_observation_names(self) -> list[str]:
        return get_observation_names(
            observation_manager=self._env.observation_manager,
            group_name=self._policy_obs_group_name,
        )

    def observations_reset(self) -> torch.Tensor:
        with torch.inference_mode():
            obs_buf, _ = self._env.reset()
            return obs_buf[self._policy_obs_group_name].clone()

    def register_evaluation_hooks(
        self,
        update: Callable[[], None],
        evaluate_substep: Callable[[int], None],
    ) -> None:
        """Register evaluation callbacks into MjLab's step loop.

        Two hook points are used:

        * ``evaluate_substep`` / ``update`` per physics substep: MjLab calls
          ``scene.update`` inside its decimation loop, which invokes ``sensor.update``
          on every registered sensor. We inject a dummy ``_ONNXSensor`` so our substep
          callbacks fire at each substep.
        * ``update`` before observation computation: observations are built at the end of
          ``step`` via ``observation_manager.compute``. We wrap that method with
          ``_ONNXObservationCompute`` so the latest network outputs are written just
          before observations are read. The ``isinstance`` guard keeps the wrap
          idempotent across repeated registrations.
        """
        # Disable lazy sensor update so sensors are called at every substep.
        if hasattr(self._env.scene, "_cfg"):
            self._env.scene._cfg.lazy_sensor_update = False

        class _ONNXSensor:
            def __init__(self):
                self.sub_step_ctr = 0

            def update(self, *args, **kwargs):
                evaluate_substep(self.sub_step_ctr)
                self.sub_step_ctr += 1
                update()

            def reset(self, *args, **kwargs):
                pass

        self._env.scene._sensors["onnx"] = _ONNXSensor()

        obs_manager = self._env.observation_manager
        if not isinstance(obs_manager.compute, _ONNXObservationCompute):
            obs_manager.compute = _ONNXObservationCompute(obs_manager.compute, update)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Step the environment forward, delegating to mjlab's own step loop."""
        self._env.scene._sensors["onnx"].sub_step_ctr = 0
        obs, _, dones, timeouts, _ = self._env.step(actions)
        is_reset: bool = torch.logical_or(dones, timeouts).any()
        return obs[self._policy_obs_group_name], is_reset
