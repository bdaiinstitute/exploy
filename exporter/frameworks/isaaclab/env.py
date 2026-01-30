import json
from collections.abc import Callable, Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from exporter.core.articulation_data import ArticulationDataSource
from exporter.core.exportable_environment import ExportableEnvironment
from exporter.core.rigid_object_data import RigidObjectDataSource
from exporter.frameworks.isaaclab import inputs, memory, outputs
from exporter.frameworks.isaaclab.utils import get_observation_names


class IsaacLabExportableEnvironment(ExportableEnvironment):
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        env_id: int,
    ):
        assert type(env) is ManagerBasedRLEnv, (
            "IsaacLabExportableEnvironment only supports ManagerBasedRLEnv environments."
        )

        super().__init__()
        self._env = env
        self._env_id: int = env_id

        # The group name of the policy observations in the environment's observation manager.
        self._policy_obs_group_name = "policy"

        # The name of the articulation in the environment's scene.
        self._articulation_name = "robot"

        self._empty_actor_observations = self._env.obs_buf[self._policy_obs_group_name].clone()
        self._empty_actions = self._env.action_manager._action.clone()

        # Replace articulation data.
        self._orig_art_data = self._env.scene.articulations[self._articulation_name]._data
        self._env.scene.articulations[self._articulation_name]._data = ArticulationDataSource(
            articulation=self._env.scene.articulations[self._articulation_name]
        )

        # Replace rigid object data.
        self._rigid_object_list = []
        for rigid_object in self._env.scene.rigid_objects.values():
            self._rigid_object_list.append(rigid_object._data)
            rigid_object._data = RigidObjectDataSource(rigid_object)

        assert self.validate()

        # Add inputs, outputs, and memory to manager.
        inputs.add_commands(
            source=self._env.command_manager,
            context_manager=self._context_manager,
        )
        inputs.add_articulation_data(
            articulation=self._env.scene[self._articulation_name],
            context_manager=self._context_manager,
        )
        inputs.add_sensor_inputs(
            articulation=self._env.scene[self._articulation_name],
            sensors=self._env.scene.sensors,
            context_manager=self._context_manager,
        )
        memory.add_memory(
            env=self._env,
            context_manager=self._context_manager,
        )
        outputs.add_outputs(
            action_manager=self._env.action_manager,
            articulation=self._env.scene[self._articulation_name],
            context_manager=self._context_manager,
        )

    @property
    def env(self) -> ManagerBasedRLEnv:
        return self._env

    @property
    def decimation(self) -> int:
        return self._env.cfg.decimation

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        self._env.scene.articulations[self._articulation_name]._data = self._orig_art_data

        for i_rigid_object, rigid_object in enumerate(self._env.scene.rigid_objects.values()):
            rigid_object._data = self._rigid_object_list[i_rigid_object]

    def compute_observations(self, device: str) -> torch.Tensor:
        """Compute and return the observations of the environment."""
        obs_dict = self._env.observation_manager.compute()
        observations = (
            obs_dict[self._policy_obs_group_name]
            .to(device=device)
            .view(
                1,
                self._env.observation_manager.group_obs_dim[self._policy_obs_group_name][
                    self._env_id
                ],
            )
        )
        return observations

    def process_actions(self, actions: torch.Tensor):
        """Process actions."""
        self._env.action_manager.process_action(actions)

    def apply_actions(self):
        """Apply processed actions (e.g., joint targets) to the environment"""
        self._env.action_manager.apply_action()

    def validate(self) -> bool:
        """Validate that the environment conforms to the ExportableEnvironment interface."""

        # Check that the expected articulation exists.
        assert self._articulation_name in self._env.scene.articulations

        # Check that the observation manager has a policy group.
        if self._policy_obs_group_name not in self._env.observation_manager.active_terms:
            raise RuntimeError(
                "[OnnxEnvironmentExporter] Could not find observation group named {self._policy_obs_group_name} in"
                " observation manager. Active terms are: {self._env.observation_manager.active_terms}"
            )

        # Check if observation terms are concatenated.
        if not self._env.observation_manager.group_obs_concatenate[self._policy_obs_group_name]:
            raise RuntimeError("[OnnxEnvironmentExporter] Observation terms must be concatenated.")

        # Check that observation noise is disabled, else it will be part of the computational graph.
        for group_cfg in self._env.observation_manager._group_obs_term_cfgs.values():
            for term_cfg in group_cfg:
                if term_cfg.noise is not None:
                    raise RuntimeError(
                        "[OnnxEnvironmentExporter] While trying to convert to ONNX, found an observation term with"
                        " noise enabled."
                        "\n[OnnxEnvironmentExporter] Hint: turn off observation noise, or use a `Play` task."
                    )

        return True

    def empty_actor_observations(self) -> torch.Tensor:
        return torch.zeros_like(
            self._env.observation_manager._obs_buffer[self._policy_obs_group_name]
        )

    def empty_actions(self) -> torch.Tensor:
        return torch.zeros_like(self._env.action_manager._action)

    def metadata(self) -> dict[str, str]:
        metadata = {}

        # Joint names in order expected by ONNX file.
        metadata["joint_names"] = ",".join(
            self._env.scene[self._articulation_name].data.joint_names
        )

        # Body names.
        metadata["body_names"] = ",".join(self._env.scene[self._articulation_name].data.body_names)

        # Command info.
        # metadata["commands"] = json.dumps(self._data_handler.command_dh.metadata)

        # # Sensor info.
        # metadata["sensors"] = json.dumps(self._data_handler.sensor_dh.metadata)

        # # Joint target info.
        # metadata["outputs"] = json.dumps(self._data_handler.action_dh.metadata())

        # Observation names.
        metadata["obs_term_names"] = json.dumps(
            self._env.observation_manager._group_obs_term_names[self._policy_obs_group_name]
        )

        # Policy update period in seconds.
        metadata["policy_dt"] = str(self._env.cfg.sim.dt * self._env.cfg.decimation)

        # Decimation info.
        metadata["decimation"] = str(self._env.cfg.decimation)

        # Sim_dt info.
        metadata["sim_dt"] = str(self._env.cfg.sim.dt)

        return metadata

    def register_evaluation_hooks(
        self,
        update: Callable[[], None],
        reset: Callable[[], None],
        evaluate_substep: Callable[[int], None],
    ):
        """Register evaluation hooks for this environment."""

        self._env.scene.cfg.lazy_sensor_update = False

        class ONNXEvaluatorSensor:
            def __init__(self):
                self.sub_step_ctr = 0

            def update(self, dt: float, force_recompute: bool):
                evaluate_substep(self.sub_step_ctr)
                self.sub_step_ctr += 1
                update()

            def reset(self, env_ids: Sequence[int]):
                reset()

        self._env.scene._sensors["onnx"] = ONNXEvaluatorSensor()

        class ONNXEvaluatorCommand:
            def compute(self, dt: float):
                update()

            def reset(self, env_ids: Sequence[int]) -> dict[str, float]:
                reset()
                return {}

        self._env.command_manager._terms["onnx"] = ONNXEvaluatorCommand()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Step the environment forward by one step."""
        self._env.scene._sensors["onnx"].sub_step_ctr = 0
        obs, _, dones, timeouts, _ = self._env.step(actions)
        is_reset: bool = torch.logical_or(dones[self._env_id], timeouts[self._env_id]).any()
        obs = obs[self._policy_obs_group_name]
        return obs, is_reset

    def get_observation_names(self) -> list[str]:
        return get_observation_names(
            observation_manager=self._env.observation_manager,
            group_name=self._policy_obs_group_name,
        )

    def observations_reset(self) -> torch.Tensor:
        with torch.inference_mode():
            obs_buf, _ = self._env.reset()
            return obs_buf[self._policy_obs_group_name].clone()
