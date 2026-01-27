# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from dataclasses import MISSING
from pathlib import Path

from exporter import DataHandlerManager, ExportEnvIdsType, SessionWrapper
from exporter.exporter import get_onnx_inputs

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


@configclass
class OnnxPolicyRunnerCfg:
    """Configuration for the ONNX policy runner."""

    onnx_path: str = MISSING
    """The path to the ONNX policy file."""

    env_id: ExportEnvIdsType = MISSING
    """The environment IDs to run the policy on."""

    robot_name: str = "robot"
    """The name of the robot articulation in the environment."""

    policy_group_name: str = "policy"
    """The name of the policy group in the environment."""


class OnnxPolicyRunner:

    _onnx_memory: dict[str, np.array]

    def __init__(self, cfg: OnnxPolicyRunnerCfg, env: ManagerBasedRLEnv):
        """Initialize the ONNX policy runner.

        Args:
            cfg (OnnxPolicyRunnerCfg): The configuration for the ONNX policy runner.
            env (ManagerBasedRLEnv): The environment to run the policy on.
        """
        self._cfg = cfg
        self._env = env
        self._device = self._env.device

        onnx_path = Path(cfg.onnx_path)
        self.session_wrapper = SessionWrapper(
            onnx_folder=onnx_path.parent,
            onnx_file_name=onnx_path.name,
            policy=None,
            optimize=False,
        )

        self.dh_manager = DataHandlerManager(
            env=env,
            env_id=cfg.env_id,
            robot_name=cfg.robot_name,
            policy_group_name=cfg.policy_group_name,
            device=self._device,
        )

        # Initialize ONNX inputs and outputs.
        self._onnx_inputs: dict[str, np.array] = self._get_onnx_inputs_from_dh()
        self._onnx_outputs: dict[str, np.array] = self._policy_inference(self._onnx_inputs)

        # Reset the session wrapper
        self.session_wrapper.reset()

    def __call__(
        self,
        commands: dict[str, np.ndarray] | None = None,
        memory: dict[str, np.ndarray] | None = None,
        to_torch: bool = True,
    ) -> dict[str, np.ndarray] | dict[str, torch.Tensor]:
        """Run one step of the policy and return the ONNX outputs.

        Args:
            commands (dict[str, np.ndarray] | None): Optional command inputs. By default, uses environment commands.
            memory (dict[str, np.ndarray] | None): Optional memory inputs. By default, uses environment memory.
            to_torch (bool): Whether to return the outputs as PyTorch tensors or NumPy arrays.

        Returns:
            dict[str, np.ndarray] | dict[str, torch.Tensor]: The ONNX outputs from the policy.
        """

        self._onnx_inputs = self._get_onnx_inputs(commands=commands, memory=memory)

        self._onnx_outputs = self._policy_inference(self._onnx_inputs)

        return self.read_onnx_outputs(to_torch=to_torch)

    def read_onnx_inputs(self, to_torch: bool = False) -> dict[str, np.ndarray] | dict[str, torch.Tensor]:
        """Read the current ONNX inputs."""
        return self._prepare_onnx_io(self._onnx_inputs, clone=True, to_torch=to_torch)

    def read_onnx_outputs(self, to_torch: bool = False) -> dict[str, np.ndarray] | dict[str, torch.Tensor]:
        """Read the current ONNX outputs."""
        return self._prepare_onnx_io(self._onnx_outputs, clone=True, to_torch=to_torch)

    def _prepare_onnx_io(
        self, src: dict[str, np.ndarray], clone: bool = True, to_torch: bool = False
    ) -> dict[str, np.ndarray] | dict[str, torch.Tensor]:
        """Prepare the ONNX inputs and outputs by cloning them to avoid memory issues.

        Args:
            clone (bool): Whether to clone the inputs and outputs.
            to_torch (bool): Whether to convert the inputs and outputs to PyTorch tensors.
        """
        if clone:
            src = {name: value.copy() for name, value in src.items()}

        if to_torch:
            return {name: torch.from_numpy(value).to(self._device) for name, value in src.items()}
        else:
            return src

    def _get_onnx_inputs(
        self, commands: dict[str, np.ndarray] | None = None, memory: dict[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        """Compute the ONNX inputs from the data handler manager.

        Args:
            commands (dict[str, np.ndarray] | None): The command inputs.
            memory (dict[str, np.ndarray] | None): The memory inputs.

        Returns:
            dict[str, np.ndarray]: The ONNX inputs as a dictionary of numpy arrays.
        """
        if commands is not None:
            command_data_torch = {name: torch.from_numpy(value).to(self._device) for name, value in commands.items()}
            self.dh_manager.set_command_data_to_source(command_data_torch)

        self.dh_manager.set_from_source()

        onnx_inputs = self._get_onnx_inputs_from_dh()

        # Update memory in ONNX inputs
        self._onnx_memory = memory if memory is not None else self._get_memory_from_session_wrapper()
        if not (self._onnx_memory.keys() <= onnx_inputs.keys()):
            raise ValueError("Memory keys do not match ONNX input keys.")
        for key, value in self._onnx_memory.items():
            onnx_inputs[key] = value

        return onnx_inputs

    def _get_onnx_inputs_from_dh(self) -> dict[str, np.ndarray]:
        """Get the ONNX inputs from the data handler manager.

        Returns:
            dict[str, np.ndarray]: The ONNX inputs as a dictionary of numpy arrays.
        """
        onnx_inputs: dict[str, np.ndarray] = get_onnx_inputs(self.dh_manager, to_numpy=True)

        # Remove unused inputs
        onnx_inputs = {name: value for name, value in onnx_inputs.items() if name in self.session_wrapper.input_names}

        return onnx_inputs

    def _get_memory_from_session_wrapper(self):
        """Get the ONNX memory inputs from the session wrapper."""
        onnx_memory = {}
        for name in self.dh_manager.memory_dh.input_names():
            onnx_memory[name] = self.session_wrapper.get_output_value(
                self.dh_manager.memory_dh.io_name_to_output_name(name)
            )
        return onnx_memory

    def _policy_inference(self, onnx_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Evaluate the policy with given ONNX inputs.

        Args:
            onnx_inputs (dict[str, np.ndarray]): The ONNX inputs to evaluate the policy with.

        Returns:
            dict[str, np.ndarray]: The ONNX outputs from the policy.
        """
        values = self.session_wrapper(**onnx_inputs)
        return {name: value for name, value in zip(self.session_wrapper.output_names, values)}
