# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch
from collections.abc import Callable

from exporter import ExportEnvIdsType

from isaaclab.envs import ManagerBasedRLEnv


class MemoryHandler:
    """Handle memory inputs and outputs.

    This class abstracts how to get and set values used in an environment
    that have memory, for example actions and previous actions. Values are retrieved
    by passing callables.
    """

    def __init__(self):
        self._memory_info = {}
        self._data = {}

    def has_memory(self) -> bool:
        """Return true if any memory element was added to this class."""
        return len(self._memory_info.keys()) > 0

    def add(
        self,
        name: str,
        getter_func: Callable,
        setter_func: Callable,
        play_getter_func: Callable = None,
    ):
        """Handle a memory value.

        Args:
            name: A string that represents this value.
            getter_func: A callable that returns the value.
        """
        assert name not in self._memory_info.keys()
        self._memory_info[name] = {
            "getter": getter_func,
            "setter": setter_func,
            "play_getter": play_getter_func if play_getter_func is not None else getter_func,
        }
        self._data[name] = getter_func().clone()

    def _names(self):
        """A list of names for each managed value."""
        return list(self._memory_info.keys())

    def input_names(self):
        """A list of names, formatted for use with the ONNX exporter inputs."""
        return [f"memory.{name}.in" for name in self._names()]

    def output_names(self):
        """A list of names used for the ONNX exporter outputs."""
        return [f"memory.{name}.out" for name in self._names()]

    def io_name_to_name(self, io_name: str) -> str:
        """Helper function to convert a name formatted for inputs or outputs to a memory element name."""
        return io_name.removeprefix("memory.").removesuffix(".in").removesuffix(".out")

    def io_name_to_output_name(self, io_name: str) -> str:
        """Helper function to convert a name formatted for inputs or outputs to the corresponding outputs to a memory element name."""
        return io_name.removesuffix(".in") + ".out"

    def set_to_source(self):
        """Given a dictionary of values, set them to their source."""
        for key, val in self._data.items():
            assert key in self._memory_info
            self._memory_info[key]["setter"](val)

    def get_updated_values_as_outputs(self) -> list[torch.Tensor]:
        """A list of updated values."""
        return [self._memory_info[key]["getter"]().clone() for key in self._memory_info.keys()]

    def set_from_source(self):
        for key, val in self._memory_info.items():
            self._data[key] = val["play_getter"]().clone()

    @property
    def data(self):
        """A dictionary holding data from a source data container."""
        return {f"memory.{key}.in": val for key, val in self._data.items()}

    @property
    def data_numpy(self):
        """Get all values from getter functions and return them as numpy arrays.

        This function returns a dictionary of strings to values, to be used as input to an ONNX session wrapper.
        The dictionary names are memory inputs names, formatted as 'memory.name', where 'name' is the memory element's
        name.
        The dictionary values are evaluated from the 'play_getter' functions stored for each memory input and converted
        to numpy arrays.
        """
        return {key: val.cpu().numpy() for key, val in self.data.items()}


def get_memory_handler_from_env(
    env: ManagerBasedRLEnv,
    env_id: ExportEnvIdsType,
    device: str,
) -> MemoryHandler:
    """Parse the managers of an environment and keep track of all elements that require memory.

    This functions parses a the managers of a `ManagerBasedRLEnv` and keeps track of all elements
    that require memory handling.

    For example, we frequently pass the latest actions back as previous action inputs to a
    trained policy.
    """
    memory_handler = MemoryHandler()

    # Keep track of previous actions.
    def setter_action_func(val: torch.Tensor) -> torch.Tensor:
        env.action_manager._action[env_id] = val

    def getter_func() -> torch.Tensor:
        return env.action_manager._action[env_id].to(device)

    memory_handler.add(
        name="actions",
        getter_func=getter_func,
        play_getter_func=getter_func,
        setter_func=setter_action_func,
    )

    for action_term_name in env.action_manager.active_terms:
        active_term = env.action_manager.get_term(action_term_name)
        if type(active_term).__name__ == "TaskSpaceMLPAction":
            # Get current low_level_actions to infer shape
            current_actions = active_term.get_low_level_policy_action()  # list of M tensors, each [env_num, dim]

            action_dims = [action.shape[1] for action in current_actions]

            # Define setter function for the low-level action
            def setter_low_level_action_func(val: torch.Tensor):
                active_term.set_low_level_policy_previous_action(list(torch.split(val, action_dims, dim=1)))

            # Getter for the current low-level action
            def get_low_level_action() -> torch.Tensor:
                low_level_actions = active_term.get_low_level_policy_action()
                return torch.cat(low_level_actions, dim=-1).to(device)

            # Add low-level action to memory handler
            memory_handler.add(
                name="actions_low_level",
                getter_func=get_low_level_action,
                play_getter_func=get_low_level_action,
                setter_func=setter_low_level_action_func,
            )
        elif type(active_term).__name__ == "FeedForwardJointExcitationAction":
            # Keep track of sim_steps
            def setter_step_func(val: torch.Tensor):
                active_term.set_step(val)

            def getter_step_func() -> torch.Tensor:
                return active_term.get_step().to(device)

            memory_handler.add(
                name="step_counter",
                getter_func=getter_step_func,
                play_getter_func=getter_step_func,
                setter_func=setter_step_func,
            )
        elif type(active_term).__name__ == "LowLevelLocomotionAction":

            def get_skill_last_action():
                return active_term._skill_last_action[env_id].to(device)

            def set_skill_last_action(val):
                active_term._skill_last_action[env_id] = val

            memory_handler.add(
                name=f"{action_term_name}.skill",
                getter_func=get_skill_last_action,
                play_getter_func=get_skill_last_action,
                setter_func=set_skill_last_action,
            )

            def get_locomotion_last_action():
                return active_term.locomotion_last_action[env_id].to(device)

            def set_locomotion_last_action(val):
                active_term.locomotion_last_action[env_id] = val

            memory_handler.add(
                name=f"{action_term_name}.locomotion",
                getter_func=get_locomotion_last_action,
                play_getter_func=get_locomotion_last_action,
                setter_func=set_locomotion_last_action,
            )
    # Add more managers here to support memory for observation managers.

    return memory_handler
