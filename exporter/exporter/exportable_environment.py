# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import abc
from collections.abc import Callable

import torch

from exporter.context_manager import ContextManager


class ExportableEnvironment(abc.ABC):
    def __init__(self):
        self._context_manager = ContextManager()

    def context_manager(self) -> ContextManager:
        return self._context_manager

    @abc.abstractmethod
    def compute_observations(self, device: str) -> torch.Tensor:
        """Compute and return the observations of the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def process_actions(self, actions: torch.Tensor):
        """Process actions."""
        pass

    @abc.abstractmethod
    def apply_actions(self):
        """Apply processed actions (e.g., joint targets) to the environment"""
        pass

    @abc.abstractmethod
    def empty_actor_observations(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def empty_actions(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def metadata(self) -> dict[str, str]:
        """Return metadata about the environment required for export."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def decimation(self) -> int:
        """Return metadata about the environment required for export."""
        raise NotImplementedError

    @abc.abstractmethod
    def register_evaluation_hooks(
        self,
        update: Callable[[], None],
        reset: Callable[[], None],
        evaluate_substep: Callable[[int], None],
    ):
        """Register evaluation hooks for this environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Step the environment forward by one step. Returns the next observations and a boolean indicating if the environment was reset."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_observation_names(self) -> list[str]:
        """Get the names of the observations in the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def observations_reset(self) -> torch.Tensor:
        """Get the observations after an environment reset."""
        raise NotImplementedError
