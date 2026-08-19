# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import abc
from collections.abc import Callable

import torch

from exploy.exporter.core.components import Connection, Memory
from exploy.exporter.core.context_manager import ContextManager


class ExportableActor(torch.nn.Module, abc.ABC):
    """Abstract interface for an actor that can be exported to ONNX."""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Given a batch of observations, compute the corresponding actions.

        Args:
            obs: A tensor of shape (batch_size, obs_dim) containing the observations."""
        raise NotImplementedError("forward() method must be implemented by subclasses.")

    def reset(self, dones: torch.Tensor):
        """Reset the actor's internal state (e.g., RNN hidden states) based on the done flags.

        Args:
            dones: A tensor of shape (batch_size,) containing boolean flags indicating which
                   environments have been reset.
        """
        pass

    def get_state(self) -> tuple[torch.Tensor, ...] | None:
        """Get the actor's internal state as a tuple of tensors, or None if there is no state."""
        return None


def make_exportable_actor(actor: torch.nn.Module) -> ExportableActor:
    """Convert a torch.nn.Module actor to an ExportableActor.

    Args:
        actor: The actor to convert.
    """

    class Actor(ExportableActor):
        def __init__(self, actor: torch.nn.Module):
            super().__init__()
            self._actor = actor

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self._actor(obs)

    return Actor(actor)


def add_actor_memory(
    context_manager: ContextManager,
    get_hidden_states_func: Callable[[], tuple[torch.Tensor, ...]],
):
    """Add inputs for actor hidden states.

    Args:
        context_manager: The context manager to add the inputs to.
        get_hidden_states_func: A function that returns a tuple of hidden state tensors, used to get the hidden states to add as inputs.
    """
    actor_state = get_hidden_states_func()
    if actor_state is None:
        return

    assert isinstance(actor_state, tuple), (
        f"Expected actor hidden states to be a tuple of tensors, got: {type(actor_state).__name__}"
    )

    for i_hs in range(len(actor_state)):

        def get_hidden_state(
            _i_hs: int = i_hs,
            _get_cb: Callable = get_hidden_states_func,
        ) -> torch.Tensor:
            return _get_cb()[_i_hs]

        def set_hidden_state(
            value: torch.Tensor,
            _i_hs: int = i_hs,
            _get_cb: Callable = get_hidden_states_func,
        ):
            _get_cb()[_i_hs][:] = value

        component_name = f"actor_hidden_state_{i_hs}"
        memory_comp = Memory(
            name=component_name,
            get_from_env_cb=get_hidden_state,
        )
        context_manager.add_component(memory_comp)
        context_manager.add_component(
            Connection(
                name=f"connection_{component_name}",
                getter=memory_comp.input.get_from_env_cb,
                setter=set_hidden_state,
            )
        )
