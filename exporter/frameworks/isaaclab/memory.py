import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm

from exporter.core import ContextManager, Memory


def add_memory(
    env: ManagerBasedRLEnv,
    context_manager: ContextManager,
):
    """Parse the managers of an environment and keep track of all elements that require memory.

    This functions parses a the managers of a `ManagerBasedRLEnv` and keeps track of all elements
    that require memory handling.

    For example, we frequently pass the latest actions back as previous action inputs to a
    trained policy.
    """

    # Keep track of previous actions.
    def getter_func() -> torch.Tensor:
        return env.action_manager._action

    context_manager.add_component(
        component=Memory(
            name="actions",
            get_from_env_cb=getter_func,
        ),
    )

    # We need to create these functions in a closure to capture the current active_term and env_id
    def make_processed_actions_funcs(active_term: ActionTerm):
        def getter_func() -> torch.Tensor:
            return active_term.processed_actions

        return getter_func

    for action_term_name in env.action_manager.active_terms:
        active_term = env.action_manager.get_term(action_term_name)

        getter = make_processed_actions_funcs(active_term)

        context_manager.add_component(
            component=Memory(
                name=f"{action_term_name}.processed_actions",
                get_from_env_cb=getter,
            )
        )
