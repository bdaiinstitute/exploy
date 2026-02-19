# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from isaaclab.assets import Articulation
from isaaclab.managers import ObservationManager


def get_articulation_actuator_gains(articulation: Articulation) -> dict:
    """Get a dictionary of actuator gains."""
    gains = {}

    def _update_dict(gain_cfg: dict | float, gain_name: str):
        if isinstance(gain_cfg, (float | int)):
            _, joint_names = articulation.find_joints(actuator_cfg.joint_names_expr)
            for name in joint_names:
                if name not in gains:
                    gains[name] = {}
                gains[name][gain_name] = float(gain_cfg)
        elif isinstance(gain_cfg, dict):
            for expr, val in gain_cfg.items():
                _, joint_names = articulation.find_joints(expr)
                for name in joint_names:
                    if name not in gains:
                        gains[name] = {}
                    gains[name][gain_name] = float(val)

    for actuator_cfg in articulation.cfg.actuators.values():
        _update_dict(gain_cfg=actuator_cfg.stiffness, gain_name="stiffness")
        _update_dict(gain_cfg=actuator_cfg.damping, gain_name="damping")

    return gains


def get_observation_names(
    observation_manager: ObservationManager,
    group_name: str = "policy",
) -> list[str]:
    """Compute a list of observation names.

    Given an `ObservationManager` and an observation group name, create a list of names for each entry in
    the manager's observation group buffer.

    Args:
        observation_manager (ObservationManager): An environment's observation manager.
        group_name (str): The observation group for which we want to generate names.

    Returns:
        A list of names for each entry in the observation manager's group buffer.
    """
    term_names = observation_manager._group_obs_term_names[group_name]
    term_dims = observation_manager._group_obs_term_dim[group_name]

    names = []
    for i_name, name in enumerate(term_names):
        assert len(term_dims[i_name]) == 1, (
            "Only 1D observation terms are supported when generating names."
        )
        dim = term_dims[i_name][0]
        for i in range(dim):
            names.append(f"{name}_{i:02}")
    return names
