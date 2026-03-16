# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import gymnasium as gym
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config import g1
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class G1FlatRNNPPORunnerCfg(G1FlatPPORunnerCfg):
    """A runner configuration that uses a recurrent policy and critic for the G1 flat environment.
    This is used for testing the export of recurrent policies.
    """

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat_rnn"

        self.policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=0.0,
            noise_std_type="log",
            actor_hidden_dims=[256, 128],
            critic_hidden_dims=[256, 128],
            activation="elu",
            rnn_type="lstm",
            rnn_hidden_dim=512,
            rnn_num_layers=2,
        )


gym.register(
    id="Exploy-Velocity-Flat-G1-RNN-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{g1.__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}:G1FlatRNNPPORunnerCfg",
    },
)
