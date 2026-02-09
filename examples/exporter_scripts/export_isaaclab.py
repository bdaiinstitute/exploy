# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse
import pathlib
from typing import TYPE_CHECKING

import gymnasium as gym
from isaaclab.app import AppLauncher

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.sim import SimulationApp
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def make_simulation_app() -> SimulationApp:
    # Create argument parser for headless mode
    parser = argparse.ArgumentParser()

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args([])
    args_cli.headless = True
    args_cli.num_envs = 1

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    return simulation_app


simulation_app = make_simulation_app()


# Import tasks to register environments
import isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c  # noqa: F401
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1  # noqa: F401
import torch
from exporter_frameworks.isaaclab.env import IsaacLabExportableEnvironment
from isaaclab.sim import SimulationContext
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import exporter


def main(task_name: str = None):
    # test_dir = pathlib.Path(tempfile.gettempdir()) / "exporter_tests"
    test_dir = pathlib.Path(__file__).parent / "exporter_tests"

    task_device = "cpu"

    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task_name, num_envs=1, device=task_device)
    agent_cfg: RslRlOnPolicyRunnerCfg = isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry(
        task_name, "rsl_rl_cfg_entry_point"
    )

    # create isaac environment
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(gym.make(task_name, cfg=env_cfg, render_mode=None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=test_dir, device=agent_cfg.device)

    # Get the policy and its normalizer.
    policy = runner.alg.policy.actor.to(env.device)
    normalizer = runner.alg.policy.actor_obs_normalizer.to(env.device)

    # Export to ONNX.
    onnx_export_dir = test_dir
    onnx_export_file = "test_export.onnx"
    env_id = 0
    export_device = "cpu"

    exportable_env = IsaacLabExportableEnvironment(env.unwrapped, env_id=env_id)

    exporter.export_environment_as_onnx(
        env=exportable_env,
        actor=policy,
        normalizer=normalizer,
        path=onnx_export_dir,
        filename=onnx_export_file,
        verbose=False,
        export_device=export_device,
    )

    exportable_env.cleanup()

    # Make a session wrapper.
    session_wrapper = exporter.SessionWrapper(
        onnx_folder=onnx_export_dir,
        onnx_file_name=onnx_export_file,
        policy=policy,
        optimize=True,
    )

    # Evaluate.
    evaluate_steps = 200
    with torch.inference_mode():
        exporter.evaluate(
            env=exportable_env,
            context_manager=exportable_env.context_manager(),
            session_wrapper=session_wrapper,
            num_steps=evaluate_steps,
            verbose=True,
        )

    # Close simulation app.
    if simulation_app:
        print("Closing simulation app...")
        SimulationContext.clear_instance()
        simulation_app.close()

    print("Done.")


if __name__ == "__main__":
    # main(task_name="Isaac-Velocity-Flat-Anymal-C-Play-v0")
    # main(task_name="Isaac-Velocity-Rough-Anymal-C-Play-v0")
    main(task_name="Isaac-Velocity-Rough-G1-Play-v0")
