# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
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


def make_simulation_app() -> tuple[SimulationApp, argparse.Namespace]:
    # Create argument parser for headless mode
    parser = argparse.ArgumentParser(description="Export Isaac Lab environment to ONNX")

    # Add custom arguments
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Velocity-Rough-G1-Play-v0",
        help="Name of the Isaac Lab task to export (default: Isaac-Velocity-Rough-G1-Play-v0)",
    )
    parser.add_argument(
        "--pause-on-failure",
        action="store_true",
        default=False,
        help="Pause on evaluation failure (default: False, useful for debugging)",
    )

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True
    args_cli.num_envs = 1

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    return simulation_app, args_cli


simulation_app, args = make_simulation_app()


# Import tasks to register environments
import isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c  # noqa: F401
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1  # noqa: F401
import torch
from isaaclab.sim import SimulationContext
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

from exploy.exporter.core.evaluator import evaluate
from exploy.exporter.core.exporter import export_environment_as_onnx
from exploy.exporter.core.session_wrapper import SessionWrapper
from exploy.exporter.frameworks.isaaclab.env import IsaacLabExportableEnvironment


def export_isaaclab(
    task_name: str = "Isaac-Velocity-Rough-G1-Play-v0", pause_on_failure: bool = False
):
    """Test Isaac Lab ONNX export and evaluation pipeline."""
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

    exportable_env = IsaacLabExportableEnvironment(env.unwrapped, env_id=env_id)

    export_environment_as_onnx(
        env=exportable_env,
        actor=policy,
        normalizer=normalizer,
        path=onnx_export_dir,
        filename=onnx_export_file,
        verbose=False,
    )

    exportable_env.cleanup()

    # Make a session wrapper.
    session_wrapper = SessionWrapper(
        onnx_folder=onnx_export_dir,
        onnx_file_name=onnx_export_file,
        policy=policy,
        optimize=True,
    )

    # Evaluate.
    evaluate_steps = 200
    with torch.inference_mode():
        export_ok, _ = evaluate(
            env=exportable_env,
            context_manager=exportable_env.context_manager(),
            session_wrapper=session_wrapper,
            num_steps=evaluate_steps,
            verbose=True,
            pause_on_failure=pause_on_failure,
        )

    # Close simulation app.
    if simulation_app:
        print("Closing simulation app...")
        SimulationContext.clear_instance()
        simulation_app.close()

    assert export_ok, "ONNX export validation failed"


if __name__ == "__main__":
    import sys

    try:
        export_isaaclab(task_name=args.task, pause_on_failure=args.pause_on_failure)
    except Exception as e:
        print(f"❌ Test ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
