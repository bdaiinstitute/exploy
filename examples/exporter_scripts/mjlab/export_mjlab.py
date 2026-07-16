# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Export a MjLab RL policy to ONNX and validate it end-to-end."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.sensor.builtin_sensor import BuiltinSensor
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.runners import OnPolicyRunner

from exploy.exporter.core.components import Input
from exploy.exporter.core.context_manager import ContextManager
from exploy.exporter.core.evaluator import evaluate
from exploy.exporter.core.exporter import export_environment_as_onnx
from exploy.exporter.core.session_wrapper import SessionWrapper
from exploy.exporter.frameworks.mjlab import inputs, memory, outputs
from exploy.exporter.frameworks.mjlab.actor import make_exportable_actor
from exploy.exporter.frameworks.mjlab.env import MjlabExportableEnvironment


def add_imu_inputs(sensors: dict, context_manager: ContextManager):
    """Add IMU sensor inputs to the context manager.

    MjLab's BuiltInSensor class does not allow us to determine the type of the sensor or the body it
    is attached to, so we need to parse the sensor name to determine this information and construct
    the appropriate input names for the ONNX graph.

    Args:
        sensors: A dictionary of sensor names to sensor objects from the MjLab environment.
        context_manager: The ContextManager to which the IMU inputs should be added.
    """
    # mjlab does not allow to infer the type of a BuiltInSensor nor the body it is attached to.
    # We therefore need to parse the sensor name to determine the quantity and body for the imu sensor inputs.
    for sensor_name, sensor in sensors.items():
        if type(sensor) is BuiltinSensor and "imu" in sensor_name.lower():
            imu_name = re.sub(r"[^a-zA-Z0-9_]", "_", str(sensor_name.lower()))
            if "ang_vel" in sensor_name.lower():
                quantity_name = "ang_vel_b_rt_w_in_b"
            elif "lin_vel" in sensor_name.lower():
                quantity_name = "lin_vel_b_rt_w_in_b"
            else:
                continue
            context_manager.add_component(
                Input(
                    name=f"sensor.imu.{imu_name}.{quantity_name}",
                    get_from_env_cb=lambda s=sensor: s.data,
                )
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for exporting a MjLab environment to ONNX.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Export MjLab environment to ONNX")
    parser.add_argument(
        "--task",
        type=str,
        default="Mjlab-Velocity-Flat-Unitree-G1",
        help="Name of the MjLab task to export",
    )
    parser.add_argument(
        "--pause-on-failure",
        action="store_true",
        default=False,
        help="Pause on evaluation failure (useful for debugging)",
    )
    return parser.parse_args()


def export_mjlab(
    task_name: str = "Mjlab-Velocity-Flat-Unitree-G1",
    pause_on_failure: bool = False,
) -> None:
    """Export and evaluate a MjLab ONNX policy.

    Args:
        task_name: The name of the MjLab task to export. Defaults to "Mjlab-Velocity-Flat-Unitree-G1".
        pause_on_failure: Whether to pause execution if the exported ONNX policy fails during evaluation.
            Defaults to False.
    """

    task_device = "cpu"

    # Get the environment's configuration and create the environment.
    env_cfg = load_env_cfg(task_name, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.events.pop("encoder_bias", None)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=task_device, render_mode=None)
    env.reset()

    # Get the agent configuration and wrap the environment for inference.
    agent_cfg: RslRlOnPolicyRunnerCfg = load_rl_cfg(task_name)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Make a runner to access the policy and create an exportable environment wrapper.
    runner_cls = load_runner_cls(task_name) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=task_device)
    runner.logger_type = "disabled"

    # Make an exportable environment. See the exporter tutorial for more discussion on this step.
    # The exportable environment wraps the original MjLab environment and provides the necessary
    # interfaces for ONNX export and evaluation.
    unwrapped_env = env.unwrapped
    exportable_env = MjlabExportableEnvironment(unwrapped_env)

    # Make an exportable actor from the runner's policy. This involves extracting the policy's actor
    # network and wrapping it in a way that makes it compatible with ONNX export. See the exporter
    # tutorial and the make_exportable_actor function for more details on this step.
    alg: PPO = runner.alg
    assert isinstance(alg, PPO), f"Expected PPO algorithm, got: {type(alg).__name__}"
    actor = make_exportable_actor(exportable_env, alg.actor, device=task_device)

    # Get the exportable environemtn's context manager. The context manager is responsible for
    # managing the inputs and outputs of the ONNX graph during export and evaluation.
    entities = unwrapped_env.scene.entities
    context_manager = exportable_env.context_manager()

    # Add inputs, outputs, and memory to the context manager. The specific inputs and outputs will
    # depend on the environment and the policy, but typically include things like joint positions,
    # velocities, sensor readings, and actions. See the exporter tutorial and the inputs, outputs,
    # and memory modules for more details on how to define these for a specific environment and
    # policy.
    inputs.add_commands(unwrapped_env.command_manager, context_manager)
    inputs.add_base_com_vel(entities, context_manager)
    inputs.add_body_pos_and_quat(entities, context_manager)
    inputs.add_body_vel(entities, context_manager)
    inputs.add_joint_pos_and_vel(entities, context_manager)
    inputs.add_sensor_inputs(unwrapped_env.scene.sensors, context_manager)
    add_imu_inputs(unwrapped_env.scene.sensors, context_manager)
    memory.add_memory(unwrapped_env, context_manager)
    outputs.add_outputs(unwrapped_env.action_manager, context_manager)

    test_dir = pathlib.Path(__file__).parent / "exporter_tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    onnx_file = "test_export.onnx"

    # Export to ONNX.
    export_environment_as_onnx(
        env=exportable_env,
        actor=actor,
        path=test_dir,
        filename=onnx_file,
        verbose=False,
    )
    exportable_env.cleanup()

    # Prepare an ONNX session wrapper for evaluation. The session wrapper handles loading the ONNX
    # model and running inference with it. See the SessionWrapper class for more details on its
    # functionality and how to customize it for specific needs (e.g. optimization, custom
    # input/output handling, etc.).
    session_wrapper = SessionWrapper(
        onnx_folder=test_dir,
        onnx_file_name=onnx_file,
        actor=actor,
        optimize=True,
    )

    # Run evaluation. The evaluate function steps both the environment and the ONNX model in
    # parallel and compares their outputs to ensure that the exported model correctly captured the
    # the environment's computation. See the evaluate function for more details.
    with torch.inference_mode():
        export_ok, _ = evaluate(
            env=exportable_env,
            context_manager=exportable_env.context_manager(),
            session_wrapper=session_wrapper,
            num_episodes=2,
            max_episode_steps=100,
            verbose=True,
            pause_on_failure=pause_on_failure,
        )

    env.close()

    assert export_ok, "ONNX export validation failed"
    print("✓ Export and evaluation passed.")


if __name__ == "__main__":
    args = parse_args()
    try:
        export_mjlab(task_name=args.task, pause_on_failure=args.pause_on_failure)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
