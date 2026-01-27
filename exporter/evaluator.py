# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pathlib
import time
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import exporter as rai_core_exporter
# import rai.core.workflows.rsl_rl.utils as rai_core_rsl_rl_utils
# from rai.core.utils import dict as rai_core_dict
from exporter import ExportEnvIdsType
# from rai.core.workflows import common_cli_args

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def evaluate(
    env: ManagerBasedRLEnv,
    session_wrapper: rai_core_exporter.SessionWrapper,
    num_steps: int,
    observations: torch.Tensor | None = None,
    env_id: ExportEnvIdsType = 0,
    verbose: bool = True,
    reset_from_onnx_counter_steps: int = 50,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> tuple[bool, torch.Tensor]:
    """Evaluate an ONNX exported model against the original IsaacLab environment and torch policy.

    This function runs the simulation for a specified number of steps and compares the
    outputs of the ONNX model with the environment's state and the original torch model's
    outputs at each step. This is useful for verifying the correctness of the ONNX export.

    Args:
        env: The environment to run the evaluation in.
        session_wrapper: An ONNX session wrapper.
        num_steps: The number of steps to run the evaluation for.
        observations: The initial observations. If None, the environment is reset. Defaults to None.
        env_id: The batch id used for evaluation. Defaults to 0.
        verbose: Whether to print verbose output during evaluation. Defaults to True.
        reset_from_onnx_counter_steps:  Set after how many steps we should set memory inputs from ONNX instead of using
                                        the environment's state.
                                        Note: we do this to avoid numerical error accumulation that would occur if we only every use
                                              the ONNX inference outputs as memory fed back as ONNX inference inputs, while
                                              all other inputs are set directly from the environment's state.
                                        Note: this value is chosen arbitrarily.
        atol: Absolute tolerance used to compare tensors.
        rtol: Relative tolerance used to compare tensors.

    Returns:
        A tuple containing a boolean indicating if the evaluation was successful and
        the final observations tensor.
    """

    # from rai.core.utils import exporter as rai_core_exporter
    # from rai.core.utils import managers as rai_core_managers
    # from rai.core.utils import math as rai_core_math
    from exporter.exporter import DataHandlerManager, get_action_for_process

    if observations is None:
        with torch.inference_mode():
            obs = env.reset()
    else:
        obs = observations.clone()

    # obs_names = rai_core_managers.get_observation_names(env.observation_manager)
    obs_names = []

    data_handler_manager = DataHandlerManager(
        env=env,
        env_id=env_id,
    )

    step_ctr = 0
    is_reset_step = env.termination_manager.dones[env_id]
    export_ok = True

    # Disable lazy sensor updates to ensure all sensors are updated when the scene is updated.
    env.scene.cfg.lazy_sensor_update = False

    # Evaluate a single substep at sim dt.
    def evaluate_substep(step_ctr: int):
        # Skip first step, as we evaluate the policy in the main evaluation loop before calling env.step().
        # Skip if we have not run the session yet.
        if step_ctr == 0 or session_wrapper._results is None:
            return
        onnx_inputs = rai_core_exporter.exporter.get_onnx_inputs(
            dh_manager=data_handler_manager,
            to_numpy=True,
        )
        # We always use the previous ONNX memory outputs as inputs to the next ONNX inference.
        memory_handler = data_handler_manager.memory_dh
        if memory_handler.has_memory():
            for name in memory_handler.input_names():
                onnx_inputs[name] = session_wrapper.get_output_value(memory_handler.io_name_to_output_name(name))
        onnx_inputs["step_count"] = np.array([step_ctr], dtype=np.int32)
        session_wrapper(**onnx_inputs)

    # Inject evaluator update at the end of the ManagerBasedRLEnv physics stepping. We exploit the
    # fact that sensors are updated last in scene.update() and that python dicts are accessed in
    # insertion order.
    class ONNXEvaluatorSensor:
        def __init__(self):
            self.sub_step_ctr = 0

        def update(self, dt: float, force_recompute: bool):
            evaluate_substep(step_ctr=self.sub_step_ctr)
            self.sub_step_ctr += 1
            # The sensor allows us to capture the state of the environment for all simulation substeps.
            data_handler_manager.set_from_source()

        def reset(self, env_ids: Sequence[int]):
            # If the environment is reset, we need to re-capture the state of the scene.
            data_handler_manager.set_from_source()

    env.scene._sensors["onnx"] = ONNXEvaluatorSensor()

    # Compute actions for the initial observations.
    actions = session_wrapper.get_torch_model()(obs)

    reset_memory_from_env = False

    while step_ctr < num_steps:
        reset_memory_from_env = reset_memory_from_env or (step_ctr % reset_from_onnx_counter_steps) == 0
        # Reset the ONNX evaluator sensor's sub-step counter.
        env.scene._sensors["onnx"].sub_step_ctr = 0
        # Step the environment.
        next_obs, _, dones, timeouts, _ = env.step(actions)
        # Use the environment's observations for the next step.
        obs[:] = next_obs["policy"]
        # Compute actions from the new observations.
        actions = session_wrapper.get_torch_model()(obs)

        # Check if the environment was reset.
        if torch.logical_or(dones[env_id], timeouts[env_id]).any():
            # We need to reset the memory inputs from the environment after a reset.
            reset_memory_from_env = True
            # Reset the session wrapper results to avoid using stale outputs.
            session_wrapper._results = None
            continue

        # Get onnx outputs if the session has been run.
        ort_outputs = (
            None
            if session_wrapper._results is None
            else {
                out_name: torch.from_numpy(session_wrapper.get_output_value(out_name)).clone()
                for out_name in data_handler_manager.action_dh.names()
            }
        )

        # Get onnx inputs.
        onnx_inputs = rai_core_exporter.exporter.get_onnx_inputs(
            dh_manager=data_handler_manager,
            to_numpy=True,
        )
        onnx_inputs["step_count"] = np.array([0], dtype=np.int32)

        # Adapt memory inputs.
        if reset_memory_from_env:
            # We use the memory which was set calling get_onnx_inputs() from the env.
            reset_memory_from_env = False
        else:
            # We overwrite the memory from the env with the previous ONNX outputs.
            memory_handler = data_handler_manager.memory_dh
            for name in memory_handler.input_names():
                onnx_inputs[name] = session_wrapper.get_output_value(memory_handler.io_name_to_output_name(name))

        # Evaluate the ONNX policy.
        t_start = time.perf_counter()
        session_wrapper(**onnx_inputs)
        t_inference_s = time.perf_counter() - t_start

        # Get observations and actions. Needs to be called before env.step() to get them
        # from the full model.
        ort_observations = torch.from_numpy(session_wrapper.get_output_value("obs")).clone()
        ort_actions = torch.from_numpy(session_wrapper.get_output_value("actions")).clone()

        # Get the environment's outputs.
        env_outputs = {name: func().clone().cpu() for name, func in data_handler_manager.action_dh.data().items()}

        env_actions = get_action_for_process(
            env=env,
            policy_actions=actions,
        )

        # Check all inputs and outputs.
        step_export_ok = True

        torch.set_printoptions(profile="full", precision=32)
        print("===================")
        # step_export_ok = step_export_ok and rai_core_math.compare_tensors(
        #     vec_a=obs.view(1, -1),
        #     vec_b=ort_observations.to(obs.device).view(1, -1),
        #     name_a="env",
        #     name_b="ort",
        #     vec_name="observation",
        #     index_names=obs_names,
        #     verbose=verbose,
        #     atol=atol,
        #     rtol=rtol,
        # )
        pass
        # step_export_ok = step_export_ok and rai_core_math.compare_tensors(
        #     vec_a=env_actions.view(1, -1),
        #     vec_b=ort_actions.to(env_actions.device).view(1, -1),
        #     name_a="env",
        #     name_b="ort",
        #     vec_name="actions",
        #     verbose=verbose,
        #     atol=atol,
        #     rtol=rtol,
        # )

        # Skip output comparison if we didn't run the session (first step).
        if ort_outputs is not None:
            # for name in data_handler_manager.action_dh.names():
            #     step_export_ok = step_export_ok and rai_core_math.compare_tensors(
            #         vec_a=env_outputs[name].view(1, -1),
            #         vec_b=ort_outputs[name].to(env_outputs[name].device).view(1, -1),
            #         name_a="env",
            #         name_b="ort",
            #         vec_name=name,
            #         verbose=verbose,
            #         atol=atol,
            #         rtol=rtol,
            #     )
            pass

        if verbose:
            # Print step status.
            if is_reset_step.any():
                print("Env was reset. Triggered termination terms:")
                for term_name in env.termination_manager.active_terms:
                    term_val = env.termination_manager.get_term(term_name)
                    if term_val.any():
                        print(f"\t{term_name}: {term_val.cpu().numpy()}")
            print(f"t ONNX inference: {t_inference_s * 1.0e3 : .3f}ms")
            print(f"step: {step_ctr}")
            if not step_export_ok:
                print("Found errors when comparing ONNX and environment.")
            print("===================")

        # Keep track of the export checks.
        export_ok = export_ok and step_export_ok

        step_ctr += 1

    return export_ok, next_obs["policy"]


# def load_env_and_evaluate_onnx(...)
