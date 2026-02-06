# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import time

import numpy as np
import torch

from exporter import ExportableEnvironment
from exporter.context_manager import ContextManager
from exporter.session_wrapper import SessionWrapper
from exporter.utils.math import compare_tensors


def evaluate(
    env: ExportableEnvironment,
    context_manager: ContextManager,
    session_wrapper: SessionWrapper,
    num_steps: int,
    observations: torch.Tensor | None = None,
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
        context_manager: The context manager handling inputs and outputs.
        session_wrapper: An ONNX session wrapper.
        num_steps: The number of steps to run the evaluation for.
        observations: The initial observations. If None, the environment is reset. Defaults to None.
        verbose: Whether to print verbose output during evaluation. Defaults to True.
        reset_from_onnx_counter_steps: Set after how many steps we should set memory inputs from ONNX instead of using
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

    obs = observations.clone() if observations is not None else env.observations_reset()

    step_ctr = 0
    export_ok = True

    # Evaluate a single substep at sim dt.
    def evaluate_substep(step_ctr: int):
        # Skip first step, as we evaluate the policy in the main evaluation loop before calling env.step().
        # Skip if we have not run the session yet.
        if step_ctr == 0 or session_wrapper._results is None:
            return
        onnx_inputs = context_manager.get_inputs(to_numpy=True)

        # We always use the previous ONNX memory outputs as inputs to the next ONNX inference.
        memory_components = context_manager.get_memory_components()
        for memory in memory_components:
            onnx_inputs[memory.input_name] = session_wrapper.get_output_value(memory.output_name)
        onnx_inputs["step_count"] = np.array([step_ctr], dtype=np.int32)
        session_wrapper(**onnx_inputs)

    def update():
        context_manager.read_inputs()

    def reset():
        context_manager.read_inputs()

    env.register_evaluation_hooks(
        update=update,
        reset=reset,
        evaluate_substep=evaluate_substep,
    )

    # Compute actions for the initial observations.
    env_actions: torch.Tensor = session_wrapper.get_torch_model()(obs)

    reset_memory_from_env = False

    while step_ctr < num_steps:
        reset_memory_from_env = (
            reset_memory_from_env or (step_ctr % reset_from_onnx_counter_steps) == 0
        )
        next_obs, is_reset_step = env.step(env_actions)
        # Use the environment's observations for the next step.
        obs[:] = next_obs
        # Compute actions from the new observations.
        env_actions = session_wrapper.get_torch_model()(obs)

        # Check if the environment was reset.
        if is_reset_step:
            # We need to reset the memory inputs from the environment after a reset.
            reset_memory_from_env = True
            # Reset the session wrapper results to avoid using stale outputs.
            session_wrapper._results = None

        # Get onnx outputs if the session has been run.
        ort_outputs = (
            None
            if session_wrapper._results is None
            else {
                out_name: torch.from_numpy(session_wrapper.get_output_value(out_name)).clone()
                for out_name in context_manager.get_output_names()
            }
        )

        # Get onnx inputs.
        onnx_inputs = context_manager.get_inputs(to_numpy=True)
        onnx_inputs["step_count"] = np.array([0], dtype=np.int32)

        # Adapt memory inputs.
        if reset_memory_from_env:
            # We use the memory which was set calling get_onnx_inputs() from the env.
            reset_memory_from_env = False
        else:
            # We overwrite the memory from the env with the previous ONNX outputs.
            memory_components = context_manager.get_memory_components()
            for memory in memory_components:
                onnx_inputs[memory.input_name] = session_wrapper.get_output_value(
                    memory.output_name
                )

        # Evaluate the ONNX policy.
        t_start = time.perf_counter()
        session_wrapper(**onnx_inputs)
        t_inference_s = time.perf_counter() - t_start

        # Get observations and actions. Needs to be called before env.step() to get them
        # from the full model.
        ort_observations = torch.from_numpy(session_wrapper.get_output_value("obs")).clone()
        ort_actions = torch.from_numpy(session_wrapper.get_output_value("actions")).clone()

        # Get the environment's outputs.
        env_outputs = {
            component.output_name: component.get_from_env_cb().clone().cpu()
            for component in context_manager.get_output_components()
        }

        # Check all inputs and outputs.
        step_export_ok = True

        torch.set_printoptions(profile="full", precision=32)
        print("===================")
        step_export_ok = step_export_ok and compare_tensors(
            vec_a=obs.view(1, -1),
            vec_b=ort_observations.to(obs.device).view(1, -1),
            name_a="env",
            name_b="ort",
            vec_name="observation",
            index_names=env.get_observation_names(),
            verbose=verbose,
            atol=atol,
            rtol=rtol,
        )

        step_export_ok = step_export_ok and compare_tensors(
            vec_a=env_actions.view(1, -1),
            vec_b=ort_actions.to(env_actions.device).view(1, -1),
            name_a="env",
            name_b="ort",
            vec_name="actions",
            verbose=verbose,
            atol=atol,
            rtol=rtol,
        )

        # Skip output comparison if we didn't run the session (first step).
        if ort_outputs is not None:
            for name in context_manager.get_output_names():
                step_export_ok = step_export_ok and compare_tensors(
                    vec_a=env_outputs[name].view(1, -1),
                    vec_b=ort_outputs[name].to(env_outputs[name].device).view(1, -1),
                    name_a="env",
                    name_b="ort",
                    vec_name=name,
                    verbose=verbose,
                    atol=atol,
                    rtol=rtol,
                )

        if verbose:
            # Print step status.
            if is_reset_step:
                print("Env was reset.")
            print(f"t ONNX inference: {t_inference_s * 1.0e3: .3f}ms")
            print(f"step: {step_ctr}")
            if not step_export_ok:
                print("Found errors when comparing ONNX and environment.")
            else:
                print("ONNX and environment outputs match.")
            print("===================")

        # Keep track of the export checks.
        export_ok = export_ok and step_export_ok

        step_ctr += 1

    return export_ok, next_obs
