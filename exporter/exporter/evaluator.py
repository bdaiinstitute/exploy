# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import time

import numpy as np
import torch

from exporter import ExportableEnvironment
from exporter.context_manager import ContextManager
from exporter.session_wrapper import SessionWrapper
from exporter.utils.math import compare_tensors


def _print_progress_bar(
    step_ctr: int,
    num_steps: int,
    failed_steps: int,
    step_export_ok: bool,
    is_reset_step: bool,
    inference_times: list[float],
) -> None:
    """Print progress bar with step information.

    Args:
        step_ctr: Current step counter.
        num_steps: Total number of steps.
        failed_steps: Number of failed steps so far.
        step_export_ok: Whether current step passed validation.
        is_reset_step: Whether environment was reset this step.
        inference_times: List of inference times.
    """
    status_emoji = "🔴" if not step_export_ok else "🟢"
    progress = (step_ctr + 1) / num_steps
    bar_length = 30
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)

    extra_info = []
    mean_time = np.mean(inference_times) * 1.0e3
    std_time = np.std(inference_times) * 1.0e3
    extra_info.append(f"⏱️  μ={mean_time:.3f}ms σ={std_time:.3f}ms")
    if is_reset_step:
        extra_info.append("RESET")
    extra_str = " | ".join(extra_info)

    print(
        f"\r{status_emoji} {bar} {step_ctr + 1}/{num_steps} | Failed: {failed_steps} | {extra_str} \n",
        end="",
        flush=True,
    )


def _compare_step_outputs(
    env_obs: torch.Tensor,
    ort_obs: torch.Tensor,
    observation_names: list[str],
    env_actions: torch.Tensor,
    ort_actions: torch.Tensor,
    env_outputs: dict[str, torch.Tensor],
    ort_outputs: dict[str, torch.Tensor] | None,
    output_names: list[str],
    verbose: bool,
    atol: float,
    rtol: float,
) -> bool:
    """Compare outputs from environment and ONNX model.

    Args:
        env_obs: Environment observations.
        ort_obs: ONNX model observations.
        observation_names: Names of observation components.
        env_actions: Environment actions.
        ort_actions: ONNX model actions.
        env_outputs: Environment outputs.
        ort_outputs: ONNX model outputs (None if session not run yet).
        output_names: Names of output components.
        verbose: Whether to print comparison results.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        True if all comparisons passed, False otherwise.
    """
    step_export_ok = True

    if verbose:
        print("\n")

    obs_ok = compare_tensors(
        vec_a=env_obs.view(1, -1),
        vec_b=ort_obs.to(env_obs.device).view(1, -1),
        name_a="env",
        name_b="ort",
        vec_name="observation",
        index_names=observation_names,
        verbose=verbose,
        atol=atol,
        rtol=rtol,
    )

    if not obs_ok and verbose:
        print("\n📋 Observation Troubleshooting Checklist:")
        print("  • Verify all observation data sources have corresponding input components")
        print(
            "  • Ensure input components reference the same data sources as observation computation"
        )
        print(
            "  • For memory-based observations, confirm memory components are properly registered"
        )
        print("  • Review compute_observation() implementation for data flow correctness")
        return False

    step_export_ok = step_export_ok and obs_ok

    actions_ok = compare_tensors(
        vec_a=env_actions.view(1, -1),
        vec_b=ort_actions.to(env_actions.device).view(1, -1),
        name_a="env",
        name_b="ort",
        vec_name="actions",
        verbose=verbose,
        atol=atol,
        rtol=rtol,
    )

    if not actions_ok and verbose:
        print("\n📋 Actions Troubleshooting Checklist:")
        print("  • Verify actor network matches between env and ONNX")
        print("  • Ensure action normalizer parameters are correctly exported")
        return False

    step_export_ok = step_export_ok and actions_ok

    # Skip output comparison if we didn't run the session (first step).
    if ort_outputs is not None:
        # Concatenate all outputs for comparison
        env_outputs_cat = torch.cat([env_outputs[name].view(1, -1) for name in output_names], dim=1)
        ort_outputs_cat = torch.cat(
            [ort_outputs[name].to(env_outputs[name].device).view(1, -1) for name in output_names],
            dim=1,
        )

        # Build expanded index names that match the concatenated tensor dimensions
        expanded_index_names = []
        for name in output_names:
            output_size = env_outputs[name].view(1, -1).shape[1]
            expanded_index_names.extend([f"{name}[{i}]" for i in range(output_size)])

        outputs_ok = compare_tensors(
            vec_a=env_outputs_cat,
            vec_b=ort_outputs_cat,
            name_a="env",
            name_b="ort",
            vec_name="outputs",
            index_names=expanded_index_names,
            verbose=verbose,
            atol=atol,
            rtol=rtol,
        )

        if not outputs_ok and verbose:
            print("\n📋 Outputs Troubleshooting Checklist:")
            print("  • Verify output components are registered for all expected outputs")
            print("  • Ensure the correct processed actions are included in memory")
            print("  • Review process_action() and apply_action() implementations for consistency")
            return False

        step_export_ok = step_export_ok and outputs_ok

    return step_export_ok


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
    pause_on_failure: bool = True,
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
    failed_steps = 0
    inference_times = []

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
        onnx_inputs["ctx.step_count"] = np.array([step_ctr], dtype=np.int32)
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
        onnx_inputs["ctx.step_count"] = np.array([0], dtype=np.int32)

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
        inference_times.append(t_inference_s)

        # Get observations and actions. Needs to be called before env.step() to get them
        # from the full model.
        ort_observations = torch.from_numpy(session_wrapper.get_output_value("obs")).clone()
        ort_actions = torch.from_numpy(session_wrapper.get_output_value("actions")).clone()

        # Get the environment's outputs.
        env_outputs = {
            component.output_name: component.get_from_env_cb().clone().cpu()
            for component in context_manager.get_output_components()
        }

        # Compare outputs from environment and ONNX model.
        step_export_ok = _compare_step_outputs(
            env_obs=obs,
            ort_obs=ort_observations,
            env_actions=env_actions,
            ort_actions=ort_actions,
            env_outputs=env_outputs,
            ort_outputs=ort_outputs,
            observation_names=env.get_observation_names(),
            output_names=context_manager.get_output_names(),
            verbose=verbose,
            atol=atol,
            rtol=rtol,
        )

        export_ok = export_ok and step_export_ok
        if not step_export_ok:
            failed_steps += 1

        # Display progress bar.
        if verbose:
            _print_progress_bar(
                step_ctr=step_ctr,
                num_steps=num_steps,
                failed_steps=failed_steps,
                step_export_ok=step_export_ok,
                is_reset_step=is_reset_step,
                inference_times=inference_times,
            )

        if not step_export_ok and verbose and pause_on_failure:
            print(f"⚠️  Step {step_ctr + 1} failed. Press ENTER to continue", end="", flush=True)
            input()

        step_ctr += 1

    return export_ok, next_obs
