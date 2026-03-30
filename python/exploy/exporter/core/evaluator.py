# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import time

import numpy as np
import torch

from exploy.exporter.core.context_manager import ContextManager
from exploy.exporter.core.exportable_environment import ExportableEnvironment
from exploy.exporter.core.session_wrapper import SessionWrapper
from exploy.exporter.core.utils.math import compare_tensors


def _print_progress_bar(
    step_ctr: int,
    num_steps: int,
    failed_steps: int,
    step_export_ok: bool,
    inference_times: list[float],
) -> None:
    """Print progress bar with step information.

    Args:
        step_ctr: Current step counter.
        num_steps: Total number of steps.
        failed_steps: Number of failed steps so far.
        step_export_ok: Whether current step passed validation.
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
    extra_str = " | ".join(extra_info)

    print(
        f"\r{status_emoji} {bar} {step_ctr + 1}/{num_steps} | Failed: {failed_steps} | {extra_str}",
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
    atol: float,
    rtol: float,
) -> tuple[bool, str]:
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
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        A tuple of (is_step_export_ok, message), where `is_step_export_ok` is a boolean indicating whether the step's outputs are close within the specified tolerances, and `message` is a string describing the comparison results, including details of any mismatches and troubleshooting checklists.
    """
    step_export_ok = True
    msg = "\n"

    # Check observation comparison
    obs_env = env_obs.view(1, -1)
    obs_ort = ort_obs.to(env_obs.device).view(1, -1)
    obs_ok, obs_message = compare_tensors(
        vec_a=obs_env,
        vec_b=obs_ort,
        name_a="env",
        name_b="ort",
        vec_name="observation",
        index_names=observation_names,
        atol=atol,
        rtol=rtol,
    )

    if not obs_ok:
        msg += "\n📋 Observation comparison failed!"
        msg += f"\n{obs_message}"
        msg += "\n📋 Observation Troubleshooting Checklist:"
        msg += "\n  • Verify all observation data sources have corresponding input components"
        msg += "\n  • Ensure input components reference the same data sources as observation computation"
        msg += (
            "\n  • For memory-based observations, confirm memory components are properly registered"
        )
        msg += "\n  • Review compute_observation() implementation for data flow correctness"
        return False, msg
    else:
        msg += "\n✅ Observations match between environment and ONNX model."

    step_export_ok = step_export_ok and obs_ok

    # Check actions comparison
    actions_env = env_actions.view(1, -1)
    actions_ort = ort_actions.to(env_actions.device).view(1, -1)
    actions_ok, actions_message = compare_tensors(
        vec_a=actions_env,
        vec_b=actions_ort,
        name_a="env",
        name_b="ort",
        vec_name="actions",
        index_names=None,
        atol=atol,
        rtol=rtol,
    )

    if not actions_ok:
        msg += "\n🎯 Actions comparison failed!"
        msg += f"\n{actions_message}"
        msg += "\n📋 Actions Troubleshooting Checklist:"
        msg += "\n  • Verify actor network matches between env and ONNX"
        msg += "\n  • Ensure action normalizer parameters are correctly exported"
        return False, msg
    else:
        msg += "\n✅ Actions match between environment and ONNX model."

    step_export_ok = step_export_ok and actions_ok

    # Skip output comparison if we didn't run the session (first step).
    if ort_outputs is not None and len(output_names) > 0:
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

        # Check outputs comparison
        outputs_ok, outputs_message = compare_tensors(
            vec_a=env_outputs_cat,
            vec_b=ort_outputs_cat,
            name_a="env",
            name_b="ort",
            vec_name="outputs",
            index_names=expanded_index_names,
            atol=atol,
            rtol=rtol,
        )

        if not outputs_ok:
            msg += "\n📊 Outputs comparison failed!"
            msg += f"\n{outputs_message}"
            msg += "\n📋 Outputs Troubleshooting Checklist:"
            msg += "\n  • Verify output components are registered for all expected outputs"
            msg += "\n  • Ensure the correct processed actions are included in memory"
            msg += (
                "\n  • Review process_action() and apply_action() implementations for consistency"
            )
            return False, msg
        else:
            msg += "\n✅ Outputs match between environment and ONNX model."

        step_export_ok = step_export_ok and outputs_ok

    return step_export_ok, msg


def evaluate(
    env: ExportableEnvironment,
    context_manager: ContextManager,
    session_wrapper: SessionWrapper,
    num_episodes: int,
    max_episode_steps: int | None = None,
    verbose: bool = True,
    reset_from_onnx_counter_steps: int = 50,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
    pause_on_failure: bool = True,
) -> tuple[bool, torch.Tensor]:
    """Evaluate an ONNX exported model against an `ExportableEnvironment` stepped through a `SessionWrapper`.

    This function runs the simulation for a specified number of episodes, each with a maximum number
    of steps, and compares the outputs of the ONNX model with the environment's state and actor's
    actions at each step. This is useful for verifying the correctness of the ONNX export.

    Args:
        env: The environment to run the evaluation in.
        context_manager: The context manager handling inputs and outputs.
        session_wrapper: An ONNX session wrapper.
        num_episodes: The number of episodes to run the evaluation for.
        max_episode_steps: The maximum number of steps per episode.
        verbose: Whether to print verbose output during evaluation. Defaults to True.
        reset_from_onnx_counter_steps: Set after how many steps we should set memory inputs from ONNX instead of using
            the environment's state.

            Note: we do this to avoid numerical error accumulation that would occur if we only every use
            the ONNX inference outputs as memory fed back as ONNX inference inputs, while
            all other inputs are set directly from the environment's state.

            Note: this value is chosen arbitrarily.
        atol: Absolute tolerance used to compare tensors.
        rtol: Relative tolerance used to compare tensors.
        pause_on_failure: Whether to pause on each failed step and wait for user input before
            continuing. Defaults to True.

    Returns:
        A tuple containing a boolean indicating if the evaluation was successful and
        the final observations tensor.
    """
    if verbose:
        print("Starting evaluation...")

    for i_episode in range(num_episodes):
        if verbose:
            print(f"\nStarting episode {i_episode + 1}/{num_episodes}...")
        export_ok, final_obs = evaluate_episode(
            env=env,
            context_manager=context_manager,
            session_wrapper=session_wrapper,
            max_num_steps=max_episode_steps,
            verbose=verbose,
            reset_from_onnx_counter_steps=reset_from_onnx_counter_steps,
            atol=atol,
            rtol=rtol,
            pause_on_failure=pause_on_failure,
        )

    if verbose:
        print("\nEvaluation complete.")

    return export_ok, final_obs


def evaluate_episode(
    env: ExportableEnvironment,
    context_manager: ContextManager,
    session_wrapper: SessionWrapper,
    max_num_steps: int | None = None,
    verbose: bool = True,
    reset_from_onnx_counter_steps: int = 50,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
    pause_on_failure: bool = True,
):
    """Run evaluation for a single episode, comparing the ONNX model outputs against the environment.

    Steps the environment and the ONNX session in lockstep, comparing observations, actions, and
    outputs at each step. Useful for fine-grained inspection of a single episode, e.g. in a
    debugging loop, without the outer episode iteration of :func:`evaluate`.

    Args:
        env: The environment to run the evaluation in.
        context_manager: The context manager handling inputs and outputs.
        session_wrapper: An ONNX session wrapper.
        max_num_steps: The maximum number of steps to run. If ``None``, runs until the environment
            signals a reset.
        verbose: Whether to print verbose output during evaluation. Defaults to True.
        reset_from_onnx_counter_steps: Set after how many steps we should set memory inputs from
            ONNX instead of using the environment's state.

            Note: we do this to avoid numerical error accumulation that would occur if we only
            ever use the ONNX inference outputs as memory fed back as ONNX inference inputs,
            while all other inputs are set directly from the environment's state.

            Note: this value is chosen arbitrarily.
        atol: Absolute tolerance used to compare tensors.
        rtol: Relative tolerance used to compare tensors.
        pause_on_failure: Whether to pause on each failed step and wait for user input before
            continuing. Defaults to True.

    Returns:
        A tuple containing a boolean indicating if the episode evaluation was successful and
        the final observations tensor.
    """
    obs = env.observations_reset()
    session_wrapper.reset()

    actor = session_wrapper.get_actor()
    if actor is None:
        raise ValueError(
            "Session wrapper has no actor. Cannot evaluate ONNX model without access to original actor for comparison."
        )
    actor.reset(torch.tensor([True], device=obs.device))

    step_ctr = 0
    export_ok = True
    failed_steps = 0
    inference_times = []

    # Hold a dictionary of environment outputs to compare against ONNX outputs at each step.
    # This is populated at each decimation step of each environment update, and later
    # compared against the ONNX outputs after running inference.
    env_outputs = {}

    def evaluate_substep(step_ctr: int):
        """Evaluate a single substep at sim dt.

        Args:
            step_ctr: The current decimation step counter.
        """
        # Get the environment's outputs.
        for component in context_manager.get_output_components():
            env_outputs[component.output_name] = component.get_from_env_cb().clone().cpu()

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
        """Callback passed to the environment's evaluation hooks to update the
        context manager's inputs from the environment's state at each step.
        """
        context_manager.read_inputs()

    env.register_evaluation_hooks(
        update=update,
        evaluate_substep=evaluate_substep,
    )

    # Compute actions for the initial observations.
    env_actions: torch.Tensor = actor(obs)

    reset_memory_from_env = False
    env.context_manager().read_inputs()

    while step_ctr < max_num_steps if max_num_steps is not None else True:
        reset_memory_from_env = (
            reset_memory_from_env or (step_ctr % reset_from_onnx_counter_steps) == 0
        )
        next_obs, is_reset_step = env.step(env_actions)
        # Use the environment's observations for the next step.
        obs[:] = next_obs

        # Check if the environment was reset.
        if is_reset_step:
            if verbose:
                print(f"\n🔄 Environment reset at step {step_ctr + 1}.")
            break

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

        # Get observations and actions.
        ort_observations = torch.from_numpy(session_wrapper.get_output_value("obs")).clone()
        ort_actions = torch.from_numpy(session_wrapper.get_output_value("actions")).clone()

        # Compute actions from the new observations.
        env_actions = actor(obs)

        # Compare outputs from environment and ONNX model.
        step_export_ok, msg = _compare_step_outputs(
            env_obs=obs,
            ort_obs=ort_observations,
            env_actions=env_actions,
            ort_actions=ort_actions,
            env_outputs=env_outputs,
            ort_outputs=ort_outputs,
            observation_names=env.get_observation_names(),
            output_names=context_manager.get_output_names(),
            atol=atol,
            rtol=rtol,
        )

        env_outputs = {}

        export_ok = export_ok and step_export_ok
        if not step_export_ok:
            failed_steps += 1

        # Display progress bar.
        if verbose:
            if not step_export_ok:
                print(msg)
            _print_progress_bar(
                step_ctr=step_ctr,
                num_steps=max_num_steps,
                failed_steps=failed_steps,
                step_export_ok=step_export_ok,
                inference_times=inference_times,
            )

        if not step_export_ok and verbose and pause_on_failure:
            print(f"⚠️  Step {step_ctr + 1} failed. Press ENTER to continue", end="", flush=True)
            input()

        step_ctr += 1

    return export_ok, next_obs
