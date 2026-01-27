# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import contextlib
import copy
import dataclasses
import functools
import gymnasium as gym
import pathlib
import queue as queue_python
import re
import shutil
import signal
import sys
import tempfile
import time
import torch
import torch.multiprocessing as mp
from argparse import Namespace
from collections.abc import Generator
from multiprocessing.context import SpawnProcess
from typing import Any

import carb  # noqa: F401, F811
import omni.log
import wandb

# import rai.core.utils.dict
from exporter import ExportEnvIdsType
from exporter.exporter import _OnnxEnvironmentExporter

import isaaclab.managers.manager_term_cfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg


class OnnxEnvExporter:
    def __init__(self, task_name: str) -> None:
        """
        Exporter of training checkpoints in ONNX format, compatible with RSL RL callbacks.

        The ONNX files produced by this class include in the graph the logic to compute the observation from the state,
        and obtaining the actuated signals from the action (raw policy output).

        This class creates a separate process (exporter process) that does not interfere with the training process.
        A new unvectorized environment is created with a matching configuration of the training environment.

        Args:
            task_name: The name of the task used to build the exported environment.
        """

        if sys.platform != "linux":
            msg = "This class uses linux-specific signals to prevent hanging processes."
            msg += f"Current platform '{sys.platform}' is unsupported."
            raise NotImplementedError(msg)

        # Name of the environment created for exporting the policy.
        # The logic of its action and observation managers will be included in the ONNX graph.
        self.task_name = task_name

        # This is populated at the first checkpoint (iteration 0) with the current configuration
        # of the training environment.
        self.env_cfg: ManagerBasedRLEnvCfg | None = None

    def __del__(self) -> None:
        # This is an additional insurance to trigger the graceful termination of the exporter process when the
        # OnnxEnvExporter object gets garbage-collected.
        self.stop()

    @functools.cached_property
    def process_data(self) -> tuple[SpawnProcess, mp.Queue]:
        """
        Property returning the ONNX exporter process and queue, both created lazily.

        Return:
            A tuple containing the started ONNX exporter process and the corresponding queue.
        """

        # CUDA can't be re-initialized in subprocesses created with `fork`, so we use the `spawn` start method instead.
        # To avoid changing the start method of the training process with possible unknown implications,
        # we create a local multiprocessing 'spawn' context and use it to set up all required resources.
        # Note: the multiprocess implementation used here comes from torch.multiprocessing.
        ctx = mp.get_context("spawn")

        # Create the queue.
        # It will be used to send to the exporter process necessary data to create the ONNX file of the checkpoint.
        queue: mp.Queue = ctx.Queue()

        # Start the exporter process.
        # This process will initialize and create the exported environment as soon as it launches.
        # After setup, it will wait for incoming messages on the queue to proceed with exporting ONNX files.
        process = ctx.Process(
            target=OnnxEnvExporter.worker,
            args=(queue, self.task_name, self.env_cfg),
        )

        # Start the process.
        process.start()

        return process, queue

    @staticmethod
    def worker(
        queue: mp.Queue,
        task_name: str,
        env_cfg: ManagerBasedRLEnvCfg,
    ) -> None:
        """
        Worker function running in the exporter process.

        Args:
            queue: The queue to receive data from the main process.
            task_name: The name of the task used to build the exported environment.
            env_cfg: The configuration of the exported environment.

        Note:
            Considering how our simulation propagates the environment configuration, this worker function
            expects to receive the configuration of the training environment. Since the exported environment
            does not need to actually run, but is used only for generating the graph later stored in the ONNX file,
            this method may slightly adjust the training configuration.
        """

        # Note: this function will run in a different process.
        # This is why there are some local imports in the body.

        # Importing carb is necessary to get omni.logs in the output, not sure why.
        import carb  # noqa: F401, F811
        import omni.log

        # First attempt of clean termination of the exporter process.
        # If the main process receives termination signals, they might get propagated to the children.
        # We try to catch them and stop the while loop that awaits for checkpoint data.
        # Note that we do this in the child process so that we don't risk to mess up signal handling
        # of the main process. In this way, we reduce possible interferences with the training.
        def signal_handler(signum, frame) -> None:
            print(f"[INFO][OnnxEnvExporter.worker] Received signal '{signum}'. Triggering graceful stop.")
            global keep_running
            keep_running = False

        # Register the handler to stop the process when it receives various termination signals.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            msg = "[INFO][OnnxEnvExporter.worker] Registering handler for signal '{sig}' in the ONNX exporter process."
            print(msg.format(sig=sig))
            signal.signal(sig, signal_handler)

        # Second attempt of clean termination of the exported process.
        # The first termination approach is not robust to a SIGKILL of the parent process.
        # When that happens, this child process would hang in the OS as zombie process, with the
        # additional consequence to consume GPU resources.
        # Since we only currently support Linux systems, the following method instructs the kernel
        # to kill this child process if the parent abruptly dies.
        # This should guarantee that the system remains clean, and we cannot do much more than this.
        import ctypes
        import ctypes.util

        libc_path = ctypes.util.find_library("c")

        if libc_path is None:
            msg = "Failed to find libc in the OS. The exporter process might not get terminated properly."
            omni.log.error(msg)
        else:
            # The kernel will send a SIGKILL to the exporter process if the training process gets SIGKILL-ed.
            PR_SET_PDEATHSIG = 1
            libc = ctypes.CDLL(libc_path)
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)

        # ===================================
        # Initialize the exported environment
        # ===================================

        import contextlib
        import io

        # Buffer to redirect stdout/stderr.
        buffer = io.StringIO()

        # Create the environment capturing all the stdout and stderr, that otherwise would
        # pollute the training output. It gets printed in case of errors while creating
        # the exported environment.
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                env = gym.make(id=task_name, cfg=env_cfg, render_mode=None)
        except Exception as e:
            msg = "Failed to create environment for ONNX exporting."
            omni.log.error(msg)
            print(buffer.getvalue())
            raise RuntimeError(msg) from e

        # =======================================================
        # While loop awaiting training data from the main process
        # =======================================================

        # Note that this is global only in the exporter process.
        # We need this global variable to handle correctly termination through OS signals.
        global keep_running
        keep_running = True

        # Helper function to export the ONNX file.
        def export_onnx(
            onnx_path: pathlib.Path,
            env: ManagerBasedRLEnv,
            actor: torch.nn.Module,
            normalizer: torch.nn.Module,
        ) -> None:
            # Verify that the path to the output directory, if it already exists, is a directory.
            if onnx_path.parent.exists() and not onnx_path.parent.is_dir():
                raise NotADirectoryError(onnx_path.parent)

            # Create the directory if it doesn't exist.
            if not onnx_path.parent.exists():
                onnx_path.parent.mkdir(parents=True, exist_ok=True)

            # Define which environment IDs to export.
            export_env_ids: ExportEnvIdsType = 0

            # Create the policy exporter.
            assert isinstance(env, ManagerBasedRLEnv), type(env)
            policy_exporter = _OnnxEnvironmentExporter(
                env=env,
                export_env_ids=export_env_ids,
                actor=actor,
                normalizer=normalizer,
                verbose=False,
            )

            # Export the policy.
            with torch.inference_mode():
                policy_exporter.export(
                    onnx_path=str(onnx_path.parent),
                    onnx_file_name=onnx_path.name,
                    model_source=dict(),
                )
            print(f"[INFO][OnnxEnvExporter.worker] Exported successfully ONNX file '{onnx_path.name}'.")

        while keep_running:
            # Create the mutex to enforce exporting ONNX file one at a time.
            lock = mp.Lock()

            # Get the training data from the training process.
            # We use a blocking polled query from the queue since this is the most reliable
            # method to prevent that this exporter process hangs forever.
            try:
                (
                    export_model_dir,
                    file_name,
                    args_cli,
                    (actor, normalizer),
                ) = queue.get(block=True, timeout=1)

                # Notify that the worker received data.
                carb.log_info("[OnnxEnvExporter.worker] Received data from the training process.")

            except queue_python.Empty:
                continue

            with lock:
                try:
                    export_onnx(
                        onnx_path=(pathlib.Path(export_model_dir) / file_name).resolve(),
                        env=env.unwrapped,
                        actor=actor,
                        normalizer=normalizer,
                    )
                except Exception as e:
                    msg = "Failed to export ONNX file."
                    omni.log.error(msg)
                    raise RuntimeError(msg) from e

        # ====================================================
        # Close the environment before terminating the process
        # ====================================================

        if env is not None:
            print("[INFO][OnnxEnvExporter.worker] Closing the exported environment.")
            env.close()
            env = None

        print("[INFO][OnnxEnvExporter.worker] Returning cleanly from the exporter process.")
        return

    @staticmethod
    def adapt_env_cfg_for_exporting(
        env_cfg: ManagerBasedRLEnvCfg,
    ) -> ManagerBasedRLEnvCfg:
        """
        Adapt the training environment configuration for ONNX export.

        Args:
            env_cfg: The configuration of the training environment.

        Returns:
            The adapted configuration for the exported environment.
        """

        # Import privately to prevent circular import errors.
        # import rai.humanoid.utils.cfgs

        # Operate on a deep copy.
        exported_env_cfg = copy.deepcopy(env_cfg)

        # Set environment to be unvectorized.
        exported_env_cfg.scene.num_envs = 1

        # Disable GUIs and recorders.
        exported_env_cfg.ui_window_class_type = None
        # exported_env_cfg.recorders = rai.humanoid.utils.cfgs.DisabledCfg()

        # Disable all randomization events.
        # exported_env_cfg.events = rai.humanoid.utils.cfgs.DisabledCfg()

        # Export on cpu.
        exported_env_cfg.sim.device = "cpu"

        def remove_noise_from_observation_group(
            obs_group_cfg: ObservationGroupCfg,
        ) -> None:
            for field in dataclasses.fields(obs_group_cfg):
                term_cfg = getattr(obs_group_cfg, field.name)
                if isinstance(
                    term_cfg,
                    isaaclab.managers.manager_term_cfg.ObservationTermCfg,
                ):
                    term_cfg.noise = None

        # Disable any observation noise.
        for obs_group in (
            getattr(exported_env_cfg.observations, field.name)
            for field in dataclasses.fields(exported_env_cfg.observations)
            if isinstance(getattr(exported_env_cfg.observations, field.name), ObservationGroupCfg)
        ):
            remove_noise_from_observation_group(obs_group_cfg=obs_group)

        return exported_env_cfg

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        actor: torch.nn.Module,
        normalizer: torch.nn.Module,
        export_model_dir: str,
        file_name: str = "policy.onnx",
        env_cfg: dict[str, Any] | None = None,
        args_cli: Namespace | None = None,
    ) -> None:
        """
        Callback function for exporting policies to ONNX during training.

        This function is designed to be used as a callback in the RSL-RL training loop.

        Args:
            env: The environment instance from the training loop.
            actor: The actor nn model from the training loop.
            normalizer: The empirical normalizer from the training loop.
            export_model_dir: The directory where the ONNX model should be saved.
            file_name: The base name for the ONNX file.
            env_cfg: The environment configuration dictionary.
            args_cli: The command-line arguments namespace.

        Note:
            By default, the iteration number is included in the ONNX file name if the training loop populates
            `env_cfg["deployment_metadata"]["model_name"]`.
        """

        # Extract the iteration number.
        # This is a bit hacky but better approaches require modifications to rsl_rl.
        def extract_iteration_number(env_cfg: dict[str, Any]) -> int:
            try:
                model_name = env_cfg["deployment_metadata"]["model_name"]
                return int("".join(filter(str.isdigit, model_name)))
            except KeyError as e:
                msg = "Failed to extract iteration number from `env_cfg`."
                omni.log.error(msg)
                raise RuntimeError(msg) from e

        try:
            iteration_number = extract_iteration_number(env_cfg) if env_cfg is not None else env.iteration
            omni.log.warn(f"[OnnxEnvExporter.__call__] Triggering ONNX export at iteration={iteration_number}.")
        except Exception:
            iteration_number = None
            omni.log.warn("[OnnxEnvExporter.__call__] Failed to extract iteration number.")

        # This logic checks whether the cached property has been already populated or not.
        process_initialized = "process_data" in self.__dict__

        # Create the configuration of the exported environment.
        exported_env_cfg = OnnxEnvExporter.adapt_env_cfg_for_exporting(env_cfg=env.cfg)

        # In the first run, we save the configuration used to build the exported environment.
        # This is later used to notify (not yet react) to possible configuration changes.
        if not process_initialized:
            self.env_cfg = copy.deepcopy(exported_env_cfg)

        # Compare the dictionaries and generate a diff dictionary.
        # out_dict = rai.core.utils.dict.compare_dicts(
        #     dict_a=self.env_cfg.to_dict(),
        #     dict_b=exported_env_cfg.to_dict(),
        # )

        # if len(out_dict["diff"]) > 0:
        #     msg = "The configuration of the training environment changed. "
        #     msg += "This is not currently supported by the ONNX exporter process.\n"
        #     msg += f"{out_dict}"
        #     omni.log.warn(msg)
        out_dict = {"diff": []}

        # Get the exporter process and the queue to exchange data.
        # They are created lazily at their first access.
        _, queue = self.process_data

        # Just to make sure that assumptions are always met.
        assert isinstance(actor, torch.nn.Module), type(actor)
        assert isinstance(normalizer, torch.nn.Module), type(normalizer)

        # Populate a tuple with data to send to the exporter process.
        with torch.inference_mode():
            # A deepcopy is performed to allow asynchronous processing without being affected
            # by the training progress.
            # This ensures that the data in the queue always reflects the correct checkpoint,
            # even if training continues in the meantime.
            # This is especially important for the first checkpoint, which is triggered at iteration 0.
            # Since creating the exported environment takes time, the corresponding ONNX file might
            # be generated only after training has already moved on to later iterations.
            onnx_file_path = f"policy_{iteration_number:07d}.onnx" if iteration_number is not None else file_name
            onnx_checkpoint_data = copy.deepcopy((
                export_model_dir,
                onnx_file_path,
                args_cli,
                (actor, normalizer),
            ))

        # Send the data to the exporter process.
        omni.log.warn(f"[OnnxEnvExporter.__call__] Sending iteration={iteration_number} data to worker process.")
        queue.put(onnx_checkpoint_data)

        # ===========================
        # Upload ONNX artifact to W&B
        # ===========================

        # Early exit if no W&B run is active.
        if wandb.run is None:
            return

        # The worker process should create this file.
        expected_onnx_file_path = (pathlib.Path(export_model_dir) / onnx_file_path).resolve()

        # Wait for the ONNX file to be created by the worker process.
        # If polling causes issues, consider implementing a notification queue
        # from the worker to the caller to signal when the ONNX file is ready.
        try:
            with OnnxEnvExporter.wait_for_file(
                path=expected_onnx_file_path, polling=5, timeout=60 * 2
            ) as found_onnx_file_path:

                artifact_file_name = "policy.onnx"
                run_name = re.sub(r"[^A-Za-z0-9._-]+", "_", wandb.run.name)

                with OnnxEnvExporter.temporary_file_copy(
                    src=found_onnx_file_path,
                    file_name=artifact_file_name,
                ) as artifact:
                    # Create the artifact.
                    onnx_artifact = wandb.Artifact(
                        name=f"{run_name}-{artifact_file_name}",
                        type="model",
                        description="ONNX policy checkpoint",
                    )
                    onnx_artifact.add_file(str(artifact))

                    # Log the artifact with appropriate tags.
                    # TODO: Consider adding a 'best' tag to indicate the best-performing checkpoint.
                    # This requires receiving information about the best checkpoint from the PPO runner,
                    # such as tracking the highest reward or best evaluation metric during training.
                    tags = ["latest"]
                    tags += [f"iter={iteration_number}"] if iteration_number is not None else []
                    wandb.log_artifact(onnx_artifact, aliases=tags)

        except TimeoutError as e:
            msg = "{} Failed to find ONNX checkpoint '{}' for upload, skipping upload: {}"
            omni.log.warn(msg.format("[OnnxEnvExporter.__call__]", expected_onnx_file_path, e))
            return

        except Exception as e:
            msg = "{} Failed to upload ONNX checkpoint '{}' to W&B: {}"
            omni.log.error(msg.format("[OnnxEnvExporter.__call__]", expected_onnx_file_path, e))
            return

    @staticmethod
    @contextlib.contextmanager
    def wait_for_file(
        path: pathlib.Path,
        polling: float,
        timeout: float | None = None,
    ) -> Generator[pathlib.Path, None, None]:
        """
        Context manager that waits for a file to be created.

        Args:
            path: The path to the file to wait for.
            polling: The polling interval in seconds.
            timeout: The maximum time to wait in seconds. If None, waits indefinitely.

        Yields:
            The path to the file once it is created.

        Raises:
            TimeoutError: If the timeout is reached before the file is created.
        """

        start_time = time.time()

        while not path.is_file():
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout reached while waiting for file '{path}'")

            time.sleep(polling)

        yield path

    @contextlib.contextmanager
    def temporary_file_copy(src: pathlib.Path, file_name: str | None = None) -> Generator[pathlib.Path, None, None]:
        """
        Context manager that creates a temporary copy of a file.

        Args:
            src: The source file path.
            file_name: The name of the temporary file. If None, uses the name of the source file.

        Yields:
            The path to the temporary copied file.
        """

        file_name = file_name if file_name is not None else src.name

        with tempfile.TemporaryDirectory() as tmp_dir:
            dst = pathlib.Path(tmp_dir) / file_name
            shutil.copy2(src, dst)
            yield dst

    def stop(self) -> None:
        """
        Stop gracefully the exporter process and cleanup its resources.
        """

        # This is the pythonic way to check if a cached property has already been populated.
        if "process_data" not in self.__dict__:
            print("[INFO][OnnxEnvExporter.stop] Nothing to cleanup.")
            return

        # Get the process and the queue.
        process, queue = self.process_data

        # If the process is not alive, terminate the queue and return.
        if not process.is_alive():
            print("[INFO][OnnxEnvExporter.stop] Process is not alive, terminating the queue.")
            queue.close()
            queue.join_thread()
            _ = self.__dict__.pop("process_data")
            return

        # Empty the queue before terminating the process.
        with contextlib.suppress(queue_python.Empty):
            while True:
                _ = queue.get_nowait()

        # Send SIGTERM if it gets stuck.
        if process.is_alive():
            print("[INFO][OnnxEnvExporter.stop] ONNX exporter process still alive, sending SIGTERM.")
            process.terminate()
            process.join(timeout=5)

        # Send SIGKILL if it is still stuck.
        if process.is_alive():
            print("[INFO][OnnxEnvExporter.stop] ONNX exporter process still alive, sending SIGKILL.")
            process.kill()
            process.join(timeout=5)

        # We've done the best we could, notify the user that the process is still alive.
        if process.is_alive():
            omni.log.error("Failed to stop exporter process.")

        # Terminate the queue and return.
        queue.close()
        queue.join_thread()

        # Remove the terminated resources from the cached property.
        _ = self.__dict__.pop("process_data")
