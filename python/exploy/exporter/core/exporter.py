# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import datetime
import json
import os
from enum import Enum

import onnx
import torch

from exploy.exporter.core.exportable_environment import ExportableEnvironment
from exploy.exporter.core.utils.onnx import construct_decimation_wrapper, find_dangling_inputs
from exploy.exporter.core.utils.paths import get_exploy_version, prepare_onnx_paths


def export_environment_as_onnx(
    env: ExportableEnvironment,
    actor: torch.nn.Module,
    path: str,
    filename: str = "policy.onnx",
    model_source: dict | None = None,
    verbose: bool = False,
    opset_version: int = 20,
    ir_version: int = 11,
):
    """Export policy into a Torch ONNX file.

    Args:
        env: The environment to be exported.
        actor: The actor torch module.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        model_source: Information about the policy's origin (e.g., wandb, local file, etc.), added to the ONNX metadata.
        verbose: Whether to print the model summary. Defaults to False.
        opset_version: Version of the operator specification referenced by the ONNX graph. Needs to be compatible with ONNX Runtime in deployment environment, check https://onnxruntime.ai/docs/reference/compatibility.html
        ir_version: Version of the intermediate representation specifications. Needs to be compatible with ONNX Runtime in deployment environment, check https://onnxruntime.ai/docs/reference/compatibility.html
    """
    if model_source is None:
        model_source = {}
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = OnnxEnvironmentExporter(
        env=env,
        actor=actor,
        verbose=verbose,
        opset_version=opset_version,
        ir_version=ir_version,
    )

    policy_exporter.export(
        onnx_path=path,
        onnx_file_name=filename,
        model_source=model_source,
    )


# The export modes supported by the ONNX exporter.
class ExportMode(Enum):
    # Default mode exports the full graph corresponding to one environment step including
    # processing of actions.
    Default = 0
    # ProcessActions mode exports only the subgraph from actions to outputs corresponding to
    # a substep of the environment at sim dt where actions are applied and the scene updated.
    ProcessActions = 1


class OnnxEnvironmentExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file using the environment's managers."""

    def __init__(
        self,
        env: ExportableEnvironment,
        actor: torch.nn.Module,
        opset_version: int,
        ir_version: int,
        verbose: bool = False,
    ):
        super().__init__()
        self._env: ExportableEnvironment = env

        self.verbose = verbose
        self.actor = actor

        self.export_mode = ExportMode.Default

        self._opset_version = opset_version
        self._ir_version = ir_version

        # Track registered modules to ensure idempotent behavior
        self._registered_modules: set[int] = set()
        self._module_registration_counter: int = 0

    def forward(
        self,
        input_data: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Use the robot's state to compute policy actions, joint position targets, and policy
            observations, and outputs that support history.

        Args:
            input_data: A dictionary containing all input tensors required for the forward pass.
                The expected keys and shapes of the tensors depend on the environment's context
                manager and the policy's computational graph.

        Notes:
            - Dictionary inputs are flattened by the torch ONNX exporter implementation.
            - Only inputs that are part of the computational graph will be required when using the
              resulting ONNX file for inference. For example, if `pos_base_in_w` is not used by
              any of the observation functions, it will not be a required input. This can be
              verified by querying the ONNX input names when using the ONNX runtime framework.

        Returns:
            tuple[torch.Tensor, ...]: Environment outputs.
            A tuple of desired joint positions (i.e., processed actions), actions (i.e., unprocessed
            actions), memory (containing the previous actions for example).
        """
        # Set data handlers from source inputs.
        self._env.context_manager().write_connections()

        # Compute.
        with torch.inference_mode():
            observations = self._env.empty_actor_observations()
            actions = self._env.empty_actions()

            # Inference: compute actions.
            match self.export_mode:
                case ExportMode.Default:
                    # Update commands.
                    for command_update in self._env.command_updates:
                        command_update()

                    # Compute observations.
                    observations = self._env.compute_observations()

                    # Compute actions.
                    actions = self.actor(observations)

                    # Process the actions.
                    self._env.process_actions(actions)

                case ExportMode.ProcessActions:
                    pass

            self._env.apply_actions()

            output_data = list(self._env.context_manager().get_outputs().values())

        return (
            actions,
            observations,
            *output_data,
        )

    def register_modules(self):
        """Register all modules from the environment's context manager.

        This method iterates over all modules in the environment's context manager
        and registers them using sequential names. This ensures that all relevant modules are
        included in the ONNX export, allowing the exported model to function correctly when loaded
        in an ONNX runtime environment.

        Calling this method multiple times will not re-register already registered modules. The
        modules' names are based on insertion order.
        """
        for module in self._env.context_manager().modules:
            module_id = id(module)
            # Skip if this module is already registered.
            if module_id in self._registered_modules:
                continue

            # Register this module.
            module_name = f"exporter_registered_module_{self._module_registration_counter}"
            self.register_module(module_name, module)

            # Track the registered module.
            self._registered_modules.add(module_id)
            self._module_registration_counter += 1

    def export(
        self,
        onnx_path: str,
        onnx_file_name: str,
        model_source: dict,
    ):
        """Export to ONNX.

        Args:
            onnx_path: The path to the folder that will contain the ONNX file.
            onnx_file_name: The name (including the `ONNX` extension) of the exported file.
            model_source: Information about the policy's origin (e.g., wandb, local file, etc.),
                added to the ONNX metadata.
        """
        self.eval()

        self.register_modules()

        # Get input values and names.
        input_values = [self._env.context_manager().get_inputs()]
        input_names = self._env.context_manager().get_input_names()

        # Passing an empty dictionary as the last input is required to tell ONNX to
        # interpret the previous dictionary inputs as a non-keyword argument.
        # From the torch.onnx source:
        #     "If a dictionary is the last element of the args tuple, it will be interpreted as
        #     containing named arguments. In order to pass a dict as the last non-keyword arg,
        #     provide an empty dict as the last element of the args tuple."
        input_values.append({})
        input_values = tuple(input_values)

        output_names = ["actions", "obs"]
        output_names += self._env.context_manager().get_output_names()

        # Prepare all export paths
        export_paths = prepare_onnx_paths(
            output_dir=onnx_path,
            filename=onnx_file_name,
            debug_suffixes=["default", "process_actions"],
        )
        for mode, file_path in (
            (ExportMode.ProcessActions, export_paths.get_debug_path("process_actions")),
            (ExportMode.Default, export_paths.get_debug_path("default")),
        ):
            self.export_mode = mode
            self._env.prepare_export()
            torch.onnx.export(
                self,
                input_values,
                str(file_path),
                export_params=True,
                opset_version=self._opset_version,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                # constant folding requires that all tensors are on the same device. as constants
                # are on cpu, we disable it to allow exporting of models on cuda. constant folding
                # optimization will be done in the ONNX runtime when we load the model for deployment.
                do_constant_folding=False,
                dynamo=False,
            )

        wrapper_model = construct_decimation_wrapper(
            model_a=onnx.load(str(export_paths.get_debug_path("default"))),
            model_b=onnx.load(str(export_paths.get_debug_path("process_actions"))),
            name_a="default",
            name_b="process_actions",
            decimation=self._env.decimation,
            opset_version=self._opset_version,
            ir_version=self._ir_version,
        )
        onnx.save(wrapper_model, str(export_paths.main))

        # Inputs are wired into the graph by object identity: an input whose tensor is never read
        # during tracing is silently dropped by torch.onnx.export. Report them so that inputs
        # meant to be graph boundaries (e.g. derived quantities) do not vanish unnoticed.
        dangling_inputs = find_dangling_inputs(wrapper_model, input_names)
        if dangling_inputs:
            print(
                "INFO [exploy.exporter]: The following registered inputs are not part of the "
                "exported ONNX graph and were dropped (their tensors were never read during "
                "tracing):\n" + "\n".join(f"  - {name}" for name in dangling_inputs)
            )

        # Load the ONNX model to add metadata to it.
        onnx_model = onnx.load(str(export_paths.main))

        # Model data, including wandb link (if available).
        meta = onnx_model.metadata_props.add()
        meta.key = "model_source"
        meta.value = json.dumps(model_source)

        # Date exported.
        meta = onnx_model.metadata_props.add()
        meta.key = "date_exported (YYMMDD.HHMMSS)"
        meta.value = str(datetime.datetime.now().strftime("%y%m%d.%H%M%S"))

        # Exploy version.
        meta = onnx_model.metadata_props.add()
        meta.key = "exploy_version"
        meta.value = json.dumps(get_exploy_version())

        # Environment metadata.
        for key, value in self._env.metadata().items():
            meta = onnx_model.metadata_props.add()
            meta.key = key
            meta.value = value

        # Context manager metadata.
        for key, value in self._env.context_manager().metadata.items():
            meta = onnx_model.metadata_props.add()
            meta.key = key
            meta.value = json.dumps(value)

        # Save the modified model.
        onnx.save(onnx_model, str(export_paths.main))

        # Copy metadata to decimation model.
        onnx_default_model = onnx.load(str(export_paths.get_debug_path("default")))
        onnx_default_model.metadata_props.extend(onnx_model.metadata_props)
        onnx.save(onnx_default_model, str(export_paths.get_debug_path("default")))
