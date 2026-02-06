# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import copy
import datetime
import json
import os
from enum import Enum

import onnx
import torch

from .exportable_environment import ExportableEnvironment
from .utils.onnx import construct_decimation_wrapper
from .utils.paths import prepare_onnx_paths


def export_environment_as_onnx(
    env: ExportableEnvironment,
    actor: torch.nn.Module,
    path: str,
    normalizer: torch.nn.Module | None = None,
    filename: str = "policy.onnx",
    model_source: dict | None = None,
    verbose: bool = False,
    export_device: str = "cpu",
):
    """Export policy into a Torch ONNX file.

    Args:
        env: The environment to be exported.
        actor_critic: The actor-critic torch module.
        path: The path to the saving directory.
        normalizer: The empirical normalizer module. If None, Identity is used.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        model_source: Information about the policy's origin (e.g., wandb, local file, etc.), added to the ONNX metadata.
        verbose: Whether to print the model summary. Defaults to False.
    """
    if model_source is None:
        model_source = {}
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = OnnxEnvironmentExporter(
        env=env,
        actor=actor,
        normalizer=normalizer,
        verbose=verbose,
    )

    policy_exporter.export(
        onnx_path=path,
        onnx_file_name=filename,
        model_source=model_source,
        export_device=export_device,
    )


def are_values_on_device(
    names: list[str],
    values: tuple,
    expected_device: str = "cpu",
    verbose: bool = True,
) -> bool:
    """Check that all input values are on the expected device."""
    correct_device = True
    input_values_debug = []
    for v in values:
        if isinstance(v, torch.Tensor):
            input_values_debug.append(v)
        elif isinstance(v, dict):
            for val in v.values():
                input_values_debug.append(val)
    for name, val in zip(names, input_values_debug, strict=False):
        if val.device.type != expected_device:
            if verbose:
                print(
                    f"Input named {name} is not on {expected_device}. Got device: {val.device.type}"
                )
            correct_device = False
    return correct_device


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
        normalizer: torch.nn.Module | None = None,
        verbose: bool = False,
    ):
        super().__init__()
        self._env: ExportableEnvironment = env

        self.verbose = verbose
        self.actor = copy.deepcopy(actor)

        self.export_mode = ExportMode.Default

        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        # Compatible versions with onnxruntime used in control (1.17)
        # See https://onnxruntime.ai/docs/reference/compatibility.html
        self._opset_version = 20
        self._ir_version = 9

    def forward(
        self,
        input_data: dict[str, torch.Tensor],
    ):
        """Use the robot's state to compute policy actions, joint position targets, and policy observations, and outputs
        that support history.

        This method sets the environment's data sources (e.g., the articulation data and the IMU sensor data) such that
        computing this method's outputs results in embedding the task's observation and action managers to be part of
        the computational graph. This implementation's design is discussed in this design doc:
        https://docs.google.com/document/d/1mOz2VPSpYvOUTK6sjLT_JNLZAldyzEnNievXJTpQvPs/

        Notes:
            - Dictionary inputs are flattened by the torch ONNX exporter implementation.
            - As discussed in the design doc above, only inputs that are part of the computational graph will be
              required when using the resulting ONNX file for inference.
              For example, if `pos_base_in_w` is not used by any of the observation functions, it will not be a required
              input. This can be verified by querying the ONNX input names when using the ONNX runtime framework.

        Assumptions:
            - Processed actions are joint targets, and all joints are actuated.

        Args:
            imu_data: A dictionary of IMU poses and angular velocities, for each available IMU.
            sensor_data: A dictionary of sensor values. IMU data is handled separately.
            command_data: A dictionary of command values.
            art_data: A dictionary of articulation data values.
            rigid_object_data: A dictionary of rigid-body objects in the scene.
            memory_data: A dictionary of inputs used to support history.

        Returns:
            joint_targets, actions, output_memory:
            A tuple of desired joint positions (i.e., processed actions), actions (i.e., unprocessed actions),
            memory (containing the previous actions for example).
        """
        # Set data handlers from source inputs.
        self._env.context_manager().write_connections()

        # Compute.
        with torch.no_grad():
            observations = self._env.empty_actor_observations()
            actions = self._env.empty_actions()

            # Inference: compute actions.
            match self.export_mode:
                case ExportMode.Default:
                    # TODO: Fix updating commands.
                    # Update required commands.
                    # Note: we explicitly only call the `_update_command` method
                    #       to enable the computational graph associated with commands.
                    #       Calling the `compute` method instead would trigger an error
                    #       due to aten::uniform not being supported by onnx.
                    # for command_term in self._data_handler.command_dh.command_terms_to_update:
                    #     command_term._update_command()

                    # Compute observations.
                    observations = self._env.compute_observations(device=self._export_device)

                    # Compute actions.
                    actions = self.actor(self.normalizer(observations))

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

    def export(
        self,
        onnx_path: str,
        onnx_file_name: str,
        model_source: dict,
        export_device: str = "cpu",
    ):
        """Export to ONNX.

        Args:
            path: The path to the folder that will contain the ONNX file.
            filename: The name (including the `ONNX` extension) of the exported file.
            model_source: Information about the policy's origin (e.g., wandb, local file, etc.), added to the ONNX metadata.
        """
        self.to(device=export_device)
        self.eval()

        # convert_pretrained_networks_in_observation(exporter=self)

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

        assert are_values_on_device(
            names=input_names,
            values=input_values,
            expected_device=export_device,
            verbose=True,
        )

        # Prepare all export paths
        export_paths = prepare_onnx_paths(
            output_dir=onnx_path,
            filename=onnx_file_name,
            debug_suffixes=["default", "process_actions"],
        )

        self._export_device = export_device

        for mode, file_path in (
            (ExportMode.ProcessActions, export_paths.get_debug_path("process_actions")),
            (ExportMode.Default, export_paths.get_debug_path("default")),
        ):
            self.export_mode = mode
            torch.onnx.export(
                self,
                input_values,
                str(file_path),
                export_params=True,
                opset_version=self._opset_version,
                verbose=self.verbose,
                input_names=input_names,
                output_names=output_names,
            )

        wrapper_model = construct_decimation_wrapper(
            model_a=onnx.load(str(export_paths.get_debug_path("default"))),
            model_b=onnx.load(str(export_paths.get_debug_path("process_actions"))),
            decimation=self._env.decimation,
            opset_version=self._opset_version,
            ir_version=self._ir_version,
        )
        onnx.save(wrapper_model, str(export_paths.main))

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
