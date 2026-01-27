# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import copy
import json
import numpy as np
import os
import pathlib
import torch
from datetime import datetime
from enum import Enum

import onnx
import onnxruntime as ort
import wandb

from exporter import ExportEnvIdsType
from exporter.articulation_data import ArticulationDataSource
from exporter.data_handler_manager import DataHandlerManager
from exporter.rigid_object_data import RigidObjectDataSource
from exporter.term_resetters import TermResetter

from isaaclab.envs import ManagerBasedRLEnv

"""Helper functions and classes to export a policy to ONNX embedding the environment managers into the ONNX
computational graph.

This implementation is used when calling the play script with `--convert_environment_onnx`.
"""


def _copy_value_info(value_info: onnx.ValueInfoProto) -> onnx.ValueInfoProto:
    """
    Create a copy of an ONNX ValueInfoProto with the same name, type, and shape.
    Args:
        value_info: The ValueInfoProto to copy.
    Returns:
        A copy of the input ValueInfoProto.
    """
    return onnx.helper.make_tensor_value_info(
        value_info.name,
        value_info.type.tensor_type.elem_type,
        [d.dim_value if d.dim_value > 0 else None for d in value_info.type.tensor_type.shape.dim],
    )


def construct_decimation_wrapper(
    model_a: onnx.ModelProto,
    model_b: onnx.ModelProto,
    decimation: int,
    opset_version: int,
    ir_version: int,
) -> onnx.ModelProto:
    """
    Wraps two ONNX models with decimation logic. Executes model_a if (step_count % decimation == 0), otherwise model_b.
    Args:
        model_a: ONNX submodel for decimation event.
        model_b: ONNX submodel for other steps.
        decimation: Decimation factor.
    Returns:
        An ONNX ModelProto with fixed periodic conditional branching.
    """
    time_input = onnx.helper.make_tensor_value_info("step_count", onnx.TensorProto.INT32, [])
    submodel_inputs = [_copy_value_info(i) for i in model_a.graph.input]
    outputs = [_copy_value_info(o) for o in model_a.graph.output]

    decimation_const = onnx.helper.make_tensor("decimation", onnx.TensorProto.INT32, (), [decimation])
    zero_const = onnx.helper.make_tensor("zero", onnx.TensorProto.INT32, (), [0])
    mod_node = onnx.helper.make_node("Mod", ["step_count", "decimation"], ["is_event"])
    eq_node = onnx.helper.make_node("Equal", ["is_event", "zero"], ["cond"])

    # Remove submodel inputs (will be passed by parent graph)
    for g in (model_a.graph, model_b.graph):
        del g.input[:]

    # Branching node
    if_node = onnx.helper.make_node(
        "If",
        inputs=["cond"],
        outputs=[o.name for o in outputs],
        then_branch=model_a.graph,
        else_branch=model_b.graph,
    )

    parent_graph = onnx.helper.make_graph(
        nodes=[mod_node, eq_node, if_node],
        name="decimation_wrapper",
        inputs=[time_input] + submodel_inputs,
        outputs=outputs,
        initializer=[decimation_const, zero_const],
    )

    model = onnx.helper.make_model(
        parent_graph,
        producer_name="construct_decimation_wrapper",
        opset_imports=[onnx.helper.make_operatorsetid("", opset_version)],
        ir_version=ir_version,
    )
    onnx.checker.check_model(model)
    return model


def get_onnx_input_values(
    dh_manager: DataHandlerManager, to_numpy: bool = False
) -> list[dict[str, torch.Tensor | np.ndarray]]:
    """Get ONNX input values from data handler manager.

    Args:
        dh_manager: The data handler manager.
        to_numpy: Whether to return inputs as numpy arrays.

    Returns:
        onnx_inputs_torch: A list of ONNX input tensors.
    """
    return [
        {key: value.cpu().numpy() if to_numpy else value for key, value in dh.data.items()}
        for dh in [
            dh_manager.imu_dh,
            dh_manager.sensor_dh,
            dh_manager.command_dh,
            dh_manager.art_dh,
            dh_manager.rigid_object_dh,
            dh_manager.memory_dh,
        ]
    ]


def get_onnx_inputs(dh_manager: DataHandlerManager, to_numpy: bool) -> dict[str, torch.Tensor | np.ndarray]:
    """Get ONNX inputs from data handler manager.

    Args:
        dh_manager: The data handler manager.
        to_numpy: Whether to return inputs as numpy arrays.

    Returns:
        onnx_inputs: A dictionary of ONNX input tensors.
    """
    return {
        key: value
        for handler_data in get_onnx_input_values(dh_manager, to_numpy=to_numpy)
        for key, value in handler_data.items()
    }


def export_environment_as_onnx(
    env: ManagerBasedRLEnv,
    actor: torch.nn.Module,
    path: str,
    export_env_ids: ExportEnvIdsType,
    normalizer: torch.nn.Module | None = None,
    filename: str = "policy.onnx",
    model_source: dict = {},
    verbose: bool = False,
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
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxEnvironmentExporter(
        env=env,
        export_env_ids=export_env_ids,
        actor=actor,
        normalizer=normalizer,
        verbose=verbose,
    )
    policy_exporter.export(
        onnx_path=path,
        onnx_file_name=filename,
        model_source=model_source,
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
    for name, val in zip(names, input_values_debug):
        if val.device.type != expected_device:
            if verbose:
                print(f"Input named {name} is not on {expected_device}. Got device: {val.device.type}")
            correct_device = False
    return correct_device


def convert_pretrained_networks_in_observation(
    exporter: "_OnnxEnvironmentExporter",
) -> None:
    """Convert pretrained perception networks to be used during ONNX export.

    This function searches through the observation terms in the environment's observation manager.
    If a term is of type 'HeightScanObservationHandler' and uses an encoder, it loads the pretrained
    network from wandb and sets it in the observation term.

    Args:
        env: The environment containing the observation manager.
        exporter: The ONNX exporter instance where the networks will be set.
    """
    # import rai.core.mdp.observations
    # from rai.core.mdp.actions import VelocityPreTrainedPolicyAction

    # Define a function to replace pretrained models.
    def replace_pretrained_models(exporter, manager):
        for active_term in manager.active_terms.keys():
            for i, obs in enumerate(manager._group_obs_term_cfgs[active_term]):
                # if (
                #     hasattr(obs, "func")
                #     and isinstance(obs.func, rai.core.mdp.observations.HeightScanObservationHandler)
                #     and obs.params["use_encoder"]
                # ):
                if False:
                    if obs.params["wandb_checkpoint"]:
                        api = wandb.Api()
                        wandb_path = manager._group_obs_term_cfgs[active_term][i].params["file_path"]
                        wandb_path_parts = wandb_path.split("_")
                        wandb_path_eager = "_".join(wandb_path_parts[:-1] + ["eager_" + wandb_path_parts[-1]])
                        artifact = api.artifact(wandb_path_eager, type="model")
                        model_dir = artifact.download()
                        model_path = f"{model_dir}/model.pt"
                    else:
                        # observation checks that file ends with .pt, and xx_eager.pt is how DVL saves eager models
                        model_path = obs.params["file_path"].replace(".pt", "_eager.pt")
                    # In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`
                    model = torch.load(model_path, weights_only=False).to(exporter._export_device).eval()
                    for p in model.parameters():
                        # todo:  Why do the weights in CNN layers get exporter with `required_gras=True`?
                        #        We should use the torchscript convert functions to turn this into a torch.nn.Module
                        #        and handle weights/gradients there.
                        p.requires_grad = False
                    exporter.height_network_eager = model
                    # When we trace through an object that uses additional "external" torch modules, ONNX currently
                    # requires that these objects are set to the traced object to allow tracing.
                    manager._group_obs_term_cfgs[active_term][i].func.height_network = exporter.height_network_eager
                    break

    replace_pretrained_models(exporter=exporter, manager=exporter.env.observation_manager)

    # Check for pretrained models nested in the environment managers.
    for active_term in exporter.env.action_manager.active_terms:
        action_term = exporter.env.action_manager.get_term(active_term)
        # if isinstance(action_term, VelocityPreTrainedPolicyAction):
        #     replace_pretrained_models(exporter=exporter, manager=action_term.low_level_obs_manager)
        pass


def get_action_for_process(env: ManagerBasedRLEnv, policy_actions: torch.Tensor) -> torch.Tensor:
    """Get the action tensor.

    This function checks each action term and returns the action tensor. This allows to support networks which have (many) sub networks and need to combine actions into a single action tensor.
    """
    action = None
    for action_term_name in env.action_manager.active_terms:
        action_term = env.action_manager.get_term(action_term_name)
        if type(action_term).__name__ == "TaskSpaceMLPAction":
            action = action_term.get_low_level_processed_policy_action()

    if action is None:
        action = policy_actions

    return action


class SessionWrapper:
    """Manage a torch Module and its associated ONNX inference session."""

    def __init__(
        self,
        onnx_folder: pathlib.Path,
        onnx_file_name: str,
        policy: torch.nn.Module | None = None,
        optimize: bool = True,
    ):
        """Construct a `SessionWrapper` to use it for policy inference.

        Args:
            onnx_folder: The folder containing an ONNX file to load.
            onnx_file_name: The name of the ONNX file contained in `ONNX_folder`.
            policy: A `torch.nn.Module` representing the actor.
            optimize: If true, optimize the ONNX graph, save it to file, and use it for inference.
        """
        # Check if the file name includes the extension.
        expected_extension = ".onnx"
        onnx_file_name = pathlib.Path(onnx_file_name)
        if onnx_file_name.suffix != expected_extension:
            onnx_file_name = onnx_file_name.with_suffix(expected_extension)

        sess_options = None

        # If required, optimize the computational graph.
        if optimize:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            # Setting `optimized_model_filepath` tells ONNX to store the optimized graph to a file.
            # This additional optimized ONNX file is useful for inspection with Netron (see https://github.com/lutzroeder/netron),
            # since the computational graph is optimized and cleaned up.
            # The optimization of the computational graph depends on additional features in ONNX and version control of
            # the ONNX dependencies, which are not correctly managed in control. For this reason, this file
            # is to be used, at the moment, only for debugging.
            onnx_debug_path = onnx_folder / "debug"
            onnx_debug_path.mkdir(parents=True, exist_ok=True)
            sess_options.optimized_model_filepath = str(
                onnx_debug_path / f"{onnx_file_name.stem}_optimized{expected_extension}"
            )

        self._onnx_file_path = onnx_folder / onnx_file_name
        session = ort.InferenceSession(
            str(self._onnx_file_path),
            sess_options=sess_options,
        )

        self.session = session
        self.input_names = [inp.name for inp in session.get_inputs()]
        self.output_names = [val.name for val in session.get_outputs()]
        self._policy = policy
        self.metadata = session.get_modelmeta()

        self._results = None

    @property
    def onnx_file_path(self) -> pathlib.Path:
        return self._onnx_file_path

    def __call__(self, **kwargs):
        in_kwargs = {name: kwargs[name] for name in self.input_names}
        self._results = self.session.run(self.output_names, in_kwargs)
        return self._results

    def get_torch_model(self) -> torch.nn.Module:
        return self._policy

    def get_output_value(self, output_name: str):
        assert (
            output_name in self.output_names
        ), f"Output '{output_name}' not found in expected outputs: {self.output_names}"
        return self._results[self.output_names.index(output_name)]

    def reset(self):
        """Reset the internal results to zeros to avoid stale data at environment reset."""
        self._results = [np.zeros_like(output) for output in self._results]


# The export modes supported by the ONNX exporter.
class ExportMode(Enum):
    # Default mode exports the full graph corresponding to one environment step including
    # processing of actions.
    Default = 0
    # ProcessActions mode exports only the subgraph from actions to outputs corresponding to
    # a substep of the environment at sim dt where actions are applied and the scene updated.
    ProcessActions = 1


class _OnnxEnvironmentExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file using the environment's managers."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        export_env_ids: ExportEnvIdsType,
        actor: torch.nn.Module,
        normalizer: torch.nn.Module | None = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.env: ManagerBasedRLEnv = env
        self.export_env_ids = export_env_ids

        # Choose the device on which this environment will be exported.
        self._export_device = "cpu"

        # The group name of the policy observations in the environment's observation manager.
        self._policy_group_name = "policy"

        # The name of the articulation in the environment's scene.
        self._articulation_name = "robot"

        # Check that the expected articulation exists.
        assert self._articulation_name in self.env.scene.articulations.keys()

        self.verbose = verbose
        self.actor = copy.deepcopy(actor)

        self.export_mode = ExportMode.Default

        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        # Check that the observation manager has a policy group.
        if self._policy_group_name not in self.env.observation_manager.active_terms:
            raise RuntimeError(
                "[OnnxEnvironmentExporter] Could not find observation group named {self._policy_group_name} in"
                " observation manager. Active terms are: {self.env.observation_manager.active_terms}"
            )

        # Check if observation terms are concatenated.
        if not self.env.observation_manager.group_obs_concatenate[self._policy_group_name]:
            raise RuntimeError("[OnnxEnvironmentExporter] Observation terms must be concatenated.")

        # Check that observation noise is disabled, else it will be part of the computational graph.
        for group_cfg in self.env.observation_manager._group_obs_term_cfgs.values():
            for term_cfg in group_cfg:
                if term_cfg.noise is not None:
                    raise RuntimeError(
                        "[OnnxEnvironmentExporter] While trying to convert to ONNX, found an observation term with"
                        " noise enabled."
                        "\n[OnnxEnvironmentExporter] Hint: turn off observation noise, or use a `Play` task."
                    )

        # Keep track of manager terms that need to be substituted with
        # exporter friendly ones. The terms will be set back to their original type
        # after exporting.
        self._term_resetter = TermResetter(env=self.env)
        self._orig_art_data = self.env.scene.articulations[self._articulation_name]._data

        def reset_articulation():
            self.env.scene.articulations[self._articulation_name]._data = self._orig_art_data

        self._term_resetter.add(reset_func=reset_articulation)
        self.env.scene.articulations[self._articulation_name]._data = ArticulationDataSource(
            articulation=self.env.scene.articulations[self._articulation_name]
        )

        for rigid_object in self.env.scene.rigid_objects.values():
            orig_rigid_object_data = rigid_object._data

            def reset_rigid_object_data():
                rigid_object._data = orig_rigid_object_data

            self._term_resetter.add(reset_func=reset_rigid_object_data)
            rigid_object._data = RigidObjectDataSource(rigid_object)

        # Store the environment's data. Later used to determine ONNX input names and values.
        self._data_handler = DataHandlerManager(
            env=self.env,
            env_id=self.export_env_ids,
            policy_group_name=self._policy_group_name,
            device=self._export_device,
        )

        # Compatible versions with onnxruntime used in control (1.17)
        # See https://onnxruntime.ai/docs/reference/compatibility.html
        self._opset_version = 20
        self._ir_version = 9

    def forward(
        self,
        imu_data: dict[str, torch.Tensor],
        sensor_data: dict[str, torch.Tensor],
        command_data: dict[str, torch.Tensor],
        art_data: dict[str, torch.Tensor],
        rigid_object_data: dict[str, torch.Tensor],
        memory_data: dict[str, torch.Tensor],
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
        self._data_handler.set_to_source()

        # Compute.
        with torch.no_grad():
            # Inference: compute actions.
            match self.export_mode:
                case ExportMode.Default:
                    # Update required commands.
                    # Note: we explicitly only call the `_update_command` method
                    #       to enable the computational graph associated with commands.
                    #       Calling the `compute` method instead would trigger an error
                    #       due to aten::uniform not being supported by onnx.
                    for command_term in self._data_handler.command_dh.command_terms_to_update:
                        command_term._update_command()

                    # Compute observations.
                    self.env.obs_buf = self.env.observation_manager.compute()
                    obs_buffer = self.env.obs_buf[self._policy_group_name].to(self._export_device)

                    observations = obs_buffer.view(
                        1, self.env.observation_manager.group_obs_dim[self._policy_group_name][self.export_env_ids]
                    )

                    actions = self.actor(self.normalizer(obs_buffer))
                case ExportMode.ProcessActions:
                    # We only want the subgraph from actions to outputs in the post-process graph.
                    # We always add memory with the name "actions", it is therefore guaranteed to be present here.
                    actions = memory_data["memory.actions.in"].unsqueeze(0)
                    # We do not want the computation of the observations to be part of the post-process graph.
                    # We therefore set it to zeros.
                    observations = torch.zeros_like(self.env.obs_buf[self._policy_group_name])

            # Process the actions.
            self.env.action_manager.process_action(actions)
            self.env.action_manager.apply_action()

            # Extract joint targets.
            outputs = self._data_handler.action_dh.values()

            # Get outputs for memory.
            output_memory = self._data_handler.memory_dh.get_updated_values_as_outputs()

            # Get the action tensor. The correct action depends on which action terms are used.
            action = get_action_for_process(
                env=self.env,
                policy_actions=actions,
            )

        return (
            action,
            observations,
            *outputs,
            *output_memory,
        )

    def export(
        self,
        onnx_path: str,
        onnx_file_name: str,
        model_source: dict,
    ):
        """Export to ONNX.

        Args:
            path: The path to the folder that will contain the ONNX file.
            filename: The name (including the `ONNX` extension) of the exported file.
            model_source: Information about the policy's origin (e.g., wandb, local file, etc.), added to the ONNX metadata.
        """
        self.to(
            device=self._export_device,
        )
        self.eval()

        convert_pretrained_networks_in_observation(exporter=self)

        # Get input values and names.
        input_values = get_onnx_input_values(self._data_handler)
        input_names = [key for handler_data in input_values for key in handler_data.keys()]

        # Passing an empty dictionary as the last input is required to tell ONNX to
        # interpret the previous dictionary inputs as a non-keyword argument.
        # From the torch.onnx source:
        #     "If a dictionary is the last element of the args tuple, it will be interpreted as
        #     containing named arguments. In order to pass a dict as the last non-keyword arg,
        #     provide an empty dict as the last element of the args tuple."
        input_values.append({})
        input_values = tuple(input_values)

        output_names = ["actions", "obs"]
        output_names += self._data_handler.action_dh.names()
        output_names += self._data_handler.memory_dh.output_names()

        assert are_values_on_device(
            names=input_names,
            values=input_values,
            expected_device=self._export_device,
            verbose=True,
        )

        # Keep track of and reset each sensor's info to avoid
        # them being updated while exporting.
        sensor_info = {}
        for sensor_name, sensor in self.env.scene.sensors.items():
            sensor_info[sensor_name] = {}
            sensor_info[sensor_name]["timestamp"] = sensor._timestamp
            sensor_info[sensor_name]["timestamp_last_update"] = sensor._timestamp_last_update
            sensor_info[sensor_name]["is_outdated"] = sensor._is_outdated.clone()
            sensor._timestamp = -100.0
            sensor._timestamp_last_update = 100.0
            sensor._is_outdated[:] = False

        path = pathlib.Path(onnx_path)
        ext = ".onnx"
        file_name = pathlib.Path(onnx_file_name)
        if file_name.suffix != ext:
            file_name = file_name.with_suffix(ext)
        debug_path = path / "debug"
        debug_path.mkdir(parents=True, exist_ok=True)

        onnx_file_path_default = str(debug_path / f"{file_name.stem}_default{ext}")
        onnx_file_path_process_actions = str(debug_path / f"{file_name.stem}_process_actions{ext}")

        for mode, file_path in (
            (ExportMode.ProcessActions, onnx_file_path_process_actions),
            (ExportMode.Default, onnx_file_path_default),
        ):
            self.export_mode = mode
            torch.onnx.export(
                self,
                input_values,
                file_path,
                export_params=True,
                opset_version=self._opset_version,
                verbose=self.verbose,
                input_names=input_names,
                output_names=output_names,
            )

        wrapper_model = construct_decimation_wrapper(
            model_a=onnx.load(onnx_file_path_default),
            model_b=onnx.load(onnx_file_path_process_actions),
            decimation=self.env.cfg.decimation,
            opset_version=self._opset_version,
            ir_version=self._ir_version,
        )
        onnx_file_path = str(path / file_name)
        onnx.save(wrapper_model, onnx_file_path)

        # Set back sensor info.
        for sensor_name, sensor in self.env.scene.sensors.items():
            sensor._timestamp = sensor_info[sensor_name]["timestamp"]
            sensor._timestamp_last_update = sensor_info[sensor_name]["timestamp_last_update"]
            sensor._is_outdated[:] = sensor_info[sensor_name]["is_outdated"]

        # Set back replaced objects.
        self._term_resetter()

        # Load the ONNX model to add metadata to it.
        onnx_model = onnx.load(onnx_file_path)

        # Model data, including wandb link (if available).
        meta = onnx_model.metadata_props.add()
        meta.key = "model_source"
        meta.value = json.dumps(model_source)

        # Date exported.
        meta = onnx_model.metadata_props.add()
        meta.key = "date_exported (YYMMDD.HHMMSS)"
        meta.value = str(datetime.now().strftime("%y%m%d.%H%M%S"))

        # Joint names in order expected by ONNX file.
        meta = onnx_model.metadata_props.add()
        meta.key = "joint_names"
        meta.value = ",".join(self.env.scene[self._articulation_name].data.joint_names)

        # Body names.
        meta = onnx_model.metadata_props.add()
        meta.key = "body_names"
        meta.value = ",".join(self.env.scene[self._articulation_name].data.body_names)

        # Command info.
        meta = onnx_model.metadata_props.add()
        meta.key = "commands"
        meta.value = json.dumps(self._data_handler.command_dh.metadata)

        # Sensor info.
        meta = onnx_model.metadata_props.add()
        meta.key = "sensors"
        meta.value = json.dumps(self._data_handler.sensor_dh.metadata)

        # Joint target info.
        meta = onnx_model.metadata_props.add()
        meta.key = "outputs"
        meta.value = json.dumps(self._data_handler.action_dh.metadata())

        # Observation names.
        meta = onnx_model.metadata_props.add()
        meta.key = "obs_term_names"
        meta.value = json.dumps(self.env.observation_manager._group_obs_term_names[self._policy_group_name])

        # Policy update period in seconds.
        meta = onnx_model.metadata_props.add()
        meta.key = "policy_dt"
        meta.value = str(self.env.cfg.sim.dt * self.env.cfg.decimation)

        # Decimation info.
        meta = onnx_model.metadata_props.add()
        meta.key = "decimation"
        meta.value = json.dumps(self.env.cfg.decimation)

        # Sim_dt info.
        meta = onnx_model.metadata_props.add()
        meta.key = "sim_dt"
        meta.value = json.dumps(self.env.cfg.sim.dt)

        # Save the modified model.
        onnx.save(onnx_model, onnx_file_path)

        # Copy metadata to decimation model.
        onnx_default_model = onnx.load(onnx_file_path_default)
        onnx_default_model.metadata_props.extend(onnx_model.metadata_props)
        onnx.save(onnx_default_model, onnx_file_path_default)
