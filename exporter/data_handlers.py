# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch

# from rai.core.sensors import LidarRangeImage
# from rai.core.sensors.imu_history import ImuHistory, ImuHistoryCfg, ImuHistoryData
# from rai.core.utils.commands.pose.command import LinkPoseCommand, YawOnlyPose3dAtDistanceCommand
# from rai.core.utils.commands.velocity.command import Se2VelocityGamepadCommand
from exporter import ExportEnvIdsType
from exporter.articulation_data import articulation_data_to_dict, dict_to_articulation_data
from exporter.rigid_object_data import dict_to_rigid_object_data, rigid_object_data_to_dict
# from rai.core.utils.managers import prim_path_to_articulation_and_body_ids, prim_path_to_body_expr

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CommandManager, CommandTerm
from isaaclab.sensors import RayCaster, RayCasterCamera, SensorBase, SensorBaseCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import (
    GridPatternCfg,
    LidarPatternCfg,
    PatternBaseCfg,
    PinholeCameraPatternCfg,
)


class DataHandler:
    """Base class for data handlers.

    This class provides a common interface for data handlers used in the rai.core ONNX exporter.
    In the exporter, data handlers are responsible for storing data from a given class (e.g., from an `ArticulationData`
    container) into a dictionary (i.e., `self._data`). The dictionary is then used to:
        - get keys and values which are passed to the `torch.onnx.export` function
        - set states to the source class (i.e., `ArticulationData` in the example above) to create the computational
          graph when exporting to ONNX

    This handler additionally provides an interface for a `metadata` container, a dictionary of info related to the
    source data container that we add to the ONNX file. For example, we store information about command types into
    the metadata dictionary.
    """

    def __init__(self, env_id: ExportEnvIdsType):
        self._env_id = env_id

        self._data = {}
        self._metadata = {}

    @property
    def data(self):
        """A dictionary holding data from a source data container."""
        return self._data

    @property
    def data_numpy(self):
        """A dictionary holding data from a source data container, where all values are converted to numpy arrays."""
        return {k: v.cpu().numpy() for k, v in self._data.items()}

    @property
    def metadata(self):
        """A dictionary holding metadata that can be stored in an ONNX file."""
        return self._metadata

    def to(self, device: str):
        """Move all values to a device."""
        for k, v in self._data.items():
            self._data[k] = v.to(torch.device(device=device))

    def set_from_source(self, *args, **kwargs):
        """Given a source data container, update this handler's data dictionary."""
        raise NotImplementedError

    def set_to_source(self, *args, **kwargs):
        """Set the data in a data container using this handler's data dictionary."""
        raise NotImplementedError

    def set_from_data(self, data: dict[str, torch.Tensor]):
        """Set the data dictionary directly.

        Args:
            data: A dictionary holding data.
        """
        for k, v in data.items():
            if k not in self._data.keys():
                raise KeyError(f"Key '{k}' not found in data handler.")
            self._data[k] = v.clone()


class CommandDataHandler(DataHandler):
    """Handle data stored in a command manager."""

    def __init__(self, command_manager: CommandManager, env_id: ExportEnvIdsType, device: str = "cpu"):
        super().__init__(env_id=env_id)

        self._device = device
        self._command_terms_to_update = []
        self.set_from_source(source=command_manager)

    def set_from_source(self, source: CommandManager):
        """Update this handler's data dictionary from a `CommandManager`.

        Args:
            source: A command manager.
        """
        # from rai.umv.umv_core.trail.mdp.commands import TrailCommand

        self._data.clear()
        self._metadata.clear()
        for command_name in source.active_terms:
            command = source.get_term(name=command_name)

            # if isinstance(command, TrailCommand):
            #     # Digital twins commands are supported, but are not intended to appear as inputs to the ONNX
            #     # computational graph.
            #     # Instead, the environment uses these commands to update an internal buffer, which is then
            #     # used by the scene sensors to generate observations.
            #     self._command_terms_to_update.append(command)
            # elif isinstance(command, LinkPoseCommand):
            #     self._data[f"command.{command_name}"] = torch.cat(
            #         [command.command_pos_b[self._env_id], command.command_orient_quat_b[self._env_id]], dim=-1
            #     )
            #     self._metadata[f"command.{command_name}"] = {
            #         "type": "se3_pose",
            #     }
            # elif isinstance(command, Se2VelocityGamepadCommand):
            #     self._data[f"command.{command_name}"] = command.se2_command[self._env_id]
            #     self._metadata[f"command.{command_name}"] = {
            #         "type": "se2_velocity",
            #         "ranges": {
            #             "lin_vel_x": command.cfg.ranges.lim_x,
            #             "lin_vel_y": command.cfg.ranges.lim_y,
            #             "ang_vel_z": command.cfg.ranges.lim_theta,
            #         },
            #     }
            if (
                type(command).__name__ == "UMVVelocityCommandSymmetric"
                or type(command).__name__ == "UMVRelHeadingVelocityCommand"
            ):
                self._data[f"command.{command_name}"] = command._command[self._env_id]
                self._metadata[f"command.{command_name}"] = {
                    "type": "se2_velocity",
                    "ranges": {
                        "lin_vel_x": command.cfg.ranges.lin_vel,
                        "lin_vel_y": command.cfg.ranges.lin_vel,
                        "ang_vel_z": command.cfg.ranges.ang_vel,
                    },
                }
            elif isinstance(command, UniformVelocityCommand):
                self._data[f"command.{command_name}"] = command.vel_command_b[self._env_id]
                self._metadata[f"command.{command_name}"] = {
                    "type": "se2_velocity",
                    "ranges": {
                        "lin_vel_x": command.cfg.ranges.lin_vel_x,
                        "lin_vel_y": command.cfg.ranges.lin_vel_y,
                        "ang_vel_z": command.cfg.ranges.ang_vel_z,
                    },
                }
            # elif isinstance(command, YawOnlyPose3dAtDistanceCommand):
            #     self._data[f"command.{command_name}"] = command.se3_command[self._env_id]
            #     self._metadata[f"command.{command_name}"] = {
            #         "type": "se3_pose",
            #     }
            elif type(command).__name__ == "ObjectTargetCommand":
                self._data[f"command.{command_name}"] = command._command[self._env_id]
                self._metadata[f"command.{command_name}"] = {
                    "type": "se3_pose",
                }
            elif type(command).__name__ == "LookAtTargetCommand":
                pass
            else:
                raise RuntimeError(f"Got unsupported command type: {type(command)}")
        self.to(self._device)

    def set_to_source(self, source: CommandManager):
        """Replace data in a `CommandManager` with the data stored in this handler.

        Args:
            source: A command manager.
        """
        # from rai.umv.umv_core.trail.mdp.commands import TrailCommand

        for key in self._data:
            command_name = key.split(".")[-1]
            assert command_name in source.active_terms
            command_term = source.get_term(command_name)

            # if isinstance(command_term, TrailCommand):
            #     # Digital twins and trail commands are supported, but are not intended to appear as inputs to the ONNX
            #     # computational graph.
            #     # Instead, the environment uses these commands to update an internal buffer, which is then
            #     # used by the scene sensors to generate observations.
            #     pass
            # elif isinstance(command_term, LinkPoseCommand):
            #     command_term.command_pos_b[self._env_id, :] = self._data[f"command.{command_name}"][:3]
            #     command_term.command_orient_quat_b[self._env_id, :] = self._data[f"command.{command_name}"][3:7]
            # elif isinstance(command_term, Se2VelocityGamepadCommand):
            #     command_term.se2_command[self._env_id, :] = self._data[f"command.{command_name}"]
            if (
                type(command_term).__name__ == "UMVVelocityCommandSymmetric"
                or type(command_term).__name__ == "UMVRelHeadingVelocityCommand"
            ):
                command_term._command[self._env_id] = self._data[f"command.{command_name}"]
            elif isinstance(command_term, UniformVelocityCommand):
                command_term.vel_command_b[self._env_id] = self._data[f"command.{command_name}"]
            # elif isinstance(command_term, YawOnlyPose3dAtDistanceCommand):
            #     command_term.se3_command[self._env_id] = self._data[f"command.{command_name}"]
            elif type(command_term).__name__ == "ObjectTargetCommand":
                command_term._command[self._env_id] = self._data[f"command.{command_name}"]
            elif type(command_term).__name__ == "LookAtTargetCommand":
                pass
            else:
                raise RuntimeError(f"Got unsupported command type: {type(command_term)}")

    @property
    def command_terms_to_update(self) -> list[CommandTerm]:
        return self._command_terms_to_update


class ArticulationDataHandler(DataHandler):
    """Handle data stored in a `ArticulationData` instance."""

    def __init__(self, articulation: Articulation, env_id: ExportEnvIdsType, device: str = "cpu"):
        self._added_body_state_names = {}
        self._device = device

        super().__init__(env_id=env_id)
        self.set_from_source(source=articulation)

    def add_body_states(self, articulation: Articulation, body_names_expression: list[str]):
        for body_expr in set(body_names_expression):
            try:
                body_ids, body_names = articulation.find_bodies(body_expr)
            except ValueError:
                print(
                    f"Could not find body for expression '{body_expr}'. Body state will not be added to data handler."
                )
                return
            for id, name in zip(body_ids, body_names):
                if id > 0:
                    self._data[f"body.{name}.pos"] = articulation.data.body_pos_w[self._env_id, id].to(self._device)
                    self._data[f"body.{name}.quat"] = articulation.data.body_quat_w[self._env_id, id].to(self._device)
                    self._added_body_state_names[name] = {"body_id": id}

    def set_from_source(self, source: Articulation):
        """Update this handler's data dictionary from a `ArticulationData` instance.

        Args:
            source: An articulation data container.
        """
        self._data = articulation_data_to_dict(source=source.data, env_id=self._env_id, device=self._device)

    def set_to_source(self, source: Articulation):
        """Replace data in a `ArticulationData` with the data stored in this handler.

        Args:
            source: A command manager.
        """
        dict_to_articulation_data(data=self._data, target=source.data, env_id=self._env_id)

        for key, value in self._added_body_state_names.items():
            body_id = value["body_id"]
            source.data._body_pos_w[self._env_id, body_id] = self._data[f"body.{key}.pos"]
            source.data._body_quat_w[self._env_id, body_id] = self._data[f"body.{key}.quat"]


class RigidObjectDataHandler(DataHandler):
    """Handle data stored in a `RigidObjectData` instance."""

    def __init__(
        self,
        rigid_body_data: dict[str, RigidObject],
        env_id: ExportEnvIdsType,
        device: str = "cpu",
    ):
        self._device = device

        super().__init__(env_id=env_id)
        self.set_from_source(source_dict=rigid_body_data)

    def set_from_source(
        self,
        source_dict: dict[str, RigidObject],
    ):
        """Update this handler's data dictionary from a `RigidObject` instance.

        Args:
            source_dict: A dictionary holding the state of a `RigidObject` in the scene.
        """
        self._data.clear()
        for object_name, source in source_dict.items():
            self._data = rigid_object_data_to_dict(
                object_name=object_name, source=source.data, env_id=self._env_id, device=self._device
            )

        self.to(self._device)

    def set_to_source(self, source_dict: dict[str, RigidObject]):
        """Replace data in a `RigidObject` with the data stored in this handler.

        Args:
            source: A dictionary holding the state of a `RigidObject` in the scene
        """
        for object_name, source in source_dict.items():
            dict_to_rigid_object_data(data=self._data, object_name=object_name, target=source.data, env_id=self._env_id)


class IMUDataHandler(DataHandler):
    """Handle IMU data stored in both a `ImuHistory` instance and an `ArticulationData` instance."""

    def __init__(
        self,
        sensors: dict[str, SensorBase],
        articulation: Articulation,
        env_id: ExportEnvIdsType,
        device: str = "cpu",
    ):
        # Store a map from IMU link name (as found in the urdf) to member variable name (as found in the
        # scene configuration).
        self._imu_link_to_imu_var_map = {}

        self._device = device

        super().__init__(env_id=env_id)
        self.set_from_source(sensors=sensors, articulation=articulation)

    def set_from_source(
        self,
        sensors: dict[str, SensorBase],
        articulation: Articulation,
    ):
        """Update this handler's data dictionary from an `Articulation` instance and a dictionary of `SensorBase`
        instances.

        Args:
            sensors: A dictionary of `SensorBase` instances.
            articulation: An `Articulation` instance.
        """
        self._data.clear()
        for sensor_cfg_name in sensors.keys():
            if isinstance(sensors[sensor_cfg_name], ImuHistory):
                sensor = sensors[sensor_cfg_name]
                sensor_cfg: ImuHistoryCfg = sensor.cfg
                sensor_data: ImuHistoryData = sensor.data
                imu_name = sensor_cfg.prim_path.split("/")[-1]
                imu_link_idx, _ = articulation.find_bodies(imu_name)
                self._imu_link_to_imu_var_map[imu_name] = sensor_cfg_name
                imu_quat_w = articulation.data.body_quat_w[self._env_id, imu_link_idx[0]]
                ang_vel_in_imu = sensor_data.ang_vel_b_history[self._env_id, 0, :]

                self._data[f"imu_data.quat.{imu_name}"] = imu_quat_w
                self._data[f"imu_data.ang_vel.{imu_name}"] = ang_vel_in_imu

        self.to(self._device)

    def set_to_source(
        self,
        sensors: dict[str, SensorBase],
        articulation: Articulation,
    ):
        """Replace IMU data in a `ArticulationData` and a dictionary of `SensorBase` with the data stored in this
        handler.

        Args:
            sensors: A dictionary of `SensorBase` instances.
            articulation: An `Articulation` instance.
        """
        # for key, val in self._data.items():
        #     sensor_name = key.split(".")[-1]
        #     if "quat" in key:
        #         # Set orientation.
        #         imu_link_idx, _ = articulation.find_bodies(sensor_name)
        #         articulation.data._body_quat_w[self._env_id, imu_link_idx[0]] = val
        #     elif "ang_vel" in key:
        #         # Set angular velocity.
        #         sensor_cfg_name = self._imu_link_to_imu_var_map[sensor_name]
        #         imu_sensor: ImuHistory = sensors[sensor_cfg_name]
        #         imu_sensor.data.ang_vel_b_history[self._env_id, 0, :] = val
        #     else:
        #         raise RuntimeError(f"Expected to find 'quat' or 'ang_vel' in key: {key}")


class SensorDataHandler(DataHandler):
    def __init__(
        self,
        sensors: dict[str, SensorBase],
        env_id: ExportEnvIdsType,
        device: str = "cpu",
    ):
        super().__init__(env_id=env_id)
        self._device = device
        self._dependent_body_names = []
        self.set_from_source(source=sensors)

    def get_dependent_body_names(self) -> list[str]:
        return self._dependent_body_names

    def set_from_source(self, source: dict[str, SensorBase]):
        """Update this handler's data dictionary from a dictionary of `SensorBase` instances.

        Args:
            source: A dictionary of `SensorBase` instances.
        """

        # from rai.umv.umv_core.trail.sensors import TrailRayCaster

        self._data.clear()
        for sensor_name_in_source in source.keys():
            sensor: SensorBase = source[sensor_name_in_source]
            sensor_key: str = f"sensor.{sensor_name_in_source}"
            if isinstance(sensor, RayCaster):
                # Prepare an empty metadata dict.
                self._metadata[sensor_key] = {}
                self._metadata[sensor_key]["offset_x"] = sensor.cfg.offset.pos[0]
                self._metadata[sensor_key]["offset_y"] = sensor.cfg.offset.pos[1]

                # Get the ray caster type and gather data.
                if type(sensor) is RayCaster:
                    self._metadata[sensor_key]["type"] = "ray_caster"
                    self._data[sensor_key] = sensor._data.ray_hits_w[self._env_id, :, 2]
                elif type(sensor) is RayCasterCamera:
                    self._metadata[sensor_key]["type"] = "depth_image"
                    self._data[sensor_key] = sensor._data.output["distance_to_image_plane"][self._env_id].flatten()
                # elif type(sensor) is LidarRangeImage:
                #     self._metadata[sensor_key]["type"] = "lidar_range_image"
                #     self._data[sensor_key] = sensor._data.lidar_range_image[self._env_id].view(1, -1)
                # elif type(sensor) is TrailRayCaster:
                #     self._metadata[sensor_key]["type"] = "trail_ray_caster"
                #     self._data[f"{sensor_key}.height"] = sensor._data.ray_hits_w[self._env_id, :, 2]
                #     self._data[f"{sensor_key}.r"] = sensor._data.rgb_of_ray_hits[self._env_id, :, 0]
                #     self._data[f"{sensor_key}.g"] = sensor._data.rgb_of_ray_hits[self._env_id, :, 1]
                #     self._data[f"{sensor_key}.b"] = sensor._data.rgb_of_ray_hits[self._env_id, :, 2]
                else:
                    raise RuntimeError(f"Got unsupported RayCaster type: {type(sensor)}")

                # Add a state dependencency on the body to which the sensor is attached.
                # self._dependent_body_names.append(prim_path_to_body_expr(sensor.cfg.prim_path))

                # Add the ray caster pattern to the metadata.
                pattern_cfg: PatternBaseCfg = sensor.cfg.pattern_cfg
                if type(pattern_cfg) is LidarPatternCfg:
                    self._metadata[sensor_key].update(**{
                        "pattern_type": "lidar_pattern",
                        "h_res": sensor._data.lidar_range_image.shape[-1],
                        "v_res": sensor._data.lidar_range_image.shape[-2],
                        "v_fov_min_deg": pattern_cfg.vertical_fov_range[0],
                        "v_fov_max_deg": pattern_cfg.vertical_fov_range[1],
                        "max_range_m": sensor.cfg.max_distance,
                        "unobserved_value": sensor.cfg.unobserved_value,
                    })
                elif type(pattern_cfg) is GridPatternCfg:
                    self._metadata[sensor_key].update(**{
                        "pattern_type": "grid_pattern",
                        "resolution": pattern_cfg.resolution,
                        "size_x": pattern_cfg.size[0],
                        "size_y": pattern_cfg.size[1],
                    })
                elif type(pattern_cfg) is PinholeCameraPatternCfg:
                    # Convert IsaacLab camera parameters to a standard pinhole intrinsic model:
                    #
                    #   f_x = f * μ_x = f[mm] * image_width[pixels]  / sensor_width[mm]
                    #   f_y = f * μ_y = f[mm] * image_height[pixels] / sensor_height[mm]
                    #   c_x, c_y are the optical center offsets in pixel coordinates.
                    #
                    # IsaacLab uses the terms "horizontal/vertical aperture" and offsets to parameterize the model.
                    # Despite the name, these "apertures" do not refer to a physical aperture (zero by definition for an ideal
                    # pinhole camera), but correspond to the sensor width and height. IsaacLab also expresses the focal length
                    # and apertures in centimeters; however, the units cancel out when converting to pixels.
                    #
                    # For reference, see IsaacLab’s internal conversions between ray-casting patterns and intrinsic matrices:
                    #   https://github.com/isaac-sim/IsaacLab/blob/244483eec605dfca6ee11561550b4b329144d24c/source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns_cfg.py#L121
                    #   https://github.com/isaac-sim/IsaacLab/blob/244483eec605dfca6ee11561550b4b329144d24c/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_camera.py#L393-L397
                    width = pattern_cfg.width
                    height = pattern_cfg.height
                    fx = width * pattern_cfg.focal_length / pattern_cfg.horizontal_aperture
                    if pattern_cfg.vertical_aperture:
                        fy = height * pattern_cfg.focal_length / pattern_cfg.vertical_aperture
                    else:
                        fy = fx
                    cx = fx * pattern_cfg.horizontal_aperture_offset + width / 2
                    cy = fy * pattern_cfg.vertical_aperture_offset + height / 2
                    self._metadata[sensor_key].update(**{
                        "pattern_type": "pinhole_pattern",
                        "height": height,
                        "width": width,
                        "fx": fx,
                        "fy": fy,
                        "cx": cx,
                        "cy": cy,
                        "data_types": sensor.cfg.data_types,
                        "depth_clipping_behavior": sensor.cfg.depth_clipping_behavior,
                    })
                else:
                    raise RuntimeError(f"Got unhandled pattern type for RayCaster: {type(pattern_cfg)}")

        self.to(self._device)

    def set_to_source(self, source: dict[str, SensorBase], articulations: dict[str, Articulation]):
        """Replace sensor data in a dictionary of `SensorBase` with the data stored in this handler.

        Args:
            source: A dictionary of `SensorBase` instances.
        """

        # from rai.umv.umv_core.trail.sensors import TrailRayCaster

        for sensor_name_in_source in source.keys():
            sensor: SensorBase = source[sensor_name_in_source]
            sensor_cfg: SensorBaseCfg = sensor.cfg
            sensor_key = f"sensor.{sensor_name_in_source}"

            if isinstance(sensor, RayCaster):
                # Set the state of the body to which the sensor is attached.
                prim_path = sensor_cfg.prim_path
                # articulation, sensor_body_ids = prim_path_to_articulation_and_body_ids(
                #     prim_path=prim_path, articulations=articulations
                # )
                if len(sensor_body_ids) > 1:
                    raise RuntimeError(
                        "Got multiple body ids, while RayCaster prim path should correspond to a single body. Got body"
                        f" ids: {sensor_body_ids}"
                    )

                sensor_pos_w = articulation.data.body_pos_w[self._env_id, sensor_body_ids[0]]
                sensor_quat_w = articulation.data.body_quat_w[self._env_id, sensor_body_ids[0]]
                sensor._data.pos_w[self._env_id, :] = sensor_pos_w
                sensor._data.quat_w[self._env_id, :] = sensor_quat_w

                # Set the ray caster data.
                if type(sensor) is RayCaster:
                    sensor._data.ray_hits_w[self._env_id, :, 2] = self._data[sensor_key]
                elif type(sensor) is LidarRangeImage:
                    v_res = sensor._data.lidar_range_image.shape[-2]
                    sensor._data.lidar_range_image[self._env_id] = self._data[sensor_key].view(1, v_res, -1)
                # elif type(sensor) is TrailRayCaster:
                #     sensor._data.rgb_of_ray_hits[self._env_id, :, 0] = self._data[f"{sensor_key}.r"]
                #     sensor._data.rgb_of_ray_hits[self._env_id, :, 1] = self._data[f"{sensor_key}.g"]
                #     sensor._data.rgb_of_ray_hits[self._env_id, :, 2] = self._data[f"{sensor_key}.b"]
                #     sensor._data.ray_hits_w[self._env_id, :, 2] = self._data[f"{sensor_key}.height"]
                else:
                    raise RuntimeError(f"Got unsupported RayCaster type: {type(sensor)}")
