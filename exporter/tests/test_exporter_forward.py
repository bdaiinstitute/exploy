# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
from typing import TYPE_CHECKING


from isaaclab.app import AppLauncher

if TYPE_CHECKING:
    import argparse
import argparse

# Create argument parser for headless mode
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.headless = True
args_cli.num_envs = 1

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import unittest
from unittest.mock import MagicMock, patch

from exporter.exporter import ExportMode, _OnnxEnvironmentExporter


class TestExporter(unittest.TestCase):
    """Test that observations are only computed in default export mode without starting simulation."""

    def create_mock_env(self):
        """Create a mock environment with all necessary attributes."""
        mock_env = MagicMock()

        # Mock observation manager
        mock_env.observation_manager = MagicMock()
        mock_env.observation_manager.active_terms = {"policy": MagicMock()}
        mock_env.observation_manager.group_obs_concatenate = {"policy": True}
        mock_env.observation_manager._group_obs_term_cfgs = {}
        # group_obs_dim is indexed by [group_name][env_id], so create a tensor-like structure
        mock_env.observation_manager.group_obs_dim = {"policy": torch.tensor([32])}
        mock_env.observation_manager.compute = MagicMock(return_value={"policy": torch.zeros(1, 32)})
        mock_env.obs_buf = {"policy": torch.zeros(1, 32)}

        # Mock action manager
        mock_env.action_manager = MagicMock()
        mock_env.action_manager.process_action = MagicMock()
        mock_env.action_manager.apply_action = MagicMock()
        mock_env.action_manager.action_term_dim = 16
        mock_env.action_manager.total_action_dim = 16

        # Mock scene with articulations and rigid objects
        mock_env.scene = MagicMock()
        mock_articulation = MagicMock()

        # Mock articulation data with proper tensors
        mock_articulation.num_bodies = 2
        mock_articulation_data = MagicMock()
        mock_articulation_data.root_lin_vel_b = torch.zeros(1, 3)
        mock_articulation_data.root_ang_vel_b = torch.zeros(1, 3)
        mock_articulation_data.body_pos_w = torch.zeros(1, 2, 3)
        mock_articulation_data.body_quat_w = torch.zeros(1, 2, 4)
        mock_articulation_data.body_lin_vel_w = torch.zeros(1, 2, 3)
        mock_articulation_data.body_ang_vel_w = torch.zeros(1, 2, 3)
        mock_articulation_data.joint_pos = torch.zeros(1, 10)
        mock_articulation_data.joint_vel = torch.zeros(1, 10)

        mock_articulation._data = mock_articulation_data
        mock_env.scene.articulations = {"robot": mock_articulation}
        mock_env.scene.rigid_objects = {}
        mock_env.scene.sensors = {}

        return mock_env

    def create_mock_actor(self):
        """Create a simple mock actor network."""
        return torch.nn.Sequential(
            torch.nn.Linear(32, 16),
        )

    def test_observations_computed_in_default_mode(self):
        """Test that observations are computed when export mode is Default."""

        # Create mock environment and actor
        mock_env = self.create_mock_env()
        mock_actor = self.create_mock_actor()

        # Create exporter - patch ArticulationDataSource to avoid complex tensor setup
        with patch("exporter.exporter.DataHandlerManager"), patch(
            "exporter.exporter.ArticulationDataSource"
        ), patch("exporter.exporter.RigidObjectDataSource"):
            exporter = _OnnxEnvironmentExporter(
                env=mock_env,
                export_env_ids=0,
                actor=mock_actor,
                normalizer=None,
                verbose=False,
            )

        # Set to Default mode
        exporter.export_mode = ExportMode.Default

        # Prepare minimal input data
        imu_data = {}
        sensor_data = {}
        command_data = {}
        art_data = {}
        rigid_object_data = {}
        memory_data = {}

        # Mock data handler to return empty dicts
        exporter._data_handler = MagicMock()
        exporter._data_handler.set_to_source = MagicMock()
        exporter._data_handler.command_dh = MagicMock()
        # Create a mock command term to verify _update_command is called
        mock_command_term = MagicMock()
        exporter._data_handler.command_dh.command_terms_to_update = [mock_command_term]
        exporter._data_handler.action_dh = MagicMock()
        exporter._data_handler.action_dh.values = MagicMock(return_value=[])
        exporter._data_handler.memory_dh = MagicMock()
        exporter._data_handler.memory_dh.get_updated_values_as_outputs = MagicMock(return_value=[])

        # Call forward pass
        with torch.no_grad():
            exporter.forward(
                imu_data=imu_data,
                sensor_data=sensor_data,
                command_data=command_data,
                art_data=art_data,
                rigid_object_data=rigid_object_data,
                memory_data=memory_data,
            )

        # Assert that compute was called
        mock_env.observation_manager.compute.assert_called_once()
        # Assert that process_action and apply_action were called
        mock_env.action_manager.process_action.assert_called_once()
        mock_env.action_manager.apply_action.assert_called_once()
        # Assert that _update_command was called on command terms
        mock_command_term._update_command.assert_called_once()

    def test_observations_not_computed_in_process_actions_mode(self):
        """Test that observations are NOT computed when export mode is ProcessActions."""

        # Create mock environment and actor
        mock_env = self.create_mock_env()
        mock_actor = self.create_mock_actor()

        # Create exporter - patch ArticulationDataSource to avoid complex tensor setup
        with patch("exporter.exporter.DataHandlerManager"), patch(
            "exporter.exporter.ArticulationDataSource"
        ), patch("exporter.exporter.RigidObjectDataSource"):
            exporter = _OnnxEnvironmentExporter(
                env=mock_env,
                export_env_ids=0,
                actor=mock_actor,
                normalizer=None,
                verbose=False,
            )

        # Set to ProcessActions mode
        exporter.export_mode = ExportMode.ProcessActions

        # Prepare minimal input data with dummy action (required for ProcessActions mode)
        imu_data = {}
        sensor_data = {}
        command_data = {}
        art_data = {}
        rigid_object_data = {}
        memory_data = {"memory.actions.in": torch.zeros(16)}

        # Mock data handler to return empty dicts
        exporter._data_handler = MagicMock()
        exporter._data_handler.set_to_source = MagicMock()
        exporter._data_handler.command_dh = MagicMock()
        # Create a mock command term to verify _update_command is NOT called
        mock_command_term = MagicMock()
        exporter._data_handler.command_dh.command_terms_to_update = [mock_command_term]
        exporter._data_handler.action_dh = MagicMock()
        exporter._data_handler.action_dh.values = MagicMock(return_value=[])
        exporter._data_handler.memory_dh = MagicMock()
        exporter._data_handler.memory_dh.get_updated_values_as_outputs = MagicMock(return_value=[])

        # Reset the mock to clear any calls from initialization
        mock_env.observation_manager.compute.reset_mock()

        # Call forward pass
        with torch.no_grad():
            exporter.forward(
                imu_data=imu_data,
                sensor_data=sensor_data,
                command_data=command_data,
                art_data=art_data,
                rigid_object_data=rigid_object_data,
                memory_data=memory_data,
            )

        # Assert that compute was NOT called
        mock_env.observation_manager.compute.assert_not_called()
        # Assert that process_action and apply_action WERE called
        mock_env.action_manager.process_action.assert_called_once()
        mock_env.action_manager.apply_action.assert_called_once()
        # Assert that _update_command was NOT called on command terms
        mock_command_term._update_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
