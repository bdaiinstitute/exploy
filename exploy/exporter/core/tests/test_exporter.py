# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Tests for ONNX exporter functionality."""

from unittest.mock import Mock

import pytest
import torch

from exploy.exporter.core.exporter import ExportMode, OnnxEnvironmentExporter


class TestOnnxEnvironmentExporter:
    """Test OnnxEnvironmentExporter class."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock ExportableEnvironment."""
        env = Mock()
        env.decimation = 4

        # Mock context manager
        context_mgr = Mock()
        context_mgr.get_inputs.return_value = {"input1": torch.tensor([1.0])}
        context_mgr.get_input_names.return_value = ["input1"]
        context_mgr.get_outputs.return_value = {"output1": torch.tensor([2.0])}
        context_mgr.get_output_names.return_value = ["output1"]
        context_mgr.write_connections.return_value = None
        context_mgr.metadata = {"test": "metadata"}

        env.context_manager.return_value = context_mgr
        env.empty_actor_observations.return_value = torch.tensor([0.5])
        env.empty_actions.return_value = torch.tensor([0.1])
        env.compute_observations.return_value = torch.tensor([0.5])
        env.command_updates = []
        env.process_actions.return_value = None
        env.apply_actions.return_value = None
        env.metadata.return_value = {"env_key": "env_value"}

        return env

    @pytest.fixture
    def mock_actor(self):
        """Create a mock actor network."""
        actor = torch.nn.Linear(1, 1)
        actor.eval()
        return actor

    def test_forward_default_mode(self, mock_env, mock_actor):
        """Test forward pass in Default export mode."""
        exporter = OnnxEnvironmentExporter(
            env=mock_env,
            actor=mock_actor,
            normalizer=None,
            verbose=False,
            opset_version=20,
            ir_version=11,
        )
        exporter.export_mode = ExportMode.Default

        input_data = {"input1": torch.tensor([1.0])}
        result = exporter.forward(input_data)

        # Should return (actions, observations, *output_data)
        assert len(result) >= 2
        assert isinstance(result[0], torch.Tensor)  # actions
        assert isinstance(result[1], torch.Tensor)  # observations

        # Verify environment methods were called
        mock_env.context_manager().write_connections.assert_called_once()
        mock_env.compute_observations.assert_called_once()
        mock_env.process_actions.assert_called_once()
        mock_env.apply_actions.assert_called_once()

    def test_forward_process_actions_mode(self, mock_env, mock_actor):
        """Test forward pass in ProcessActions export mode."""
        exporter = OnnxEnvironmentExporter(
            env=mock_env,
            actor=mock_actor,
            normalizer=None,
            verbose=False,
            opset_version=20,
            ir_version=11,
        )
        exporter.export_mode = ExportMode.ProcessActions

        input_data = {"input1": torch.tensor([1.0])}
        result = exporter.forward(input_data)

        # Should return (actions, observations, *output_data)
        assert len(result) >= 2

        # In ProcessActions mode, compute_observations and process_actions should not be called
        mock_env.compute_observations.assert_not_called()
        mock_env.process_actions.assert_not_called()
        # But apply_actions should still be called
        mock_env.apply_actions.assert_called_once()
