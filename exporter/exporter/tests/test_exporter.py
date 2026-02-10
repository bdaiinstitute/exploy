# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Tests for ONNX exporter functionality."""

import pathlib
import tempfile
from unittest.mock import MagicMock, Mock, patch

import onnx
import pytest
import torch

from exporter.exporter import (
    ExportMode,
    OnnxEnvironmentExporter,
    are_values_on_device,
    export_environment_as_onnx,
)


class TestAreValuesOnDevice:
    """Test device validation utility."""

    def test_all_values_on_cpu(self):
        """Test when all tensors are on CPU."""
        tensors = (torch.tensor([1.0]), torch.tensor([2.0]))
        names = ["tensor1", "tensor2"]
        assert are_values_on_device(names, tensors, expected_device="cpu", verbose=False)

    def test_wrong_device_check(self):
        """Test when checking for wrong device returns False."""
        tensors = (torch.tensor([1.0]), torch.tensor([2.0]))
        names = ["tensor1", "tensor2"]
        # Tensors are on CPU, but we're checking for cuda - should return False
        assert not are_values_on_device(names, tensors, expected_device="cuda", verbose=False)

    def test_with_dict_values(self):
        """Test when values contain dictionaries."""
        dict_val = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        tensors = (dict_val,)
        names = ["dict"]
        assert are_values_on_device(names, tensors, expected_device="cpu", verbose=False)


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
        )
        exporter._export_device = "cpu"
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
        )
        exporter._export_device = "cpu"
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
