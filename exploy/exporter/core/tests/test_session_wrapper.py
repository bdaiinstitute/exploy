# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Tests for ONNX session wrapper."""

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from exploy.exporter.core.session_wrapper import SessionWrapper


class TestSessionWrapper:
    """Test SessionWrapper class."""

    @pytest.fixture
    def mock_onnx_model(self):
        """Create a simple ONNX model for testing."""
        # Create a simple torch model
        model = torch.nn.Linear(3, 2)
        model.eval()

        # Export to ONNX
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = pathlib.Path(tmpdir) / "test_model.onnx"
            dummy_input = torch.randn(1, 3)

            torch.onnx.export(
                model,
                dummy_input,
                str(model_path),
                input_names=["input"],
                output_names=["output"],
            )

            yield model_path

    def test_session_wrapper_initialization(self, mock_onnx_model):
        """Test SessionWrapper initialization."""
        policy = torch.nn.Linear(3, 2)
        wrapper = SessionWrapper(
            onnx_folder=mock_onnx_model.parent,
            onnx_file_name=mock_onnx_model.name,
            policy=policy,
            optimize=True,
        )

        # Check that optimized model path exists in debug directory
        debug_dir = mock_onnx_model.parent / "debug"
        assert debug_dir.exists()

        optimized_path = debug_dir / f"{mock_onnx_model.stem}_optimized.onnx"
        # Optimized model should be created during session initialization
        assert optimized_path.exists()

        assert wrapper.get_torch_model() == policy

    def test_session_wrapper_inference(self, mock_onnx_model):
        """Test SessionWrapper inference."""
        wrapper = SessionWrapper(
            onnx_folder=mock_onnx_model.parent,
            onnx_file_name=mock_onnx_model.name,
            policy=None,
            optimize=False,
        )

        # Run inference
        input_data = np.random.randn(1, 3).astype(np.float32)
        results = wrapper(input=input_data)

        assert results is not None
        assert len(results) > 0
        assert isinstance(results[0], np.ndarray)

        output = wrapper.get_output_value("output")
        assert output is not None
        assert isinstance(output, np.ndarray)

        # Reset
        wrapper.reset()

        # After reset, results should be zeros with same shape
        output = wrapper.get_output_value("output")
        assert np.all(output == 0)
        assert output.shape == results[0].shape

        # Run multiple inferences
        for _ in range(5):
            input_data = np.random.randn(1, 3).astype(np.float32)
            results = wrapper(input=input_data)
            assert results is not None
            assert len(results) > 0
