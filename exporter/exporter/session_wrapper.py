# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pathlib

import numpy as np
import onnxruntime as ort
import torch

from exporter.utils.paths import prepare_onnx_paths


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
        # Prepare file paths
        session_paths = prepare_onnx_paths(
            output_dir=onnx_folder,
            filename=onnx_file_name,
            debug_suffixes=["optimized"],
        )

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
            sess_options.optimized_model_filepath = str(session_paths.get_debug_path("optimized"))

        self._onnx_file_path = session_paths.main
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
        assert output_name in self.output_names, (
            f"Output '{output_name}' not found in expected outputs: {self.output_names}"
        )
        return self._results[self.output_names.index(output_name)]

    def reset(self):
        """Reset the internal results to zeros to avoid stale data at environment reset."""
        self._results = [np.zeros_like(output) for output in self._results]
