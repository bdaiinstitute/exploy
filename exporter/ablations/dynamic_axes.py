# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import onnxruntime as ort


class ExportMethod(Enum):
    ONNX = auto()
    ONNX_DYNAMO = auto()
    JIT_TRACE = auto()
    JIT_SCRIPT = auto()


@dataclass
class DynamicAxesCaseCfg:
    """Configuration for a specific test case."""

    env_id: Callable[[int], int | list | slice] = lambda num_envs: 0
    num_envs: int = 1
    direct_allocation: bool = False
    num_joints: int = 29
    name: str = "current"


@dataclass
class AblationCfg:
    current: DynamicAxesCaseCfg = field(default_factory=DynamicAxesCaseCfg)
    sequence: DynamicAxesCaseCfg = field(
        default_factory=lambda: DynamicAxesCaseCfg(
            env_id=lambda num_envs: list(range(num_envs)),
            num_envs=1,
            direct_allocation=False,
            name="sequence",
        )
    )
    sequence_direct_allocation: DynamicAxesCaseCfg = field(
        default_factory=lambda: DynamicAxesCaseCfg(
            env_id=lambda num_envs: list(range(num_envs)),
            num_envs=1,
            direct_allocation=True,
            name="sequence_direct_allocation",
        )
    )
    slice: DynamicAxesCaseCfg = field(
        default_factory=lambda: DynamicAxesCaseCfg(
            env_id=lambda num_envs: slice(None),
            num_envs=1,
            direct_allocation=False,
            name="slice",
        )
    )


def create_log_dir() -> Path:
    log_dir = Path("/workspace/logs") / "dynamic_axes" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# =============================================================================
# MockModel (same for ALL export methods)
# =============================================================================


class MockModel(torch.nn.Module):
    """Demonstrates export issues with fixed-size buffers."""

    def __init__(self, case: DynamicAxesCaseCfg):
        super().__init__()
        self.case = case
        self._env_id = case.env_id(case.num_envs)

        # Pre-allocated buffers with concrete shape
        self.joint_pos = torch.zeros(case.num_envs, case.num_joints)
        self.default_joint_pos = torch.zeros(case.num_envs, case.num_joints)

    def forward(self, joint_pos_input: torch.Tensor) -> torch.Tensor:
        if self.case.direct_allocation:
            self.joint_pos = joint_pos_input
        else:
            self.joint_pos[self._env_id] = joint_pos_input

        joint_pos_rel = self.joint_pos - self.default_joint_pos
        return joint_pos_rel


# =============================================================================
# ONNX Export & Test
# =============================================================================


def export_onnx(model: torch.nn.Module, case: DynamicAxesCaseCfg, file_path: Path) -> bool:
    """Export model to ONNX format (TorchScript-based, legacy)."""
    try:
        torch.onnx.export(
            model,
            (torch.zeros(case.num_envs, case.num_joints),),
            str(file_path),
            input_names=["joint_pos"],
            output_names=["joint_pos_rel"],
            dynamic_axes={name: {0: "num_envs"} for name in ["joint_pos", "joint_pos_rel"]},
            export_params=True,
            opset_version=13,
            verbose=False,
        )
        print(f"  [ONNX] Export successful: {file_path.name}")
        return True
    except Exception as e:
        print(f"  [ONNX] Export failed: {e}")
        return False


def export_onnx_dynamo(model: torch.nn.Module, case: DynamicAxesCaseCfg, file_path: Path) -> bool:
    """Export model to ONNX format using torch.export (dynamo-based, preferred).

    Uses torch.export.Dim for dynamic_shapes specification instead of dynamic_axes.
    This is the modern approach and should handle dynamic shapes more robustly.
    """
    try:
        # Define symbolic dimension for batch size
        num_envs_dim = torch.export.Dim("num_envs", min=1)

        # dynamic_shapes maps input arg position (tuple) or kwarg name (dict)
        # Format: {arg_name: {dim_index: Dim}} or tuple of {dim_index: Dim}
        dynamic_shapes = ({0: num_envs_dim},)  # First positional arg: joint_pos_input has dynamic dim 0

        torch.onnx.export(
            model,
            (torch.zeros(case.num_envs, case.num_joints),),
            str(file_path),
            input_names=["joint_pos"],
            output_names=["joint_pos_rel"],
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            external_data=False,  # Keep weights in single file for small models
            report=False,
            verify=False,
            opset_version=21,
        )
        print(f"  [ONNX_DYNAMO] Export successful: {file_path.name}")
        return True
    except Exception as e:
        print(f"  [ONNX_DYNAMO] Export failed: {e}")
        return False


def test_onnx(file_path: Path, num_envs: int, num_joints: int) -> bool:
    """Test ONNX model with specified batch size."""
    session = ort.InferenceSession(str(file_path))
    test_input = np.random.randn(num_envs, num_joints).astype(np.float32)
    expected_shape = (num_envs, num_joints)
    try:
        outputs = session.run(None, {"joint_pos": test_input})
        actual_shape = outputs[0].shape
        if actual_shape == expected_shape:
            print(f"    ✓ num_envs={num_envs}: output shape {actual_shape}")
            return True
        else:
            print(f"    ✗ num_envs={num_envs}: shape mismatch! got {actual_shape}, expected {expected_shape}")
            return False
    except Exception as e:
        print(f"    ✗ num_envs={num_envs}: {e}")
        return False


# =============================================================================
# JIT Trace Export & Test
# =============================================================================


def export_jit_trace(model: torch.nn.Module, case: DynamicAxesCaseCfg, file_path: Path) -> bool:
    """Export model via torch.jit.trace (captures concrete shapes)."""
    try:
        example_input = torch.zeros(case.num_envs, case.num_joints)
        traced = torch.jit.trace(model, (example_input,))
        traced.save(str(file_path))
        print(f"  [JIT_TRACE] Export successful: {file_path.name}")
        return True
    except Exception as e:
        print(f"  [JIT_TRACE] Export failed: {e}")
        return False


def test_jit(file_path: Path, num_envs: int, num_joints: int) -> bool:
    """Test JIT model with specified batch size."""
    model = torch.jit.load(str(file_path))
    model.eval()
    test_input = torch.randn(num_envs, num_joints)
    expected_shape = (num_envs, num_joints)
    try:
        with torch.no_grad():
            output = model(test_input)
        actual_shape = tuple(output.shape)
        if actual_shape == expected_shape:
            print(f"    ✓ num_envs={num_envs}: output shape {actual_shape}")
            return True
        else:
            print(f"    ✗ num_envs={num_envs}: shape mismatch! got {actual_shape}, expected {expected_shape}")
            return False
    except Exception as e:
        print(f"    ✗ num_envs={num_envs}: {e}")
        return False


# =============================================================================
# JIT Script Export & Test
# =============================================================================


def export_jit_script(model: torch.nn.Module, file_path: Path) -> bool:
    """Export model via torch.jit.script (preserves control flow)."""
    try:
        scripted = torch.jit.script(model)
        scripted.save(str(file_path))
        print(f"  [JIT_SCRIPT] Export successful: {file_path.name}")
        return True
    except Exception as e:
        print(f"  [JIT_SCRIPT] Export failed: {e}")
        return False


# =============================================================================
# Test Runner
# =============================================================================


@dataclass
class TestResult:
    method: ExportMethod
    case_name: str
    export_success: bool
    inference_results: dict[int, bool] = field(default_factory=dict)


def run_case(case: DynamicAxesCaseCfg, log_dir: Path, test_batch_sizes: list[int]) -> list[TestResult]:
    """Run all export methods for a single case configuration."""
    results = []
    print(f"\n{'='*60}")
    print(f"Case: {case.name}")
    print(f"  env_id type: {type(case.env_id(case.num_envs)).__name__}")
    print(f"  direct_allocation: {case.direct_allocation}")
    print(f"{'='*60}")

    # # ONNX (legacy TorchScript-based)
    # onnx_path = log_dir / f"{case.name}.onnx"
    # model = MockModel(case)
    # result = TestResult(ExportMethod.ONNX, case.name, export_success=False)
    # if export_onnx(model, case, onnx_path):
    #     result.export_success = True
    #     print("  [ONNX] Inference tests:")
    #     for batch_size in test_batch_sizes:
    #         result.inference_results[batch_size] = test_onnx(onnx_path, batch_size, case.num_joints)
    # results.append(result)

    # ONNX Dynamo (torch.export-based)
    onnx_dynamo_path = log_dir / f"{case.name}_dynamo.onnx"
    model = MockModel(case)  # Fresh instance
    result = TestResult(ExportMethod.ONNX_DYNAMO, case.name, export_success=False)
    if export_onnx_dynamo(model, case, onnx_dynamo_path):
        result.export_success = True
        print("  [ONNX_DYNAMO] Inference tests:")
        for batch_size in test_batch_sizes:
            result.inference_results[batch_size] = test_onnx(onnx_dynamo_path, batch_size, case.num_joints)
    results.append(result)

    # # JIT Trace
    # jit_trace_path = log_dir / f"{case.name}_trace.pt"
    # model = MockModel(case)  # Fresh instance
    # result = TestResult(ExportMethod.JIT_TRACE, case.name, export_success=False)
    # if export_jit_trace(model, case, jit_trace_path):
    #     result.export_success = True
    #     print("  [JIT_TRACE] Inference tests:")
    #     for batch_size in test_batch_sizes:
    #         result.inference_results[batch_size] = test_jit(jit_trace_path, batch_size, case.num_joints)
    # results.append(result)

    # # JIT Script (same MockModel)
    # jit_script_path = log_dir / f"{case.name}_script.pt"
    # model = MockModel(case)  # Fresh instance
    # result = TestResult(ExportMethod.JIT_SCRIPT, case.name, export_success=False)
    # if export_jit_script(model, jit_script_path):
    #     result.export_success = True
    #     print("  [JIT_SCRIPT] Inference tests:")
    #     for batch_size in test_batch_sizes:
    #         result.inference_results[batch_size] = test_jit(jit_script_path, batch_size, case.num_joints)
    # results.append(result)

    return results


def print_summary(all_results: list[TestResult], test_batch_sizes: list[int]):
    """Print summary table of all results."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Header
    header = f"{'Case':<30} {'Method':<12} {'Export':<8}"
    for bs in test_batch_sizes:
        header += f" {'BS='+str(bs):<8}"
    print(header)
    print("-" * 80)

    for result in all_results:
        row = f"{result.case_name:<30} {result.method.name:<12} "
        row += "✓" if result.export_success else "✗"
        row += " " * 7
        for bs in test_batch_sizes:
            if bs in result.inference_results:
                row += "✓" if result.inference_results[bs] else "✗"
            else:
                row += "-"
            row += " " * 7
        print(row)


def main():
    configs = AblationCfg()
    log_dir = create_log_dir()
    print(f"Logging to: {log_dir}")

    test_batch_sizes = [1, 16]
    all_results = []

    cases = [
        configs.current,
        configs.sequence,
        configs.sequence_direct_allocation,
        configs.slice,
    ]

    for case in cases:
        # Reset num_envs to original value for each case
        case.num_envs = 1
        results = run_case(case, log_dir, test_batch_sizes)
        all_results.extend(results)

    print_summary(all_results, test_batch_sizes)


if __name__ == "__main__":
    main()
