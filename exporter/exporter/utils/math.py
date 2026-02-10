# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import torch


def compare_tensors(
    vec_a: torch.tensor,
    vec_b: torch.tensor,
    vec_name: str,
    index_names: list[str] = None,
    name_a: str = None,
    name_b: str = None,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
    verbose: bool = False,
) -> bool:
    """Compare two tensors, and print a message showing the differences.

    Args:
        vec_a: One of the tensors being compared.
        vec_b: One of the tensors being compared.
        vec_name: A name identifying the compared tensors.
        index_names: A list of names, one for each element of the input tensors.
        name_a: A name identifying the source of the first input tensor.
        name_a: A name identifying the source of the second input tensor.
        atol : Absolute tolerance.
        rtol : Relative tolerance.
        verbose: If true, print information about the tensor difference to console.
    """
    is_close = torch.isclose(vec_a, vec_b, atol=atol, rtol=rtol)

    msg = f"Comparing {vec_name}: "
    if torch.all(is_close):
        msg += "\033[92mall elements are close.\033[0m"
        print(msg)
        return True

    mismatched_indices = (torch.logical_not(is_close)).nonzero(as_tuple=False)

    msg += "\033[91mfound mismatch.\033[0m"
    name_a = "a" if name_a is None else name_a
    name_b = "b" if name_b is None else name_b
    for idx in mismatched_indices:
        idx_tuple = tuple(idx.tolist())
        val_a = vec_a[idx_tuple].item()
        val_b = vec_b[idx_tuple].item()
        val_name = f"[{index_names[idx_tuple[1]]}]" if index_names is not None else ""
        msg += (
            f"\n\t{idx_tuple}{val_name}:\t"
            f"{name_a}={val_a:+.5f}\t"
            f"{name_b}={val_b:+.5f}\t"
            f"Δ={np.abs(val_a - val_b):.5f}"
        )

    if verbose:
        print(msg)
    return False
