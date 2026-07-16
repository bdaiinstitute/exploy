# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import isaaclab.utils.math as math_utils
import torch

from exploy.exporter.core.derived_tensors import DerivedTensor
from exploy.exporter.core.tensor_proxy import to_tensor


def _compute_body_link_lin_vel_b(data) -> torch.Tensor:
    return math_utils.quat_apply_inverse(
        to_tensor(data.body_link_quat_w), to_tensor(data.body_link_lin_vel_w)
    )


def _compute_body_link_ang_vel_b(data) -> torch.Tensor:
    return math_utils.quat_apply_inverse(
        to_tensor(data.body_link_quat_w), to_tensor(data.body_link_ang_vel_w)
    )


# Body link velocities rt the world frame, expressed in each body's own frame, with shape
# (num_instances, num_bodies, 3). Stock Isaac Lab only exposes per-body velocities in the world
# frame, so these are derived from the world-frame link velocities and the body orientations, and
# materialized on the `ArticulationDataSource` during export (see `add_body_vel`).
body_link_lin_vel_b = DerivedTensor(compute_fn=_compute_body_link_lin_vel_b, split_dim=1)
body_link_ang_vel_b = DerivedTensor(compute_fn=_compute_body_link_ang_vel_b, split_dim=1)
