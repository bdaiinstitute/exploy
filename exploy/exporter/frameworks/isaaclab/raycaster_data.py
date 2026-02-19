# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster, RayCasterData

from exploy.exporter.core.tensor_proxy import TensorProxy


class RayCasterDataSource:
    """Mimic the interface of RayCasterData, but manage its own tensor data.

    This class is an adaptor for RayCasterData, mimicking its interface. However,
    it computes pos_w dynamically from the articulation's base position.
    """

    def __init__(self, sensor: RayCaster, articulation: Articulation):
        sensor_data: RayCasterData = sensor.data
        self._data = sensor_data
        self._articulation = articulation
        body_expr = sensor.cfg.prim_path.split("/")[-1]
        self._body_id = articulation.find_bodies(body_expr)[0][0]

        # Split ray_hits_w by last dimension (XYZ) using TensorProxy
        ray_hits = sensor_data.ray_hits_w
        self._ray_hits_w = TensorProxy(ray_hits, split_dim=ray_hits.ndim - 1)

    @property
    def pos_w(self) -> torch.Tensor:
        """Ray origin positions in world frame, computed from articulation base."""
        return self._articulation.data.body_pos_w[:, self._body_id]

    @property
    def ray_hits_w(self) -> torch.Tensor:
        """Ray hit positions in world frame."""
        return self._ray_hits_w

    def update(self, *args, **kwargs):
        """Empty update call to implement the interface of RayCasterData."""
        pass
