# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pytest
import torch

from exploy.exporter.core.tensor_proxy import TensorProxy


class TestTensorProxyInit:
    """Test TensorProxy initialization."""

    def test_init_with_tensor(self):
        """Test initialization with a tensor that gets split."""
        tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        proxy = TensorProxy(tensor, split_dim=0)

        assert len(proxy.tensors) == 2
        assert torch.equal(proxy.tensors[0], tensor[0])
        assert torch.equal(proxy.tensors[1], tensor[1])

    def test_init_with_tensor_split_dim_1(self):
        """Test initialization with split on dimension 1."""
        tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        proxy = TensorProxy(tensor, split_dim=1)

        assert len(proxy.tensors) == 2
        assert torch.equal(proxy.tensors[0], tensor[:, 0])
        assert torch.equal(proxy.tensors[1], tensor[:, 1])

    def test_init_with_tensor_split_dim_2(self):
        """Test initialization with split on last dimension."""
        tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        proxy = TensorProxy(tensor, split_dim=2)

        assert len(proxy.tensors) == 3
        assert torch.equal(proxy.tensors[0], tensor[..., 0])
        assert torch.equal(proxy.tensors[1], tensor[..., 1])
        assert torch.equal(proxy.tensors[2], tensor[..., 2])

    def test_init_with_split_dim_too_large(self):
        """Test that initialization fails when split_dim exceeds tensor dimensions."""
        tensor = torch.rand((2, 3, 4))

        # Should fail when split_dim >= tensor.dim()
        with pytest.raises(IndexError):
            TensorProxy(tensor, split_dim=3)

        with pytest.raises(IndexError):
            TensorProxy(tensor, split_dim=5)

    def test_init_with_split_dim_negative(self):
        """Test that initialization fails when split_dim is negative."""
        tensor = torch.rand((2, 3, 4))

        # Should fail when split_dim < 0
        with pytest.raises(IndexError):
            TensorProxy(tensor, split_dim=-1)

    def test_cached_tensor_optimization(self):
        """Test that cached tensors are returned when possible."""
        tensor = torch.rand((2, 3, 4))
        proxy = TensorProxy(tensor, split_dim=1)

        # Direct access to cached tensor
        cached_tensor_0 = proxy.tensors[0]  # Shape: (2, 4) - split dim removed

        # Integer indexing with full slices on other dimensions should return cached tensor
        result_int = proxy[:, 0, :]
        assert result_int is cached_tensor_0, "Integer indexing should return cached tensor"

        # Test that single-element slice returns cached tensor when optimization applies
        result_slice_direct = proxy[:, 0:1, :]  # This should use the optimization path
        assert result_slice_direct is cached_tensor_0, (
            "Single-element slice with full slices should return cached tensor"
        )

        # Test that single-element list returns cached tensor when optimization applies
        result_list_direct = proxy[:, [0], :]  # This should use the optimization path
        assert result_list_direct is cached_tensor_0, (
            "Single-element list with full slices should return cached tensor"
        )

        # Test that tensor indexing returns cached tensor when optimization applies
        result_tensor_direct = proxy[
            :, torch.tensor([0]), :
        ]  # This should use the optimization path
        assert result_tensor_direct is cached_tensor_0, (
            "Single-element tensor index with full slices should return cached tensor"
        )


class TestTensorProxyIndexing:
    """Test TensorProxy indexing operations."""

    def test_single_index_access(self):
        """Test accessing a single element along split dimension."""
        tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
        proxy = TensorProxy(tensor, split_dim=0)

        result = proxy[1]
        expected = tensor[1]
        assert torch.equal(result, expected)

    def test_ellipsis_with_int_index(self):
        """Test indexing with ellipsis and integer on split dimension."""
        tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        proxy = TensorProxy(tensor, split_dim=2)

        # Access the last channel (Z component)
        result = proxy[..., 2]
        expected = tensor[..., 2]
        assert torch.equal(result, expected)

    def test_ellipsis_with_slice(self):
        """Test indexing with ellipsis and slice on split dimension."""
        tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        proxy = TensorProxy(tensor, split_dim=2)

        # Access first two channels
        result = proxy[..., :2]
        expected = tensor[..., :2]
        assert torch.equal(result, expected)

    def test_explicit_indexing_instead_of_ellipsis(self):
        """Test explicit indexing to get Z component (ellipsis not yet supported)."""
        tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        proxy = TensorProxy(tensor, split_dim=2)

        # Access the last channel - use .tensors property
        result = proxy.tensors[2]
        expected = tensor[..., 2]
        assert torch.equal(result, expected)

    def test_multi_dimensional_indexing(self):
        """Test complex multi-dimensional indexing."""
        tensor = torch.rand((3, 4, 5))
        proxy = TensorProxy(tensor, split_dim=1)

        # Index into batch, body, and feature dimensions
        result = proxy[1, 2, 3]
        expected = tensor[1, 2, 3]
        assert torch.equal(result, expected)

    def test_batch_and_split_indexing(self):
        """Test indexing along batch dimension and split dimension."""
        tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        proxy = TensorProxy(tensor, split_dim=2)

        # Index batch 0, all bodies, channel 1
        result = proxy[0, :, 1]
        expected = tensor[0, :, 1]
        assert torch.equal(result, expected)

    def test_slice_on_split_dimension(self):
        """Test slicing on the split dimension."""
        tensor = torch.rand((2, 3, 4))
        proxy = TensorProxy(tensor, split_dim=1)

        # Slice bodies 1:3
        result = proxy[:, 1:3, :]
        expected = tensor[:, 1:3, :]
        assert torch.equal(result, expected)

    def test_list_indexing_on_split_dimension(self):
        """Test list indexing on split dimension."""
        tensor = torch.rand((2, 4, 3))
        proxy = TensorProxy(tensor, split_dim=1)

        # Select specific bodies
        result = proxy[:, [0, 2], :]
        expected = tensor[:, [0, 2], :]
        assert torch.equal(result, expected)

    def test_integer_and_slice_indexing_patterns(self):
        """Test indexing with integer before split_dim and slice on split_dim."""
        t = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
        proxy = TensorProxy(t, split_dim=1)

        # proxy[0, 1:3, :]
        out = proxy[0, 1:3, :]
        assert out.shape == (2, 4)
        assert torch.equal(out, t[0, 1:3, :])

        # proxy[0, :, :]
        out = proxy[0, :, :]
        assert out.shape == (3, 4)
        assert torch.equal(out, t[0, :, :])

        # proxy[1, 0:2, 2]
        out = proxy[1, 0:2, 2]
        assert out.shape == (2,)
        assert torch.equal(out, t[1, 0:2, 2])

        # proxy[:, 1:3, :]
        out = proxy[:, 1:3, :]
        assert out.shape == (2, 2, 4)
        assert torch.equal(out, t[:, 1:3, :])


class TestTensorProxySetItem:
    """Test TensorProxy setitem operations."""

    def test_set_single_element(self):
        """Test setting a single element."""
        tensor = torch.zeros((2, 3, 4))
        proxy = TensorProxy(tensor.clone(), split_dim=1)

        proxy[0, 1, 2] = 99.0
        tensor[0, 1, 2] = 99.0

        assert torch.equal(proxy.to_tensor(), tensor)

    def test_set_with_explicit_indexing(self):
        """Test setting with explicit indexing notation."""
        tensor = torch.zeros((2, 3, 4))
        proxy = TensorProxy(tensor.clone(), split_dim=2)

        # Use explicit indexing instead of ellipsis
        proxy[:, :, 2] = torch.full((2, 3), 5.0)
        tensor[:, :, 2] = 5.0

        assert torch.equal(proxy.to_tensor(), tensor)

    def test_set_slice(self):
        """Test setting a slice."""
        tensor = torch.zeros((2, 4, 3))
        proxy = TensorProxy(tensor.clone(), split_dim=1)

        new_values = torch.ones((2, 2, 3))
        proxy[:, 1:3, :] = new_values
        tensor[:, 1:3, :] = new_values

        assert torch.equal(proxy.to_tensor(), tensor)


class TestTensorProxyToTensor:
    """Test converting TensorProxy back to tensor."""

    def test_to_tensor_split_dim_0(self):
        """Test to_tensor with split on dimension 0."""
        tensor = torch.rand((3, 4, 5))
        proxy = TensorProxy(tensor, split_dim=0)

        result = proxy.to_tensor()
        assert torch.equal(result, tensor)
        assert result.shape == tensor.shape

    def test_to_tensor_split_dim_1(self):
        """Test to_tensor with split on dimension 1."""
        tensor = torch.rand((2, 5, 3))
        proxy = TensorProxy(tensor, split_dim=1)

        result = proxy.to_tensor()
        assert torch.equal(result, tensor)
        assert result.shape == tensor.shape

    def test_to_tensor_split_dim_2(self):
        """Test to_tensor with split on last dimension."""
        tensor = torch.rand((2, 3, 4))
        proxy = TensorProxy(tensor, split_dim=2)

        result = proxy.to_tensor()
        assert torch.equal(result, tensor)
        assert result.shape == tensor.shape


class TestTensorProxyEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_dimension_tensor(self):
        """Test with a tensor that has only one element along split dimension."""
        tensor = torch.rand((1, 2, 3))
        proxy = TensorProxy(tensor, split_dim=0)

        assert len(proxy.tensors) == 1
        result = proxy[0]
        assert torch.equal(result, tensor[0])

    def test_1d_tensors(self):
        """Test with 1D tensors after splitting."""
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        proxy = TensorProxy(tensor, split_dim=0)

        assert proxy.to_tensor().shape == (2, 3)
        assert torch.equal(proxy[0], tensor[0])
        assert torch.equal(proxy[1], tensor[1])

    def test_raycaster_use_case(self):
        """Test typical raycaster use case with XYZ data."""
        # Simulate ray hits with shape (batch=1, num_rays=187, xyz=3)
        ray_hits = torch.rand((1, 187, 3))
        proxy = TensorProxy(ray_hits, split_dim=2)

        # Access Z component (height) with ellipsis
        z_component = proxy[..., 2]
        assert z_component.shape == (1, 187)
        assert torch.equal(z_component, ray_hits[..., 2])

    def test_body_position_use_case(self):
        """Test typical articulation body position use case."""
        # Simulate body positions with shape (batch=1, num_bodies=10, xyz=3)
        body_pos = torch.rand((1, 10, 3))
        proxy = TensorProxy(body_pos, split_dim=1)

        # Access specific body
        body_5 = proxy[:, 5, :]
        assert body_5.shape == (1, 3)
        assert torch.equal(body_5, body_pos[:, 5, :])

        # Access via .tensors property
        body_5_direct = proxy.tensors[5]
        assert torch.equal(body_5_direct, body_pos[:, 5, :])


class TestTensorProxyTorchFunction:
    """Test TensorProxy integration with torch functions."""

    def test_torch_stack(self):
        """Test using TensorProxy with torch.stack."""
        tensor1 = torch.rand((2, 3))
        tensor2 = torch.rand((2, 3))
        proxy1 = TensorProxy(tensor1, split_dim=0)
        proxy2 = TensorProxy(tensor2, split_dim=0)

        result = torch.stack([proxy1, proxy2])
        expected = torch.stack([tensor1, tensor2])
        assert torch.equal(result, expected)

    def test_torch_cat(self):
        """Test using TensorProxy with torch.cat."""
        tensor1 = torch.rand((2, 3))
        tensor2 = torch.rand((2, 3))
        proxy1 = TensorProxy(tensor1, split_dim=0)
        proxy2 = TensorProxy(tensor2, split_dim=0)

        result = torch.cat([proxy1, proxy2], dim=0)
        expected = torch.cat([tensor1, tensor2], dim=0)
        assert torch.equal(result, expected)

    def test_repr(self):
        """Test string representation."""
        tensor = torch.rand((2, 3, 4))
        proxy = TensorProxy(tensor, split_dim=1)

        repr_str = repr(proxy)
        assert "TensorProxy" in repr_str
        assert "shape" in repr_str
        assert "[2, 3, 4]" in repr_str
