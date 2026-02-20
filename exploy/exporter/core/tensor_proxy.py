# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import typing

import torch


class TensorProxy:
    """Manage a list of tensors and expose them to the user as a stacked tensor.

    This class takes a tensor, splits it along a specified dimension, and exposes it as if it were the original tensor.
    For example, this class allows the user to implement the following:

        # Make a tensor of all body positions.
        body_pos = torch.rand((batch_dim, num_bodies, 3))

        # Wrap the tensor in a `TensorProxy` object, splitting along dimension 1 (num_bodies).
        body_pos_proxy = TensorProxy(body_pos, split_dim=1)

    Now, the user can index into `body_pos_proxy` as if it was the original `body_pos`. Indexing
    will index into one of the split tensors created from unbinding along the split dimension.

    One use case for this implementation is to split the `body_state_w` tensor from `ArticulationData` into separate body tensors
    to improve exporting.
    """

    def __init__(self, tensor: torch.Tensor, split_dim: int):
        # Split the tensor into a list of tensors along the split_dim
        # We use unbind to remove the dimension entirely for each element in the list
        self._tensors = list(torch.unbind(tensor, dim=split_dim))
        self._split_dim = split_dim
        self._total_dim = tensor.dim()
        if split_dim < 0:
            raise IndexError(f"split_dim must be non-negative, got {split_dim}")

    def __getitem__(self, idx):
        """Index into a `TensorProxy` as if the user was indexing into the un-split list of tensors."""
        if not isinstance(idx, tuple):
            idx = (idx,)

        # Expand Ellipsis (...) to appropriate number of slice(None) objects
        if Ellipsis in idx:
            ellipsis_idx = idx.index(Ellipsis)
            # Calculate how many dimensions the ellipsis represents
            num_remaining = self._total_dim - len(idx) + 1  # +1 because Ellipsis counts as 1
            expanded_slices = (slice(None),) * num_remaining
            idx = idx[:ellipsis_idx] + expanded_slices + idx[ellipsis_idx + 1 :]

        # Ensure total dimensions
        while len(idx) < self._total_dim:
            idx = idx + (slice(None),)

        # If any integer index appears before split_dim, the effective split_dim is reduced
        # This is covered in test_integer_and_slice_indexing_patterns
        effective_split_dim = self._split_dim
        for i in range(self._split_dim):
            if isinstance(idx[i], int):
                effective_split_dim -= 1
        split_slice = idx[self._split_dim]
        full_index = idx[: self._split_dim] + idx[self._split_dim + 1 :]
        use_cache = all(isinstance(i, slice) and i == slice(None) for i in full_index)

        def _gather_and_cat(indices, full_index):
            """Helper to stack/concatenate sliced tensors along the split dimension, matching PyTorch indexing semantics."""
            selected = [self._tensors[i][full_index] for i in indices]
            # Stack along axis 0, then move axis 0 to effective_split_dim position
            stacked = torch.stack(selected, dim=0)
            if effective_split_dim != 0:
                # Move axis 0 to effective_split_dim
                stacked = stacked.movedim(0, effective_split_dim)
            return stacked

        def _get_cached_tensor(indices):
            """Stack the tensors at the given indices along the split dimension when all non-split indices are full slices."""
            selected = [self._tensors[i] for i in indices]
            stacked = torch.stack(selected, dim=0)
            if effective_split_dim != 0:
                stacked = stacked.movedim(0, effective_split_dim)
            return stacked

        # Handle different split indexing cases
        if isinstance(split_slice, int):
            indices = [split_slice]
            if use_cache:
                return self._tensors[indices[0]]
            return self._tensors[split_slice][full_index]

        elif isinstance(split_slice, slice):
            indices = list(range(*split_slice.indices(len(self._tensors))))
            if use_cache:
                return _get_cached_tensor(indices)
            return _gather_and_cat(indices, full_index)

        elif isinstance(split_slice, (list, tuple, torch.Tensor)):
            if isinstance(split_slice, torch.Tensor):
                split_slice = split_slice.tolist()
            if use_cache:
                return _get_cached_tensor(split_slice)
            return _gather_and_cat(split_slice, full_index)

        else:
            raise TypeError(f"Unsupported index type: {type(split_slice)}")

    def __setitem__(self, idx, value: torch.Tensor):
        """Set into a `TensorProxy` as if the user was indexing into the un-split list of tensors."""
        if not isinstance(idx, tuple):
            idx = (idx,)

        while len(idx) < self._total_dim:
            idx = idx + (slice(None),)

        split_slice = idx[self._split_dim]
        full_index = idx[: self._split_dim] + idx[self._split_dim + 1 :]

        if isinstance(split_slice, int):
            self._tensors[split_slice][full_index] = value
        else:
            indices = (
                range(*split_slice.indices(len(self._tensors)))
                if isinstance(split_slice, slice)
                else split_slice
            )
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
            # value must have same shape as selection along split_dim
            for j, i in enumerate(indices):
                val_idx = value
                if value.shape[self._split_dim] > 1 and len(indices) > 1:
                    val_idx = value.index_select(self._split_dim, torch.tensor([j])).squeeze(
                        self._split_dim
                    )
                self._tensors[i][full_index] = val_idx

    def to_tensor(self) -> torch.Tensor:
        """Convert the data stored in a `TensorProxy` into a `torch.Tensor`."""
        expanded = [t.unsqueeze(self._split_dim) for t in self._tensors]
        return torch.cat(expanded, dim=self._split_dim)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Allow using this class with the torch API.

        Torch provides a mechanism to treat any object as a `torch.Tensor`. This is enabled by implementing
        the `__torch_function__` method for any python class.

        For more details on how to implement `__torch_function__`, see:
            https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-python-api

        For a discussion on a concrete implementation of `__torch_function__`, see:
            https://github.com/docarray/notes/blob/main/blog/02-this-weeks-in-docarray-01.md#__torch_function__--or-how-to-give-pytorch-a-little-bit-more-confidence
        """
        if kwargs is None:
            kwargs = {}

        # Only handle functions that involve TensorProxy or Tensor objects.
        if not all(issubclass(t, torch.Tensor | TensorProxy) for t in types):
            return NotImplemented

        # Convert all `TensorProxy`` elements in `args` to `torch.Tensor` objects.
        new_args = args_to_tensor(args=args)

        # todo: Convert TensorProxy to Tensor in kwargs.

        return (
            func.__wrapped__(*new_args, **kwargs)
            if hasattr(func, "__wrapped__")
            else func(*new_args, **kwargs)
        )

    def __repr__(self):
        return f"TensorProxy(shape={self.to_tensor().shape})"

    @property
    def tensors(self) -> list[torch.Tensor]:
        """Get the list of tensors stored in this `TensorProxy`."""
        return self._tensors


def args_to_tensor(args):
    """Convert each element in a sequence to torch.Tensor, preserving the sequence structure."""
    new_args = []

    for arg in args:
        if isinstance(arg, TensorProxy):
            new_args.append(arg.to_tensor())
        elif isinstance(arg, typing.Sequence):
            new_args.append(
                args_to_tensor(
                    args=arg,
                )
            )
        else:
            new_args.append(arg)

    return tuple(new_args)
