# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import typing

import torch


class TensorProxy:
    """Manage a list of tensors and expose them to the user as a stacked tensor.

    This class manages a `list[torch.Tensor]` and exposes it as a single tensor. For example, this class allows the user
    to implement the following:

        # Make a tensor of all body positions.
        body_pos = torch.rand((batch_dim, num_bodies, 3))

        # Split all body positions and store into a list of tensors.
        all_body_pos_list = [body_pos[:, id, :].clone() for id in range(num_bodies)]

        # Wrap the list in a `TensorProxy` object.
        body_pos_proxy = TensorProxy(tensors=all_body_pos_list, split_dim=1,)

    Now, the user can index into `body_pos_proxy` as it if was the original `body_pos`. Indexing will index into
    one of the split tensors in `all_body_pos_list`.

    The original use case for this implementation was to split the `body_state_w` tensor from `ArticulationData` into separate body tensors
    to improve exporting.

    Note:   The tensors in the `tensors` parameter to this class do not need to have a shape set as `(batch_dim, num_bodies, 3)`.
            The tensors passed stored in that list can have any number of dimensions.
    """

    def __init__(self, tensors: list[torch.Tensor], split_dim: int):
        self._tensors = tensors

        # todo assert all tensors same shape
        self._total_dim = len(tensors[0].shape) + 1

        self._split_dim = split_dim
        assert self._split_dim < self._total_dim and self._split_dim >= 0

    @property
    def device(self) -> torch.device:
        # TODO: get correct device.
        return torch.device("cpu")

    def __getitem__(self, idx):
        """Index into a `TensorProxy` as if the user was indexing into the un-split list of tensors."""
        if not isinstance(idx, tuple):
            idx = (idx,)

        # Single-element shortcut (like proxy[batch_idx])
        if len(idx) == 1:
            batch_idx = idx[0]
            selected = [t[batch_idx] for t in self._tensors]
            return torch.stack(selected, dim=0)

        # Ensure total dimensions
        while len(idx) < self._total_dim:
            idx = idx + (slice(None),)

        split_slice = idx[self._split_dim]
        full_index = idx[: self._split_dim] + idx[self._split_dim + 1 :]

        # Handle different split indexing cases
        if isinstance(split_slice, int):
            return self._tensors[split_slice][full_index]

        elif isinstance(split_slice, slice):
            indices = range(*split_slice.indices(len(self._tensors)))
            selected = [self._tensors[i][full_index].unsqueeze(self._split_dim) for i in indices]
            return torch.cat(selected, dim=self._split_dim)

        elif isinstance(split_slice, list | tuple | torch.Tensor):
            if isinstance(split_slice, torch.Tensor):
                split_slice = split_slice.tolist()
            selected = [
                self._tensors[i][full_index].unsqueeze(self._split_dim) for i in split_slice
            ]
            return torch.cat(selected, dim=self._split_dim)

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
