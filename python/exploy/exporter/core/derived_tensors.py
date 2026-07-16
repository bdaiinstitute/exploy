# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import weakref
from collections.abc import Callable

import torch

from exploy.exporter.core.tensor_proxy import TensorProxy, to_tensor

# Materialization key: identities of all positional arguments and of all (sorted) keyword
# arguments a value was materialized with.
_Key = tuple[tuple[int, ...], tuple[tuple[str, int], ...]]


def _make_key(args: tuple, kwargs: dict) -> _Key:
    return (
        tuple(id(arg) for arg in args),
        tuple((name, id(value)) for name, value in sorted(kwargs.items())),
    )


def _argument_values(args: tuple, kwargs: dict) -> tuple:
    return (*args, *(value for _, value in sorted(kwargs.items())))


class DerivedTensor:
    """A derived quantity that can be materialized as an ONNX-traceable leaf tensor.

    The ONNX exporter wires graph inputs by object identity: an ``Input`` becomes a graph input
    only if the exact tensor object it holds is read during tracing. A quantity that is *computed*
    on access produces a fresh tensor per call and can never become a graph input. Materializing a
    derived quantity computes it once and stores the clone, so subsequent reads return the stored
    object and inputs bound to it (or to its per-body slices when stored as a
    :class:`TensorProxy`) are traced as ONNX graph inputs, cutting the graph at the derived
    quantity.

    Materialized values are stored inside this object, keyed by the identity of all arguments
    they were computed from (typically a single data-source proxy). This keeps the data-source
    proxies pure framework adaptors (no dynamic attributes) and gives the stored values the same
    lifecycle as the export data-source swap: materialize with the data-source proxy during
    export, and once the proxy is discarded (after ``cleanup()`` restores the framework's native
    data) calls fall through to computing the quantity live.

    Bundling the compute function and split dimension in one object guarantees that
    materialization (during export) and live computation (during training or evaluation) can
    never use mismatched functions.

    Typical usage: define the quantity once at module level, materialize it on the data-source
    proxy when registering inputs, and call it wherever the value is needed::

        body_link_lin_vel_b = DerivedTensor(compute_fn=..., split_dim=1)

        # Input registration (export script):
        body_link_lin_vel_b.materialize(entity.data)
        Input(..., get_from_env_cb=lambda: body_link_lin_vel_b(entity.data)[:, idx])

        # Observation term (training and export):
        def obs_term(env, asset_cfg):
            return body_link_lin_vel_b(env.scene[asset_cfg.name].data)[:, asset_cfg.body_ids]

    The compute function may take any signature — including that of a class-based observation
    term — since :meth:`materialize` and :meth:`__call__` forward ``*args`` and ``**kwargs`` to
    it unchanged.
    """

    def __init__(
        self,
        compute_fn: Callable[..., torch.Tensor],
        split_dim: int | None = None,
    ):
        """Construct a DerivedTensor.

        Args:
            compute_fn: Callback that computes the quantity. Any callable works — a free
                function, a bound method, or a callable class instance; ``materialize`` and
                ``__call__`` forward their arguments to it unchanged.
            split_dim: If given, materialized values are wrapped in a ``TensorProxy`` split along
                this dimension so that each slice is an independent leaf (e.g. one ONNX input per
                body).
        """
        self._compute_fn = compute_fn
        self._split_dim = split_dim
        # Materialized values keyed by the id() of every call argument, because framework data
        # objects are not necessarily hashable (e.g. eq-dataclasses). Each entry stores one
        # reference per argument — a weak reference where the argument supports it (guarding
        # against id reuse after garbage collection and auto-evicting the entry when the argument
        # dies), a strong reference otherwise (e.g. ints or strings, which cannot be weakly
        # referenced and are harmless to hold).
        self._materialized: dict[_Key, tuple[tuple, torch.Tensor | TensorProxy]] = {}

    def _get_materialized(self, args: tuple, kwargs: dict) -> torch.Tensor | TensorProxy | None:
        """Return the value materialized for these arguments, or None if there is none."""
        entry = self._materialized.get(_make_key(args, kwargs))
        if entry is None:
            return None
        refs, stored = entry
        for ref, value in zip(refs, _argument_values(args, kwargs), strict=True):
            target = ref() if isinstance(ref, weakref.ref) else ref
            if target is not value:
                return None
        return stored

    def materialize(self, *args, **kwargs) -> torch.Tensor | TensorProxy:
        """Compute this quantity once and store it as a leaf tensor.

        Subsequent calls of this ``DerivedTensor`` with the same arguments return the stored
        value, preserving the object identity required for ONNX tracing. The stored value lives
        as long as the arguments do. Materializing the same arguments again returns the already
        stored value without recomputing.

        Only materialize export data-source proxies: their state is a frozen snapshot, so the
        stored value stays consistent with them. Materializing live framework data would freeze
        the quantity while the underlying state keeps changing.

        Args:
            *args: Positional arguments forwarded to the compute function, typically an exploy
                data-source proxy.
            **kwargs: Keyword arguments forwarded to the compute function.

        Returns:
            The stored tensor, or ``TensorProxy`` if ``split_dim`` was given.
        """
        stored = self._get_materialized(args, kwargs)
        if stored is not None:
            return stored

        value = to_tensor(self._compute_fn(*args, **kwargs)).detach().clone()
        stored = (
            TensorProxy(value, split_dim=self._split_dim) if self._split_dim is not None else value
        )

        key = _make_key(args, kwargs)
        refs = []
        for argument in _argument_values(args, kwargs):
            try:
                refs.append(
                    weakref.ref(argument, lambda _, key=key: self._materialized.pop(key, None))
                )
            except TypeError:
                refs.append(argument)
        self._materialized[key] = (tuple(refs), stored)
        return stored

    def __call__(self, *args, **kwargs) -> torch.Tensor | TensorProxy:
        """Return the value materialized for these arguments if present, else compute it live.

        This lets the same accessor work both during export, where the quantity has been
        materialized on the data-source proxy (returning the stored leaf preserves the object
        identity required for ONNX tracing), and outside of export (e.g. evaluation or training),
        where the quantity is computed from live state instead.

        Args:
            *args: Positional arguments forwarded to the compute function, typically an exploy
                data-source proxy or native framework data.
            **kwargs: Keyword arguments forwarded to the compute function.

        Returns:
            The stored tensor/``TensorProxy`` if materialized for these arguments, else
            ``compute_fn(*args, **kwargs)``.
        """
        stored = self._get_materialized(args, kwargs)
        if stored is not None:
            return stored
        return self._compute_fn(*args, **kwargs)
