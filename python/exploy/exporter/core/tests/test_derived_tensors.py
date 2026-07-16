# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Tests for materializing derived tensors."""

import dataclasses
import gc

import torch

from exploy.exporter.core.derived_tensors import DerivedTensor
from exploy.exporter.core.tensor_proxy import TensorProxy


class FakeDataSource:
    """Minimal data source holding one stored tensor."""

    def __init__(self):
        self.stored = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)


@dataclasses.dataclass
class UnhashableDataSource:
    """Mimics framework data objects that are eq-dataclasses and therefore unhashable."""

    stored: torch.Tensor = dataclasses.field(
        default_factory=lambda: torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    )


class CountingComputeFn:
    """Compute callback that counts how often it is called."""

    def __init__(self, fn):
        self._fn = fn
        self.num_calls = 0

    def __call__(self, *args, **kwargs):
        self.num_calls += 1
        return self._fn(*args, **kwargs)


def _make_derived(split_dim: int | None = 1) -> tuple[DerivedTensor, CountingComputeFn]:
    compute_fn = CountingComputeFn(lambda d: d.stored * 2.0)
    return DerivedTensor(compute_fn=compute_fn, split_dim=split_dim), compute_fn


class TestDerivedTensor:
    def test_materialize_stores_leaf_and_call_returns_it(self):
        data = FakeDataSource()
        derived, compute_fn = _make_derived()

        stored = derived.materialize(data)

        assert isinstance(stored, TensorProxy)
        # After materialization, calling the DerivedTensor returns the stored object without
        # recomputing (identity is what makes the quantity an ONNX graph boundary).
        assert derived(data) is stored
        assert derived(data)[:, 0] is stored[:, 0]
        assert compute_fn.num_calls == 1
        assert torch.allclose(stored.to_tensor(), data.stored * 2.0)

    def test_materialized_value_is_a_clone(self):
        data = FakeDataSource()
        derived, _ = _make_derived(split_dim=None)

        stored = derived.materialize(data)

        data.stored += 1.0
        assert torch.allclose(stored, (data.stored - 1.0) * 2.0)

    def test_materialized_value_is_detached_from_autograd(self):
        """Materializing a grad-tracking result must not retain the autograd graph in the cache."""
        data = FakeDataSource()
        data.stored.requires_grad_(True)
        derived = DerivedTensor(compute_fn=lambda d: d.stored * 2.0, split_dim=None)

        stored = derived.materialize(data)

        assert stored.grad_fn is None
        assert not stored.requires_grad

    def test_materialize_is_idempotent(self):
        data = FakeDataSource()
        derived, compute_fn = _make_derived()

        first = derived.materialize(data)
        second = derived.materialize(data)

        assert second is first
        assert compute_fn.num_calls == 1

    def test_call_computes_live_when_not_materialized(self):
        data = FakeDataSource()
        derived, compute_fn = _make_derived(split_dim=None)

        value = derived(data)

        assert compute_fn.num_calls == 1
        assert torch.allclose(value, data.stored * 2.0)
        # Without materialization, every call computes a fresh value from live data.
        assert derived(data) is not value
        assert compute_fn.num_calls == 2

    def test_tensor_proxy_result_is_converted_and_cloned(self):
        data = FakeDataSource()
        derived = DerivedTensor(
            compute_fn=lambda d: TensorProxy(d.stored, split_dim=1), split_dim=1
        )

        stored = derived.materialize(data)

        assert isinstance(stored, TensorProxy)
        assert stored[:, 0] is not data.stored
        assert torch.allclose(stored.to_tensor(), data.stored)

    def test_independent_data_objects_do_not_share_materializations(self):
        data_a = FakeDataSource()
        data_b = FakeDataSource()
        derived, compute_fn = _make_derived()

        stored_a = derived.materialize(data_a)

        # data_b is not materialized: calls compute live.
        assert derived(data_b) is not stored_a
        assert compute_fn.num_calls == 2
        stored_b = derived.materialize(data_b)
        assert stored_b is not stored_a

    def test_unhashable_data_object_is_supported(self):
        """Framework data objects can be eq-dataclasses without __hash__ (e.g. mjlab EntityData);
        both materialization and live-compute fall-through must work for them."""
        data = UnhashableDataSource()
        derived, compute_fn = _make_derived()

        # Live compute on an unhashable, unmaterialized object.
        live = derived(data)
        assert torch.allclose(live, data.stored * 2.0)

        stored = derived.materialize(data)
        assert derived(data) is stored
        assert compute_fn.num_calls == 2

    def test_materialization_dies_with_the_data_object(self):
        """Discarding the data object (as cleanup() does with the proxy) unsets the
        materialization: later calls compute live again and no stale entry can be hit."""
        data = FakeDataSource()
        derived, compute_fn = _make_derived()

        derived.materialize(data)
        assert len(derived._materialized) == 1

        del data
        gc.collect()

        # The weakref callback evicted the entry.
        assert len(derived._materialized) == 0

        # A fresh object (potentially reusing the old id) computes live.
        new_data = FakeDataSource()
        value = derived(new_data)
        assert torch.allclose(value, new_data.stored * 2.0)

    def test_multiple_args_and_kwargs_are_forwarded_and_key_the_materialization(self):
        """Compute callbacks can take any signature; materializations are keyed by the identity
        of all call arguments, including non-weakrefable ones like floats and strings."""
        data = FakeDataSource()
        compute_fn = CountingComputeFn(lambda d, scale, offset=0.0: d.stored * scale + offset)
        derived = DerivedTensor(compute_fn=compute_fn, split_dim=None)

        stored = derived.materialize(data, 2.0, offset=1.0)

        assert torch.allclose(stored, data.stored * 2.0 + 1.0)
        assert derived(data, 2.0, offset=1.0) is stored
        assert compute_fn.num_calls == 1

        # Different argument values compute live and do not share the materialization.
        other = derived(data, 3.0, offset=1.0)
        assert other is not stored
        assert torch.allclose(other, data.stored * 3.0 + 1.0)
        assert compute_fn.num_calls == 2

    def test_class_based_compute_callback(self):
        """The compute callback can be a callable class instance constructed with the environment
        (like an IsaacLab ManagerTermBase observation term) instead of a free function."""

        class FakeEnv:
            def __init__(self):
                self.data = FakeDataSource()

        class ObsTerm:
            """Mimics an observation-term class: built with the env, called with (env, params)."""

            def __init__(self, env):
                self._dim = env.data.stored.shape[-1]

            def __call__(self, env, scale=1.0):
                assert env.data.stored.shape[-1] == self._dim
                return env.data.stored * scale

        env = FakeEnv()
        derived = DerivedTensor(compute_fn=ObsTerm(env), split_dim=1)

        stored = derived.materialize(env, scale=2.0)

        assert isinstance(stored, TensorProxy)
        assert derived(env, scale=2.0) is stored
        assert derived(env, scale=2.0)[:, 0] is stored[:, 0]
        assert torch.allclose(stored.to_tensor(), env.data.stored * 2.0)

        # Different call arguments fall through to live computation.
        live = derived(env, scale=3.0)
        assert live is not stored
        assert torch.allclose(live, env.data.stored * 3.0)

    def test_materialization_dies_with_any_weakrefable_argument(self):
        """An entry materialized from several objects is evicted as soon as any of them dies."""
        data_a = FakeDataSource()
        data_b = FakeDataSource()
        derived = DerivedTensor(compute_fn=lambda a, b: a.stored + b.stored, split_dim=None)

        derived.materialize(data_a, data_b)
        assert len(derived._materialized) == 1

        del data_a
        gc.collect()

        assert len(derived._materialized) == 0
        new_data = FakeDataSource()
        value = derived(new_data, data_b)
        assert torch.allclose(value, new_data.stored + data_b.stored)
