"""Tests for core components."""

import numpy as np
import pytest
import torch

from exporter.core.components import Group, Input, Memory, Output


class TestInput:
    """Test Input component."""

    def test_input_creation(self):
        """Test creating an Input component."""
        data = torch.tensor([1.0, 2.0])

        def getter():
            return data

        inp = Input(
            name="test_input",
            get_from_env_cb=getter,
            set_to_env_cb=lambda x: None,
            metadata={"unit": "radians"},
        )

        assert inp.input_name == "test_input"
        assert inp.metadata == {"unit": "radians"}

    def test_input_numpy_conversion(self):
        """Test converting input data to numpy."""
        data = torch.tensor([4.0, 5.0, 6.0])
        inp = Input(
            name="test",
            get_from_env_cb=lambda: data,
            set_to_env_cb=lambda x: None,
        )

        numpy_data = inp.input_data_numpy
        assert isinstance(numpy_data, np.ndarray)
        assert np.array_equal(numpy_data, data.numpy())

    def test_input_read_write(self):
        """Test reading and writing input data."""
        initial_data = torch.tensor([1.0, 2.0, 3.0])
        stored = {"value": initial_data}

        def getter():
            return stored["value"]

        def setter(value):
            stored["value"] = value

        inp = Input(
            name="test",
            get_from_env_cb=getter,
            set_to_env_cb=setter,
        )

        # Modify stored data
        stored["value"] = torch.tensor([7.0, 8.0, 9.0])

        # Write initial data to environment
        inp.write()
        assert torch.equal(stored["value"], initial_data)

        # Read from environment
        stored["value"] = torch.tensor([10.0, 11.0, 12.0])
        inp.read()
        assert torch.equal(inp.input_data, stored["value"])


class TestOutput:
    """Test Output component."""

    def test_output_creation(self):
        """Test creating an Output component."""
        data = torch.tensor([1.0, 2.0])

        def getter():
            return data

        out = Output(
            name="test_output",
            get_from_env_cb=getter,
            metadata={"unit": "radians"},
        )

        assert out.output_name == "test_output"
        assert out.metadata == {"unit": "radians"}

    def test_output_numpy_conversion(self):
        """Test converting output value to numpy."""
        data = torch.tensor([13.0, 14.0, 15.0])

        def getter():
            return data

        out = Output(name="test", get_from_env_cb=getter)

        numpy_value = out.value_numpy
        assert isinstance(numpy_value, np.ndarray)
        assert np.array_equal(numpy_value, data.numpy())


class TestMemory:
    """Test Memory component."""

    def test_memory_creation(self):
        """Test creating a Memory component."""
        data = torch.zeros(2, 10)

        def getter():
            return data

        def setter(value):
            pass

        mem = Memory(
            name="actions",
            get_from_env_cb=getter,
            set_to_env_cb=setter,
        )

        assert mem.input_name == "memory.actions.in"
        assert mem.output_name == "memory.actions.out"
        assert torch.equal(mem.input_data, data)
        assert torch.equal(mem.value, data)

    def test_memory_name_conversion(self):
        """Test Memory name conversion utilities."""
        mem = Memory(
            name="actions",
            get_from_env_cb=lambda: torch.zeros(3),
            set_to_env_cb=lambda x: None,
        )

        # Test io_name_to_name
        assert mem.io_name_to_name("memory.actions.in") == "actions"
        assert mem.io_name_to_name("memory.actions.out") == "actions"

        # Test io_name_to_output_name
        assert mem.io_name_to_output_name("memory.actions.in") == "memory.actions.out"


class TestGroup:
    """Test Group component."""

    def test_group_creation(self):
        """Test creating a Group component."""
        inp1 = Input(
            name="input1",
            get_from_env_cb=lambda: torch.zeros(3),
            set_to_env_cb=lambda x: None,
        )
        out1 = Output(name="output1", get_from_env_cb=lambda: torch.zeros(2))

        group = Group(
            name="test_group",
            items=[inp1, out1],
            metadata={"description": "Test group"},
        )

        assert group.name == "test_group"
        assert len(group.items) == 2
        assert group.metadata == {"description": "Test group"}

    def test_group_item_name_prefixing(self):
        """Test that Group prefixes item names."""
        inp = Input(
            name="joint_pos",
            get_from_env_cb=lambda: torch.zeros(12),
            set_to_env_cb=lambda x: None,
        )
        out = Output(name="joint_targets", get_from_env_cb=lambda: torch.zeros(12))

        group = Group(name="robot", items=[inp, out])

        for item in group.items:
            if isinstance(item, (Input)):
                assert item.input_name.startswith("robot.")
            elif isinstance(item, (Output)):
                assert item.output_name.startswith("robot.")
