# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from collections.abc import Callable
from typing import Any

import numpy as np
import torch


class Input:
    """Abstraction for controller inputs.

    The input encapsulates a callback that retrieves data from the environment
    and provides it as a tensor for use in the generation of a computational graph.
    """

    def __init__(
        self,
        name: str,
        get_from_env_cb: Callable[[], torch.Tensor],
        metadata: Any = None,
    ):
        """Constuct an Input.

        Args:
            name (str): Identifier for this input.
            get_from_env_cb (Callable[[], torch.Tensor]): Callback function that retrieves the input
                from the environment as a torch.Tensor.
            metadata (Any): Optional metadata associated with this input (e.g., shape, data type,
                semantic information).
        """
        self._name = name
        self._get_from_env_cb = get_from_env_cb
        self._metadata: Any = metadata
        self._data: torch.Tensor = self._get_from_env_cb()
        self._id = id(self._data)

    @property
    def input_data(self) -> torch.Tensor:
        """Get internal data as a torch tensor."""
        return self._data

    @property
    def input_data_numpy(self) -> dict[str, np.ndarray]:
        """Get internal data as a numpy array."""
        return self._data.cpu().numpy()

    @property
    def metadata(self) -> Any:
        """Return metadata about this handler's data"""
        return self._metadata

    @property
    def input_name(self) -> str:
        """Return the name of this input."""
        return self._name

    @property
    def id(self) -> int:
        """Return the unique identifier of this input."""
        return self._id

    def read(self) -> None:
        """Get the latest data from the environment by calling the callback."""
        self._data = self._get_from_env_cb().clone()

    @property
    def get_from_env_cb(self) -> Callable[[], torch.Tensor]:
        """Get the callback function that retrieves the input from the environment."""
        return self._get_from_env_cb


class Output:
    """Abstract interface for outputs."""

    def __init__(
        self,
        name: str,
        get_from_env_cb: Callable[[], torch.Tensor],
        metadata: Any = None,
    ):
        self._name = name
        self._get_from_env_cb = get_from_env_cb
        self._metadata: Any = metadata

    @property
    def metadata(self) -> Any:
        """Get metadata about this output."""
        return self._metadata

    @property
    def output_name(self) -> str:
        """Return the name of this output."""
        return self._name

    @property
    def value(self) -> torch.Tensor:
        """Get the latest value from the environment by calling the callback."""
        return self._get_from_env_cb()

    @property
    def value_numpy(self) -> torch.Tensor:
        """Get the latest value from the environment as a numpy array by calling the callback."""
        return self._get_from_env_cb().cpu().numpy()

    @property
    def get_from_env_cb(self) -> Callable[[], torch.Tensor]:
        """Get the callback function that retrieves the output from the environment."""
        return self._get_from_env_cb


class Memory(Input, Output):
    """Handle memory inputs and outputs.

    This class abstracts how to get and set values used in an environment that has memory, for
    example actions and previous actions. Values are retrieved by passing callables.
    """

    def __init__(
        self,
        name: str,
        get_from_env_cb: Callable[[], torch.Tensor],
    ):
        Input.__init__(
            self,
            name=name,
            get_from_env_cb=get_from_env_cb,
        )
        Output.__init__(
            self,
            name=name,
            get_from_env_cb=get_from_env_cb,
        )
        self._memory_info = {}

    @property
    def input_name(self) -> str:
        """This component's name, formatted for use as an ONNX exporter input."""
        return f"memory.{self._name}.in"

    @property
    def output_name(self) -> str:
        """This component's name, formatted for use as an ONNX exporter output."""
        return f"memory.{self._name}.out"

    def io_name_to_name(self, io_name: str) -> str:
        """Helper function to convert a name formatted for inputs or outputs to a memory element name."""
        return io_name.removeprefix("memory.").removesuffix(".in").removesuffix(".out")

    def io_name_to_output_name(self, io_name: str) -> str:
        """Helper function to convert a name formatted for inputs or outputs to the corresponding outputs to a memory element name."""
        return io_name.removesuffix(".in") + ".out"


class Group:
    """Abstraction for grouping related inputs and outputs together."""

    def __init__(
        self,
        name: str,
        items: list[Input | Output],
        metadata: Any = None,
    ):
        self._metadata = metadata
        self._name = name
        self._items = items
        for item in self._items:
            item._name = f"{self._name}.{item._name}"

    @property
    def name(self) -> str:
        """Return the name of this group."""
        return self._name

    @property
    def items(self) -> list[Input | Output]:
        """Get the items in this group."""
        return self._items

    @property
    def metadata(self) -> Any:
        """Get metadata about this group."""
        return self._metadata


class Connection:
    """Abstraction for connecting existing inputs to data sources."""

    def __init__(
        self,
        name: str,
        getter: callable,
        setter: callable,
    ):
        self._name = name
        self._getter = getter
        self._setter = setter

    @property
    def name(self) -> str:
        """The name of this connection."""
        return self._name

    def write(self) -> None:
        """Write data to the environment by calling the setter with the value from the getter."""
        self._setter(self._getter())
