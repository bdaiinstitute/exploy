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
    def data(self) -> torch.Tensor:
        """Get internal data as a torch tensor."""
        return self._data

    @property
    def data_numpy(self) -> np.ndarray:
        """Get internal data as a numpy array."""
        return self._data.cpu().numpy()

    @property
    def metadata(self) -> Any:
        """Return metadata about this handler's data"""
        return self._metadata

    @property
    def name(self) -> str:
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
    def name(self) -> str:
        """Return the name of this output."""
        return self._name

    @property
    def value(self) -> torch.Tensor:
        """Get the latest value from the environment by calling the callback."""
        return self._get_from_env_cb()

    @property
    def value_numpy(self) -> np.ndarray:
        """Get the latest value from the environment as a numpy array by calling the callback."""
        return self._get_from_env_cb().cpu().numpy()

    @property
    def get_from_env_cb(self) -> Callable[[], torch.Tensor]:
        """Get the callback function that retrieves the output from the environment."""
        return self._get_from_env_cb


class Memory:
    """Handle memory inputs and outputs.

    This class abstracts how to get and set values used in an environment that has memory, for
    example actions and previous actions. Values are retrieved by passing callables.

    A memory element pairs an `Input` and an `Output` that share the same environment callback:
    the output of one inference step is fed back as the input of the next one.
    """

    def __init__(
        self,
        name: str,
        get_from_env_cb: Callable[[], torch.Tensor],
        metadata: Any = None,
    ):
        """Construct a Memory element.

        Args:
            name (str): Identifier for this memory element.
            get_from_env_cb (Callable[[], torch.Tensor]): Callback function that retrieves the
                memory value from the environment as a torch.Tensor.
            metadata (Any): Optional metadata associated with this memory element. It is stored on
                the memory element itself, under the name shared by its input and output.
        """
        self._metadata = metadata
        self._name = f"memory.{name}"
        self._input = Input(
            name=f"{self._name}.in",
            get_from_env_cb=get_from_env_cb,
        )
        self._output = Output(
            name=f"{self._name}.out",
            get_from_env_cb=get_from_env_cb,
        )

    @property
    def name(self) -> str:
        """Return the name shared by this memory element's input and output."""
        return self._name

    @property
    def metadata(self) -> Any:
        """Get metadata about this memory element."""
        return self._metadata

    @property
    def input(self) -> Input:
        """Return the input side of this memory element."""
        return self._input

    @property
    def output(self) -> Output:
        """Return the output side of this memory element."""
        return self._output


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
