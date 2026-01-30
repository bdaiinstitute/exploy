from collections.abc import Callable
from typing import Any

import numpy as np
import torch


class Input:
    """Abstract base class for data handlers"""

    def __init__(
        self,
        name: str,
        get_from_env_cb: Callable[[], torch.Tensor],
        metadata: Any = None,
    ):
        self._name = name
        self._get_from_env_cb = get_from_env_cb
        self._metadata: Any = metadata
        self._data: torch.Tensor = self._get_from_env_cb()

    @property
    def input_data(self) -> torch.Tensor:
        """Get internal data as tensor dict"""
        return self._data

    @property
    def input_data_numpy(self) -> dict[str, np.ndarray]:
        """Get internal data as numpy arrays"""
        return self._data.cpu().numpy()

    @property
    def metadata(self) -> Any:
        """Return metadata about this handler's data"""
        return self._metadata

    @property
    def input_name(self) -> str:
        return self._name

    def read(self) -> None:
        self._data = self._get_from_env_cb()

    @property
    def get_from_env_cb(self) -> Callable[[], torch.Tensor]:
        return self._get_from_env_cb


class Output:
    """Abstract interface for action handlers"""

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
        """Get action handler metadata"""
        return self._metadata

    @property
    def output_name(self) -> str:
        return self._name

    @property
    def value(self) -> torch.Tensor:
        return self._get_from_env_cb()

    @property
    def value_numpy(self) -> torch.Tensor:
        return self._get_from_env_cb().cpu().numpy()

    @property
    def get_from_env_cb(self) -> Callable[[], torch.Tensor]:
        return self._get_from_env_cb


class Memory(Input, Output):
    """Handle memory inputs and outputs.

    This class abstracts how to get and set values used in an environment
    that have memory, for example actions and previous actions. Values are retrieved
    by passing callables.
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
        """A list of names, formatted for use with the ONNX exporter inputs."""
        return f"memory.{self._name}.in"

    @property
    def output_name(self) -> str:
        """A list of names used for the ONNX exporter outputs."""
        return f"memory.{self._name}.out"

    def io_name_to_name(self, io_name: str) -> str:
        """Helper function to convert a name formatted for inputs or outputs to a memory element name."""
        return io_name.removeprefix("memory.").removesuffix(".in").removesuffix(".out")

    def io_name_to_output_name(self, io_name: str) -> str:
        """Helper function to convert a name formatted for inputs or outputs to the corresponding outputs to a memory element name."""
        return io_name.removesuffix(".in") + ".out"


class Group:
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
        return self._name

    @property
    def items(self) -> list[Input | Output]:
        return self._items

    @property
    def metadata(self) -> Any:
        return self._metadata


class Connection:
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
        return self._name

    def write(self) -> None:
        if self._setter is not None:
            self._setter(self._getter())
