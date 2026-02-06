from typing import Any

import numpy as np
import torch

from exporter.components import Connection, Group, Input, Memory, Output


class ContextManager:
    """Manages all components (inputs, outputs, memory) for an environment export."""

    def __init__(
        self,
    ):
        self._components: list[Input | Output | Connection] = []
        self._groups: list[Group] = []

    def add_component(self, component: Input | Output | Connection) -> None:
        if isinstance(component, Input):
            self.assert_unique_name(component.input_name)
        elif isinstance(component, Output):
            self.assert_unique_name(component.output_name)

        self._components.append(component)

    def add_group(self, group: Group) -> None:
        self.assert_unique_name(group.name)

        for item in group.items:
            if isinstance(item, Group):
                self.add_group(item)
            else:
                self.add_component(item)
        self._groups.append(group)

    def get_input_components(self) -> list[Input]:
        return [comp for comp in self._components if isinstance(comp, Input | Memory)]

    def get_connection_components(self) -> list[Connection]:
        return [comp for comp in self._components if isinstance(comp, Connection)]

    def get_output_components(self) -> list[Output]:
        return [comp for comp in self._components if isinstance(comp, Output | Memory)]

    def get_memory_components(self) -> list[Memory]:
        return [comp for comp in self._components if isinstance(comp, Memory)]

    def get_component_by_name(self, name: str) -> Input | Output | None:
        for component in self._components:
            if isinstance(component, Output) and component.output_name == name:
                return component
            if isinstance(component, Input) and component.input_name == name:
                return component
        return None

    def read_inputs(self) -> None:
        for component in self.get_input_components():
            component.read()

    def write_connections(self) -> None:
        for component in self.get_connection_components():
            component.write()

    def get_inputs(self, to_numpy: bool = False) -> dict[str, torch.Tensor] | dict[str, np.ndarray]:
        inputs = {}
        for component in self.get_input_components():
            inputs[component.input_name] = (
                component.input_data_numpy if to_numpy else component.input_data
            )
        return inputs

    def get_input_names(self) -> list[str]:
        return [component.input_name for component in self.get_input_components()]

    def get_outputs(self, to_numpy: bool = False) -> dict[str, torch.Tensor]:
        outputs = {}
        for component in self.get_output_components():
            outputs[component.output_name] = component.value_numpy if to_numpy else component.value
        return outputs

    def get_output_names(self) -> list[str]:
        return [output.output_name for output in self.get_output_components()]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **{
                comp.input_name: comp.metadata
                for comp in self._components
                if isinstance(comp, Input) and comp.metadata is not None
            },
            **{
                comp.output_name: comp.metadata
                for comp in self._components
                if isinstance(comp, Output) and comp.metadata is not None
            },
            **{group.name: group.metadata for group in self._groups if group.metadata is not None},
        }

    def assert_unique_name(self, name: str) -> bool:
        assert name not in [g.name for g in self._groups]
        assert name not in [comp.input_name for comp in self.get_input_components()]
        assert name not in [comp.output_name for comp in self.get_output_components()]
        return True
