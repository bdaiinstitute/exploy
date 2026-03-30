# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Tests for exporting environments to ONNX and evaluating them using the exported ONNX graph.

The tests implemented in this file create a simple environment with a few inputs, outputs, and a
simple dynamics update. The environment is exported to ONNX using the OnnxEnvironmentExporter, and
then evaluated using the exported ONNX graph.
The evaluation checks that the exported ONNX graph produces the same outputs as the original
environment when given the same inputs, and that the environment and ONNX wrapper stay in sync
across environment resets.

The tests also check that torch Modules used in the environment are correctly included in the
exported ONNX graph.
"""

import pathlib
import tempfile
from collections.abc import Callable

import torch

from exploy.exporter.core.actor import ExportableActor, add_actor_memory
from exploy.exporter.core.context_manager import Group, Input, Memory, Output
from exploy.exporter.core.evaluator import evaluate
from exploy.exporter.core.exportable_environment import ExportableEnvironment
from exploy.exporter.core.exporter import export_environment_as_onnx
from exploy.exporter.core.session_wrapper import SessionWrapper


class DataSource:
    """A simple data source that provides three tensors to compute observations for an environment.

    This data source implements a step function that updates the tensors to emulate changing
    environment state across steps.
    """

    def __init__(self):
        self._init_foo = torch.Tensor([[1.0, 2.0, 3.0, 4.0]])
        self._init_bar = torch.Tensor([[0.5, 0.6]])
        self._init_baz = torch.Tensor([[-7.0, -8.0]])

        self.foo = self._init_foo.clone()
        self.bar = self._init_bar.clone()
        self.baz = self._init_baz.clone()

    def reset(self):
        """Reset the data source to its initial state."""
        self.foo[:] = self._init_foo
        self.bar[:] = self._init_bar
        self.baz[:] = self._init_baz

    def step(self):
        """Update the data source to emulate changing environment state across steps."""
        self.foo[:] += 0.1
        self.bar[:] += 0.2
        self.baz[:] += 0.3


class Environment:
    """Emulate a simple Reinforcement Learning environment, holding its own data sources,
    observations, and actions.
    """

    def __init__(self, data_source: DataSource):
        super().__init__()

        self._data_source = data_source
        self._decimation = 4
        self._actions = torch.zeros(1, 2)
        self._processed_action = torch.zeros_like(self._actions)
        self._output = torch.zeros_like(self._actions)

        self._post_substep_callbacks = {}

        self._reset_after_steps = 10
        self._step_count = 0

    def add_post_substep_callback(self, name: str, cb: Callable[[], None]):
        self._post_substep_callbacks[name] = cb

    @property
    def num_act(self) -> int:
        return self._actions.shape[-1]

    @property
    def num_obs(self) -> int:
        return self.compute_obs().shape[-1]

    @property
    def data_source(self):
        return self._data_source

    def compute_obs(self) -> torch.Tensor:
        """Compute the observations for the environment.

        Returns:
            torch.Tensor: The computed observations.
        """
        return torch.cat(
            [
                self.data_source.foo + 1.0,
                self.data_source.bar + 2.0 * self.data_source.baz,
                self.data_source.baz,
                self._actions,
            ],
            dim=-1,
        )

    def process_actions(self, actions: torch.Tensor):
        """Process the actions.

        Args:
            actions (torch.Tensor): The actions to be processed.
        """
        self._actions[:] = actions
        self._processed_action[:] = 3 * actions

    def apply_actions(self):
        """Apply the processed actions to the environment."""
        self._output[:] = self._processed_action + 2

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Perform a step in the environment.

        Args:
            actions (torch.Tensor): The actions to be applied.

        Returns:
            tuple[torch.Tensor, bool]: A tuple containing the observations and a boolean indicating
            whether the environment was reset.
        """
        done = False

        # Process the actions.
        self.process_actions(actions)

        # Emulate physics update.
        for _ in range(self._decimation):
            self.apply_actions()
            self.data_source.step()
            for cb in self._post_substep_callbacks.values():
                cb()

        self._step_count += 1

        # Check for resets.
        if self._step_count >= self._reset_after_steps:
            self._step_count = 0
            self._data_source.reset()
            self._actions[:] = torch.zeros_like(self._actions)
            done = True

        # Compute observations.
        obs = self.compute_obs()

        # Return observations and done flag.
        return obs, done


class EnvironmentWithTorchModule(Environment):
    """A simple environment that uses a torch Module to compute observations. The module is added
    to the export context, so it should be included in the exported ONNX graph.
    """

    def __init__(
        self,
        data_source: DataSource,
    ):
        module_dim_in = data_source.foo.shape[-1]
        module_dim_out = module_dim_in
        self._module = torch.nn.Linear(module_dim_in, module_dim_out)
        super().__init__(data_source=data_source)

    @property
    def module(self) -> torch.nn.Module:
        return self._module

    def compute_obs(self) -> torch.Tensor:
        """Compute observations using a torch Module. This will test that the module is correctly
        included in the exported ONNX graph and that its parameters are correctly exported and used
        in the ONNX graph.
        """
        return torch.cat(
            [
                self._module(self.data_source.foo) + 1.0,
                self.data_source.bar + 2.0 * self.data_source.baz,
                self.data_source.baz,
                self._actions,
            ],
            dim=-1,
        )


class PostSubstepUpdater:
    """A class used to inject evaluation callbacks at the end of each substep during environment
    stepping, to test that the ONNX graph stays in sync with the environment across substeps.
    """

    def __init__(self, evaluate_substep: Callable[[int], None], update: Callable[[], None]):
        self.sub_step_ctr = 0
        self._evaluate_substep = evaluate_substep
        self._update = update

    def __call__(self, *args, **kwargs):
        self._evaluate_substep(self.sub_step_ctr)
        self.sub_step_ctr += 1
        self._update()


class ExportableEnv(ExportableEnvironment):
    """A wrapper for an environment that makes it exportable to ONNX."""

    def __init__(self, env: Environment):
        super().__init__()
        self._env = env

    @property
    def env(self) -> Environment:
        return self._env

    def prepare_export(self):
        pass

    def empty_actor_observations(self) -> torch.Tensor:
        return torch.zeros_like(self.env.compute_obs())

    def empty_actions(self) -> torch.Tensor:
        return torch.zeros_like(self.env._actions)

    def get_observation_names(self) -> list[str]:
        obs1_names = [f"foo_{i}" for i in range(self.env.data_source.foo.shape[-1])]
        obs2_names = [f"bar_{i}" for i in range(self.env.data_source.bar.shape[-1])]
        obs3_names = [f"baz_{i}" for i in range(self.env.data_source.baz.shape[-1])]
        obs4_names = [f"actions_{i}" for i in range(self.env._actions.shape[-1])]
        return obs1_names + obs2_names + obs3_names + obs4_names

    def observations_reset(self) -> torch.Tensor:
        return self.env.compute_obs()

    def register_evaluation_hooks(self, update, evaluate_substep):
        self._env.add_post_substep_callback(
            name="onnx_evaluator_callback",
            cb=PostSubstepUpdater(
                evaluate_substep=evaluate_substep,
                update=update,
            ),
        )

    def metadata(self) -> dict:
        return {
            "env_name": "Env",
            "version": "1.0",
        }

    @property
    def decimation(self) -> int:
        return self.env._decimation

    def compute_observations(self) -> torch.Tensor:
        return self.env.compute_obs()

    def apply_actions(self) -> torch.Tensor:
        return self.env.apply_actions()

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.env.process_actions(actions)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, bool]:
        self._env._post_substep_callbacks["onnx_evaluator_callback"].sub_step_ctr = 0
        return self.env.step(actions)


class Actor(ExportableActor):
    """An actor network that takes in observations and outputs actions.
    This will be used as the policy network in the environment.
    """

    def __init__(self, num_obs: int, num_act: int):
        super().__init__()

        self._net = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=num_obs,
                out_features=10,
            ),
            torch.nn.ELU(),
            torch.nn.Linear(
                in_features=10,
                out_features=10,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=10,
                out_features=num_act,
            ),
            torch.nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class RNNActor(ExportableActor):
    """An RNN-based actor network that takes in observations and outputs actions.
    This will be used as the policy network in the environment.
    """

    def __init__(self, num_obs: int, num_act: int):
        super().__init__()

        hidden_dim = 5
        num_rnn_layers = 1

        self._rnn = torch.nn.LSTM(
            input_size=num_obs,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=False,
        )

        self._net = Actor(num_obs=hidden_dim, num_act=num_act)

        self._state = None

    def reset(self, dones: torch.Tensor):
        if self._state is None:
            return
        for state in self._state:
            state[:, dones, :] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, self._state = self._rnn(x.unsqueeze(dim=0), self._state)
        return self._net(rnn_out.squeeze(dim=0))

    def get_state(self) -> tuple[torch.Tensor, ...] | None:
        return self._state


def export_and_evaluate_env(
    exp_env: ExportableEnv,
    actor: ExportableActor,
    onnx_file_name: str,
    num_eval_episodes: int,
    max_eval_steps_per_episode: int,
    verbose: bool = False,
) -> bool:
    """Helper function to export an environment and evaluate it using the exported ONNX graph."""
    exp_env.context_manager().add_components(
        [
            # Treat foo as an independent input.
            Input(
                name="foo",
                get_from_env_cb=lambda: exp_env.env.data_source.foo,
                metadata={"description": "The foo tensor from the data source."},
            ),
            # Add an output.
            Output(
                name="out",
                get_from_env_cb=lambda: exp_env.env._output,
                metadata={"description": "The output tensor computed from the actions."},
            ),
            # Add a memory component.
            Memory(
                name="actions",
                get_from_env_cb=lambda: exp_env.env._actions,
            ),
            Memory(
                name="process_actions",
                get_from_env_cb=lambda: exp_env.env._processed_action,
            ),
        ]
    )

    # Treat `bar` and `baz` as part of a group of related inputs, to test exporting groups.
    exp_env.context_manager().add_group(
        Group(
            name="bar_baz_group",
            items=[
                Input(
                    name="bar",
                    get_from_env_cb=lambda: exp_env.env.data_source.bar,
                ),
                Input(
                    name="baz",
                    get_from_env_cb=lambda: exp_env.env.data_source.baz,
                ),
            ],
            metadata={"description": "A group of related inputs."},
        )
    )

    # Export to ONNX.
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = pathlib.Path(tmpdir) / "exploy"
        export_environment_as_onnx(
            env=exp_env,
            actor=actor,
            path=onnx_path,
            filename=onnx_file_name,
            verbose=verbose,
        )

        # Make a session wrapper.
        session_wrapper = SessionWrapper(
            onnx_folder=onnx_path,
            onnx_file_name=onnx_file_name,
            actor=actor,
            optimize=True,
        )

        # Evaluate.
        with torch.inference_mode():
            export_ok, _ = evaluate(
                env=exp_env,
                context_manager=exp_env.context_manager(),
                session_wrapper=session_wrapper,
                num_episodes=num_eval_episodes,
                max_episode_steps=max_eval_steps_per_episode,
                verbose=verbose,
                pause_on_failure=False,
            )
    return export_ok


class TestExportableEnvironment:
    def test_env(self):
        """Test exporting an environment."""
        data_source = DataSource()
        env = Environment(data_source=data_source)
        exp_env = ExportableEnv(env=env)
        actor = Actor(num_obs=env.num_obs, num_act=env.num_act).eval()

        export_ok = export_and_evaluate_env(
            exp_env=exp_env,
            actor=actor,
            onnx_file_name="test_export_env.onnx",
            num_eval_episodes=2,
            max_eval_steps_per_episode=20,
        )
        assert export_ok, "ONNX export validation failed"

    def test_env_with_module(self):
        """Test exporting an environment that uses a torch Module
        to compute observations."""
        data_source = DataSource()
        env = EnvironmentWithTorchModule(data_source=data_source)
        exp_env = ExportableEnv(env=env)
        actor: ExportableActor = Actor(num_obs=env.num_obs, num_act=env.num_act).eval()
        exp_env.context_manager().add_module(env.module)

        export_ok = export_and_evaluate_env(
            exp_env=exp_env,
            actor=actor,
            onnx_file_name="test_export_env_with_module.onnx",
            num_eval_episodes=2,
            max_eval_steps_per_episode=20,
        )
        assert export_ok, "ONNX export validation failed"

    def test_env_with_module_and_rnn_actor(self):
        """Test exporting an environment that uses a torch Module
        to compute observations, and uses an RNN-based actor network as the policy."""
        data_source = DataSource()
        env = EnvironmentWithTorchModule(data_source=data_source)
        exp_env = ExportableEnv(env=env)
        exp_env.context_manager().add_module(env.module)

        actor: ExportableActor = RNNActor(num_obs=env.num_obs, num_act=env.num_act).eval()

        # Call the actor once to initialize its hidden state.
        actor(exp_env.empty_actor_observations())

        add_actor_memory(
            context_manager=exp_env.context_manager(),
            get_hidden_states_func=actor.get_state,
        )

        export_ok = export_and_evaluate_env(
            exp_env=exp_env,
            actor=actor,
            onnx_file_name="test_export_env_with_rnn_actor.onnx",
            num_eval_episodes=2,
            max_eval_steps_per_episode=20,
        )
        assert export_ok, "ONNX export validation failed"
