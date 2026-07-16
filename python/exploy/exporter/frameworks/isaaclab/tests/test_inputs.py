# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING


def test_add_body_vel(sim_setup):
    """Test that `add_body_vel` adds body-frame velocity inputs for every non-root body.

    This exercises the exporter path that exposes per-body velocities expressed in each body's own
    frame. Stock Isaac Lab only exposes world-frame per-body velocities, so the body-frame
    quantities are derived and materialized against the `ArticulationDataSource` as stored
    per-body leaves when `IsaacLabExportableEnvironment` swaps the data sources in. The inputs
    added by `add_body_vel` must hold the exact stored slice objects (required for ONNX tracing)
    and must fall back to computing from live data once the original articulation data is
    restored; no Isaac Lab modification is required.
    """
    # Import after AppLauncher is initialized
    import gymnasium as gym
    import isaaclab.utils.math as math_utils
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from exploy.exporter.core.context_manager import ContextManager
    from exploy.exporter.core.tensor_proxy import TensorProxy
    from exploy.exporter.frameworks.isaaclab import derived_tensors, inputs
    from exploy.exporter.frameworks.isaaclab.env import IsaacLabExportableEnvironment

    if TYPE_CHECKING:
        from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    import isaaclab_tasks.manager_based.locomotion.velocity.config.g1  # noqa: F401

    task_name = "Isaac-Velocity-Rough-G1-Play-v0"
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task_name, num_envs=2, device="cpu")
    env_cfg.seed = 42
    env: ManagerBasedRLEnv = gym.make(
        task_name,
        cfg=env_cfg,
        render_mode=None,
    ).unwrapped

    # Step the environment for a few steps to populate the articulation data with a non-default state.
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(5):
        env.step(actions)

    # Swap in the exportable data sources so that the derived tensors are materialized against
    # the export proxies.
    exportable_env = IsaacLabExportableEnvironment(env)
    articulations = env.scene.articulations

    context_manager = ContextManager()
    inputs.add_body_vel(articulations=articulations, context_manager=context_manager)

    input_components = context_manager.get_input_components()
    comp_by_name = {comp.input_name: comp for comp in input_components}

    # Build the set of names we expect: two velocity inputs per non-root body, root skipped.
    expected_names: set[str] = set()
    num_non_root_bodies = 0
    for obj_name, articulation in articulations.items():
        root_name = articulation.body_names[0]
        for body_name in articulation.data.body_names:
            if body_name == root_name:
                continue
            num_non_root_bodies += 1
            prefix = f"obj.{obj_name}.{body_name}"
            expected_names.add(f"{prefix}.lin_vel_b_rt_w_in_b")
            expected_names.add(f"{prefix}.ang_vel_b_rt_w_in_b")

    assert set(comp_by_name) == expected_names
    assert len(input_components) == 2 * num_non_root_bodies

    # Root bodies must be skipped.
    for obj_name, articulation in articulations.items():
        root_name = articulation.body_names[0]
        assert f"obj.{obj_name}.{root_name}.lin_vel_b_rt_w_in_b" not in comp_by_name
        assert f"obj.{obj_name}.{root_name}.ang_vel_b_rt_w_in_b" not in comp_by_name

    # The body-frame velocities must be materialized against the data source as per-body leaves.
    for articulation in articulations.values():
        assert isinstance(derived_tensors.body_link_lin_vel_b(articulation.data), TensorProxy)
        assert isinstance(derived_tensors.body_link_ang_vel_b(articulation.data), TensorProxy)

    # Each input must match the corresponding body-frame velocity, derived from the world-frame
    # link velocity rotated into the body's own frame, and must hold the exact stored leaf object
    # so that it is traced as an ONNX graph input.
    for obj_name, articulation in articulations.items():
        root_name = articulation.body_names[0]
        for body_name in articulation.data.body_names:
            if body_name == root_name:
                continue
            body_ids, _ = articulation.find_bodies(body_name)
            idx = body_ids[0]
            prefix = f"obj.{obj_name}.{body_name}"

            lin_vel_comp = comp_by_name[f"{prefix}.lin_vel_b_rt_w_in_b"]
            ang_vel_comp = comp_by_name[f"{prefix}.ang_vel_b_rt_w_in_b"]
            lin_vel = lin_vel_comp.input_data
            ang_vel = ang_vel_comp.input_data

            # Identity: the input holds the stored slice, and repeated callback reads return the
            # same object.
            assert lin_vel is derived_tensors.body_link_lin_vel_b(articulation.data)[:, idx]
            assert ang_vel is derived_tensors.body_link_ang_vel_b(articulation.data)[:, idx]
            assert lin_vel_comp.get_from_env_cb() is lin_vel
            assert ang_vel_comp.get_from_env_cb() is ang_vel

            body_quat_w = articulation.data.body_link_quat_w[:, idx, :]
            expected_lin_vel = math_utils.quat_apply_inverse(
                body_quat_w, articulation.data.body_link_lin_vel_w[:, idx, :]
            )
            expected_ang_vel = math_utils.quat_apply_inverse(
                body_quat_w, articulation.data.body_link_ang_vel_w[:, idx, :]
            )

            assert lin_vel.shape == (env.num_envs, 3)
            assert ang_vel.shape == (env.num_envs, 3)
            assert torch.allclose(lin_vel, expected_lin_vel)
            assert torch.allclose(ang_vel, expected_ang_vel)

    # After the original articulation data is restored (as during post-export evaluation), the
    # input callbacks must fall back to computing the body-frame velocities from live data.
    exportable_env.cleanup()
    for obj_name, articulation in articulations.items():
        root_name = articulation.body_names[0]
        for body_name in articulation.data.body_names:
            if body_name == root_name:
                continue
            body_ids, _ = articulation.find_bodies(body_name)
            idx = body_ids[0]
            prefix = f"obj.{obj_name}.{body_name}"

            lin_vel = comp_by_name[f"{prefix}.lin_vel_b_rt_w_in_b"].get_from_env_cb()
            ang_vel = comp_by_name[f"{prefix}.ang_vel_b_rt_w_in_b"].get_from_env_cb()

            body_quat_w = articulation.data.body_link_quat_w[:, idx, :]
            expected_lin_vel = math_utils.quat_apply_inverse(
                body_quat_w, articulation.data.body_link_lin_vel_w[:, idx, :]
            )
            expected_ang_vel = math_utils.quat_apply_inverse(
                body_quat_w, articulation.data.body_link_ang_vel_w[:, idx, :]
            )

            assert torch.allclose(lin_vel, expected_lin_vel)
            assert torch.allclose(ang_vel, expected_ang_vel)

    env.close()
