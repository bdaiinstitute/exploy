# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations


def test_add_body_vel(env):
    """Test that ``add_body_vel`` adds body-frame velocity inputs for every non-root body.

    mjlab only exposes per-body velocities in the world frame, so the body-frame quantities are
    derived by rotating the world-frame link velocities into each body's own frame and
    materialized against the ``EntityDataSource`` as stored per-body leaves when the data sources
    are swapped in. The inputs added by ``add_body_vel`` must hold the exact stored slice objects
    (required for ONNX tracing) and must fall back to computing from live data once the original
    entity data is restored.
    """
    import mjlab.utils.lab_api.math as math_utils
    import torch

    from exploy.exporter.core.context_manager import ContextManager
    from exploy.exporter.core.tensor_proxy import TensorProxy
    from exploy.exporter.frameworks.mjlab import derived_tensors, inputs
    from exploy.exporter.frameworks.mjlab.entity_data import EntityDataSource

    # Step for a few steps to populate data with a non-default state.
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(5):
        env.step(actions)

    entities = env.scene.entities

    # Swap in the exportable data sources and materialize the derived tensors against them,
    # mirroring MjlabExportableEnvironment. The env fixture is shared across tests in the
    # session, so the original data is restored at the end of this test.
    original_data = {name: entity._data for name, entity in entities.items()}
    for entity in entities.values():
        entity._data = EntityDataSource(entity=entity)
        derived_tensors.body_link_lin_vel_b.materialize(entity._data)
        derived_tensors.body_link_ang_vel_b.materialize(entity._data)

    try:
        context_manager = ContextManager()
        inputs.add_body_vel(entities, context_manager)

        input_components = context_manager.get_input_components()
        comp_by_name = {comp.input_name: comp for comp in input_components}

        # Build the set of names we expect: two velocity inputs per non-root body, root skipped.
        expected_names: set[str] = set()
        num_non_root_bodies = 0
        for obj_name, entity in entities.items():
            root_body_name = entity.root_body.name.split("/")[-1]
            for body_name in entity.body_names:
                if body_name == root_body_name:
                    continue
                num_non_root_bodies += 1
                prefix = f"obj.{obj_name}.{body_name}"
                expected_names.add(f"{prefix}.lin_vel_b_rt_w_in_b")
                expected_names.add(f"{prefix}.ang_vel_b_rt_w_in_b")

        assert num_non_root_bodies > 0
        assert set(comp_by_name) == expected_names
        assert len(input_components) == 2 * num_non_root_bodies

        # Root bodies must be skipped.
        for obj_name, entity in entities.items():
            root_body_name = entity.root_body.name.split("/")[-1]
            assert f"obj.{obj_name}.{root_body_name}.lin_vel_b_rt_w_in_b" not in comp_by_name
            assert f"obj.{obj_name}.{root_body_name}.ang_vel_b_rt_w_in_b" not in comp_by_name

        # The body-frame velocities must be materialized against the data source as per-body
        # leaves.
        for entity in entities.values():
            assert isinstance(derived_tensors.body_link_lin_vel_b(entity.data), TensorProxy)
            assert isinstance(derived_tensors.body_link_ang_vel_b(entity.data), TensorProxy)

        # Each input must match the corresponding body-frame velocity, derived from the
        # world-frame link velocity rotated into the body's own frame, and must hold the exact
        # stored leaf object so that it is traced as an ONNX graph input.
        for obj_name, entity in entities.items():
            root_body_name = entity.root_body.name.split("/")[-1]
            for idx, body_name in enumerate(entity.body_names):
                if body_name == root_body_name:
                    continue
                prefix = f"obj.{obj_name}.{body_name}"

                lin_vel_comp = comp_by_name[f"{prefix}.lin_vel_b_rt_w_in_b"]
                ang_vel_comp = comp_by_name[f"{prefix}.ang_vel_b_rt_w_in_b"]
                lin_vel = lin_vel_comp.input_data
                ang_vel = ang_vel_comp.input_data

                # Identity: the input holds the stored slice, and repeated callback reads return
                # the same object.
                assert lin_vel is derived_tensors.body_link_lin_vel_b(entity.data)[:, idx]
                assert ang_vel is derived_tensors.body_link_ang_vel_b(entity.data)[:, idx]
                assert lin_vel_comp.get_from_env_cb() is lin_vel
                assert ang_vel_comp.get_from_env_cb() is ang_vel

                body_quat_w = entity.data.body_link_quat_w[:, idx]
                expected_lin_vel = math_utils.quat_apply_inverse(
                    body_quat_w, entity.data.body_link_lin_vel_w[:, idx]
                )
                expected_ang_vel = math_utils.quat_apply_inverse(
                    body_quat_w, entity.data.body_link_ang_vel_w[:, idx]
                )

                assert lin_vel.shape == (env.num_envs, 3)
                assert ang_vel.shape == (env.num_envs, 3)
                assert torch.allclose(lin_vel, expected_lin_vel)
                assert torch.allclose(ang_vel, expected_ang_vel)
    finally:
        for name, entity in entities.items():
            entity._data = original_data[name]

    # After the original entity data is restored (as during post-export evaluation), the input
    # callbacks must fall back to computing the body-frame velocities from live data.
    for obj_name, entity in entities.items():
        root_body_name = entity.root_body.name.split("/")[-1]
        for idx, body_name in enumerate(entity.body_names):
            if body_name == root_body_name:
                continue
            prefix = f"obj.{obj_name}.{body_name}"
            lin_vel = comp_by_name[f"{prefix}.lin_vel_b_rt_w_in_b"].get_from_env_cb()
            ang_vel = comp_by_name[f"{prefix}.ang_vel_b_rt_w_in_b"].get_from_env_cb()

            body_quat_w = entity.data.body_link_quat_w[:, idx]
            expected_lin_vel = math_utils.quat_apply_inverse(
                body_quat_w, entity.data.body_link_lin_vel_w[:, idx]
            )
            expected_ang_vel = math_utils.quat_apply_inverse(
                body_quat_w, entity.data.body_link_ang_vel_w[:, idx]
            )

            assert torch.allclose(lin_vel, expected_lin_vel)
            assert torch.allclose(ang_vel, expected_ang_vel)
