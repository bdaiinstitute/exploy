# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING


def test_add_body_vel(sim_setup):
    """Test that `add_body_vel` adds body-frame velocity inputs for every body.

    This exercises the exporter path that exposes per-body velocities expressed in each body's own
    frame. Stock Isaac Lab only exposes world-frame per-body velocities, so the body-frame
    quantities are derived and materialized against the `ArticulationDataSource` as stored
    per-body leaves when `IsaacLabExportableEnvironment` swaps the data sources in. The inputs
    added by `add_body_vel` must hold the exact stored slice objects (required for ONNX tracing)
    and must fall back to computing from live data once the original articulation data is
    restored; no Isaac Lab modification is required.

    The root body is special-cased: `ArticulationDataSource` keeps `root_lin_vel_b`/
    `root_ang_vel_b` as separate stored leaves rather than deriving them from `body_lin_vel_w`,
    so the root inputs must bind to those attributes.
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

    # Build the set of names we expect: two velocity inputs per body.
    expected_names: set[str] = set()
    num_bodies = 0
    for obj_name, articulation in articulations.items():
        for body_name in articulation.data.body_names:
            num_bodies += 1
            prefix = f"obj.{obj_name}.{body_name}"
            expected_names.add(f"{prefix}.lin_vel_b_rt_w_in_b")
            expected_names.add(f"{prefix}.ang_vel_b_rt_w_in_b")

    assert set(comp_by_name) == expected_names
    assert len(input_components) == 2 * num_bodies

    # The root inputs must bind to the separately stored root velocity leaves.
    for obj_name, articulation in articulations.items():
        root_name = articulation.body_names[0]
        root_prefix = f"obj.{obj_name}.{root_name}"
        root_lin_vel_comp = comp_by_name[f"{root_prefix}.lin_vel_b_rt_w_in_b"]
        root_ang_vel_comp = comp_by_name[f"{root_prefix}.ang_vel_b_rt_w_in_b"]
        assert root_lin_vel_comp.input_data is articulation.data.root_lin_vel_b
        assert root_ang_vel_comp.input_data is articulation.data.root_ang_vel_b
        assert root_lin_vel_comp.get_from_env_cb() is root_lin_vel_comp.input_data
        assert root_ang_vel_comp.get_from_env_cb() is root_ang_vel_comp.input_data

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

    # The root inputs must likewise read the live root velocities.
    for obj_name, articulation in articulations.items():
        root_prefix = f"obj.{obj_name}.{articulation.body_names[0]}"
        root_lin_vel = comp_by_name[f"{root_prefix}.lin_vel_b_rt_w_in_b"].get_from_env_cb()
        root_ang_vel = comp_by_name[f"{root_prefix}.ang_vel_b_rt_w_in_b"].get_from_env_cb()
        assert torch.allclose(root_lin_vel, articulation.data.root_lin_vel_b)
        assert torch.allclose(root_ang_vel, articulation.data.root_ang_vel_b)

    env.close()


def test_add_body_inputs_rigid_object(sim_setup):
    """Test that the body pose/velocity inputs also work for `RigidObject` assets.

    `add_body_pos`, `add_body_quat` and `add_body_vel` are shared between `Articulation` and
    `RigidObject`. For the registered inputs to become ONNX graph inputs (instead of being
    constant-folded at trace time), their callbacks must read the exact tensors stored on the
    swapped-in `RigidObjectDataSource`. This is a regression test for the case where the source
    stored the root pose as a plain tensor and exposed `body_pos_w` as a fresh `view`, which made
    the pose inputs bind to throwaway tensors that were never read during tracing.
    """
    # Import after AppLauncher is initialized
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.prims as prim_utils
    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import build_simulation_context

    from exploy.exporter.core.context_manager import ContextManager
    from exploy.exporter.frameworks.isaaclab import inputs
    from exploy.exporter.frameworks.isaaclab.rigid_object_data import (
        RigidObjectDataSource,
        dict_to_rigid_object_data,
        rigid_object_data_to_dict,
    )

    device = "cpu"
    num_cubes = 3
    obj_name = "cube"
    torch.manual_seed(42)

    with build_simulation_context(
        sim_utils.SimulationCfg(), device=device, auto_add_lighting=True
    ) as sim:
        # Disable app control callback (from Isaac Lab test pattern)
        sim._app_control_on_stop_handle = None

        # Create Top-level Xforms, one for each cube.
        origins = torch.tensor([(i * 2.0, 0.0, 1.0) for i in range(num_cubes)]).to(device)
        for i, origin in enumerate(origins):
            prim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

        cube_object = RigidObject(
            cfg=RigidObjectCfg(
                prim_path="/World/Table_.*/Object",
                spawn=sim_utils.CuboidCfg(
                    size=(0.2, 0.2, 0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            )
        )

        sim.reset()
        assert cube_object.is_initialized

        # Step the simulation to populate the rigid object data with a non-default state.
        for step in range(10):
            if step % 5 == 0:
                cube_object.set_external_force_and_torque(
                    torch.randn(num_cubes, 1, 3, device=device) * 100.0,
                    torch.randn(num_cubes, 1, 3, device=device) * 10.0,
                )
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(dt=sim.cfg.dt)

        body_name = cube_object.body_names[0]
        prefix = f"obj.{obj_name}.{body_name}"

        # Snapshot the live state before the export data source is swapped in.
        original_data = cube_object.data
        expected_pos = original_data.body_pos_w[:, 0].clone()
        expected_quat = original_data.body_quat_w[:, 0].clone()
        expected_lin_vel = original_data.root_lin_vel_b.clone()
        expected_ang_vel = original_data.root_ang_vel_b.clone()

        # Swap in the exportable data source, as `IsaacLabExportableEnvironment` does.
        source = RigidObjectDataSource(cube_object)
        cube_object._data = source

        rigid_objects = {obj_name: cube_object}
        context_manager = ContextManager()
        inputs.add_body_pos_and_quat(articulations=rigid_objects, context_manager=context_manager)
        inputs.add_body_vel(articulations=rigid_objects, context_manager=context_manager)

        comp_by_name = {comp.input_name: comp for comp in context_manager.get_input_components()}

        # A rigid object has exactly one body, so we expect one input per quantity.
        assert set(comp_by_name) == {
            f"{prefix}.pos_b_rt_w_in_w",
            f"{prefix}.w_Q_b",
            f"{prefix}.lin_vel_b_rt_w_in_b",
            f"{prefix}.ang_vel_b_rt_w_in_b",
        }

        pos_comp = comp_by_name[f"{prefix}.pos_b_rt_w_in_w"]
        quat_comp = comp_by_name[f"{prefix}.w_Q_b"]
        lin_vel_comp = comp_by_name[f"{prefix}.lin_vel_b_rt_w_in_b"]
        ang_vel_comp = comp_by_name[f"{prefix}.ang_vel_b_rt_w_in_b"]

        # Identity: each input must hold the exact tensor stored on the data source, and repeated
        # callback reads must return that same object. Without this, the inputs are dropped during
        # tracing and the quantities are baked into the ONNX graph as constants.
        assert pos_comp.input_data is source.body_pos_w[:, 0]
        assert quat_comp.input_data is source.body_quat_w[:, 0]
        assert lin_vel_comp.input_data is source.root_lin_vel_b
        assert ang_vel_comp.input_data is source.root_ang_vel_b

        for comp in (pos_comp, quat_comp, lin_vel_comp, ang_vel_comp):
            assert comp.get_from_env_cb() is comp.input_data

        # The root pose must be derived from the very same stored leaves as the body pose, so that
        # observation terms reading either view are connected to the same graph inputs.
        assert source.root_pos_w is pos_comp.input_data
        assert source.root_quat_w is quat_comp.input_data

        # Values must round-trip from the live simulation state.
        assert pos_comp.input_data.shape == (num_cubes, 3)
        assert quat_comp.input_data.shape == (num_cubes, 4)
        assert lin_vel_comp.input_data.shape == (num_cubes, 3)
        assert ang_vel_comp.input_data.shape == (num_cubes, 3)
        assert torch.allclose(pos_comp.input_data, expected_pos)
        assert torch.allclose(quat_comp.input_data, expected_quat)
        assert torch.allclose(lin_vel_comp.input_data, expected_lin_vel)
        assert torch.allclose(ang_vel_comp.input_data, expected_ang_vel)

        # Writing new state into the data source must be visible through the registered inputs,
        # i.e. the inputs are live leaves rather than snapshots taken at registration time.
        new_state = {
            f"body.{obj_name}.pos": torch.tensor([1.0, 2.0, 3.0], device=device),
            f"body.{obj_name}.quat": torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
            f"body.{obj_name}.lin_vel": torch.tensor([4.0, 5.0, 6.0], device=device),
            f"body.{obj_name}.ang_vel": torch.tensor([7.0, 8.0, 9.0], device=device),
        }
        dict_to_rigid_object_data(data=new_state, object_name=obj_name, target=source, env_id=0)
        assert torch.allclose(pos_comp.input_data[0], new_state[f"body.{obj_name}.pos"])
        assert torch.allclose(quat_comp.input_data[0], new_state[f"body.{obj_name}.quat"])
        assert torch.allclose(lin_vel_comp.input_data[0], new_state[f"body.{obj_name}.lin_vel"])
        assert torch.allclose(ang_vel_comp.input_data[0], new_state[f"body.{obj_name}.ang_vel"])

        # The read-back helper must observe the same values.
        read_back = rigid_object_data_to_dict(object_name=obj_name, source=source, env_id=0)
        for key, value in new_state.items():
            assert torch.allclose(read_back[key], value)

        # Once the original data is restored (as during post-export evaluation), the callbacks must
        # fall back to reading the live rigid object data.
        cube_object._data = original_data
        assert torch.allclose(pos_comp.get_from_env_cb(), expected_pos)
        assert torch.allclose(quat_comp.get_from_env_cb(), expected_quat)
        assert torch.allclose(lin_vel_comp.get_from_env_cb(), expected_lin_vel)
        assert torch.allclose(ang_vel_comp.get_from_env_cb(), expected_ang_vel)
