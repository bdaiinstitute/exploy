# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import copy

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, ArticulationData

from exporter.tensor_proxy import TensorProxy


class ArticulationDataSource:
    """Mimic the interface of an `ArticulationData`, but manage its own tensor data.

    This class is an adaptor for an `ArticulationData` class, mimicking its full interface. However,
    it holds its own tensor data. The main use case for this class is to be used in the context of
    exporting an MDP to ONNX.
    """

    def __init__(self, articulation: Articulation):
        articulation_data: ArticulationData = articulation.data
        self._data = articulation_data
        self.device = articulation_data.device

        # Pose of the com of each body wrt the actor frame of the body.
        # TODO: port fix back
        self._coms_pos_b = articulation_data.body_com_pose_b.data[:, :, :3].clone()
        self._coms_quat_b = articulation_data.body_com_pose_b[:, :, 3:7].clone()

        self._com_root_pos_b = self._coms_pos_b[:, 0]
        self._com_root_quat_b = self._coms_quat_b[:, 0]

        # Initialize constants.
        self.GRAVITY_VEC_W = articulation_data.GRAVITY_VEC_W.clone()
        self.FORWARD_VEC_B = articulation_data.FORWARD_VEC_B.clone()

        # Initialize history for finite differencing.
        self._previous_joint_vel = articulation_data._previous_joint_vel.clone()

        # The tensors below serve as a single source of truth from which states in different frames can be calculated.
        # We assume having access the velocities in the base frame and positions/orientations in the world frame.
        # TODO currently we have two sources of truth for the velocities. Consider just having base frame velocities.

        # Initialize root state tensors.
        self._root_lin_vel_b = articulation_data.root_lin_vel_b.clone()
        self._root_ang_vel_b = articulation_data.root_ang_vel_b.clone()

        # Initialize body tensors.
        # Note: we use a `TensorProxy` class that allows us to split the original
        #       body tensors into one tensor per body.
        self._root_body_id = 0
        num_bodies = articulation.num_bodies

        self._body_pos_w = TensorProxy(
            tensors=[articulation_data.body_pos_w[:, id].clone() for id in range(num_bodies)],
            split_dim=1,
        )
        self._body_quat_w = TensorProxy(
            tensors=[articulation_data.body_quat_w[:, id].clone() for id in range(num_bodies)],
            split_dim=1,
        )
        self._body_lin_vel_w = TensorProxy(
            tensors=[articulation_data.body_lin_vel_w[:, id].clone() for id in range(num_bodies)],
            split_dim=1,
        )
        self._body_ang_vel_w = TensorProxy(
            tensors=[articulation_data.body_ang_vel_w[:, id].clone() for id in range(num_bodies)],
            split_dim=1,
        )

        self._body_acc_w = articulation_data.body_acc_w.clone()

        self._joint_pos = articulation_data.joint_pos.clone()
        self._joint_vel = articulation_data.joint_vel.clone()
        self._joint_acc = articulation_data.joint_acc.clone()

        self._body_incoming_joint_wrench_b = articulation_data.body_incoming_joint_wrench_b.clone()

        # Clone all tensors from source class.
        for k, v in vars(articulation_data).items():
            if v is None:
                setattr(self, k, v)
            elif isinstance(v, torch.Tensor):
                setattr(self, k, v.clone())
            elif isinstance(v, list):
                setattr(self, k, copy.copy(v))
            else:
                continue

    def update(self, *args, **kwargs):
        """Empty `update` call to implement the interface of `ArticulationData`."""
        pass

    @property
    def root_state_w(self) -> torch.Tensor:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame relative to the world. Meanwhile,
        the linear and angular velocities are of the articulation root's center of mass frame.
        """
        return torch.cat(
            [
                self.root_pos_w,
                self.root_quat_w,
                self.root_lin_vel_w,
                self.root_ang_vel_w,
            ],
            dim=-1,
        )

    @property
    def root_link_state_w(self) -> torch.Tensor:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root's actor frame relative to the
        world.
        """
        pose = torch.cat([self.root_pos_w, self.root_quat_w], dim=-1)
        twist = self.root_vel_w.clone()
        twist[:, :3] += torch.linalg.cross(
            twist[:, 3:],
            math_utils.quat_apply(pose[:, 3:7], -self.com_pos_b[:, 0, :]),
            dim=-1,
        )
        return torch.cat([pose, twist], dim=-1)

    @property
    def root_com_state_w(self) -> torch.Tensor:
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root link's center of mass frame
        relative to the world. Center of mass frame is assumed to be the same orientation as the link rather than the
        orientation of the principle inertia.
        """
        pos, quat = math_utils.combine_frame_transforms(
            self.root_pos_w,
            self.root_quat_w,
            self.com_pos_b[:, 0, :],
            self.com_quat_b[:, 0, :],
        )
        twist = self.root_vel_w.clone()
        return torch.cat([pos, quat, twist], dim=-1)

    @property
    def body_state_w(self) -> torch.Tensor:
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and quaternion are of all the articulation links's actor frame. Meanwhile, the linear and angular
        velocities are of the articulation links's center of mass frame.
        """
        return torch.cat(
            [
                self.body_pos_w,
                self.body_quat_w,
                self.body_lin_vel_w,
                self.body_ang_vel_w,
            ],
            dim=-1,
        )

    @property
    def body_link_state_w(self) -> torch.Tensor:
        """State of all bodies' link frame`[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """
        return torch.cat(
            [
                self.body_link_pos_w,
                self.body_link_quat_w,
                self.body_link_lin_vel_w,
                self.body_link_ang_vel_w,
            ],
            dim=-1,
        )

    @property
    def body_com_state_w(self) -> torch.Tensor:
        """State of all bodies center of mass `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """
        return torch.cat(
            [
                self.body_com_pos_w,
                self.body_com_quat_w,
                self.body_com_lin_vel_w,
                self.body_com_ang_vel_w,
            ],
            dim=-1,
        )

    @property
    def body_acc_w(self) -> torch.Tensor:
        """Acceleration of all bodies (center of mass). Shape is (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        return self._body_acc_w

    @property
    def body_incoming_joint_wrench_b(self) -> torch.Tensor:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation"""
        return self._body_incoming_joint_wrench_b

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self) -> torch.Tensor:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = math_utils.quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    @property
    def joint_pos(self) -> torch.Tensor:
        """Joint positions of all joints. Shape is (num_instances, num_joints)."""
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        """Joint velocities of all joints. Shape is (num_instances, num_joints)."""
        return self._joint_vel

    @property
    def joint_acc(self) -> torch.Tensor:
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        return self._joint_acc

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the articulation root relative to the world.
        """
        return self.body_pos_w[:, self._root_body_id]

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the articulation root relative to the world.
        """
        return self.body_quat_w[:, self._root_body_id]

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        return torch.cat([self.root_lin_vel_w, self.root_ang_vel_w], dim=-1)

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame relative to the world.
        """
        return math_utils.quat_apply(self.root_quat_w, self.root_lin_vel_b)

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame relative to the world.
        """
        return math_utils.quat_apply(self.root_quat_w, self.root_ang_vel_b)

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame relative to the world
        with respect to the articulation root's actor frame.
        """
        return self._root_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame relative to the world with
        respect to the articulation root's actor frame.
        """
        return self._root_ang_vel_b

    ##
    # Derived Root Link Frame Properties
    ##

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self.root_pos_w

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self.root_quat_w

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        return torch.cat([self.root_link_lin_vel_w, self.root_link_ang_vel_w], dim=-1)

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return math_utils.quat_apply(self.root_link_quat_w, self.root_link_lin_vel_b)

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return math_utils.quat_apply(self.root_link_quat_w, self.root_link_ang_vel_b)

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return self.root_lin_vel_b + torch.linalg.cross(
            self.root_ang_vel_b,
            -self.com_pos_b[:, 0, :],
            dim=-1,
        )

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return self.root_ang_vel_b

    ##
    # Root Center of Mass state properties
    ##

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        pos, _ = math_utils.combine_frame_transforms(
            self.root_pos_w,
            self.root_quat_w,
            self.com_pos_b[:, 0, :],
            self.com_quat_b[:, 0, :],
        )
        return pos

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        _, quat = math_utils.combine_frame_transforms(
            self.root_pos_w,
            self.root_quat_w,
            self.com_pos_b[:, 0, :],
            self.com_quat_b[:, 0, :],
        )
        return quat

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame relative to the world.
        """
        return torch.cat([self.root_com_lin_vel_w, self.root_com_ang_vel_w], dim=-1)

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self.root_lin_vel_w

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self.root_ang_vel_w

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the rigid bodies' actor frame relative to the world.
        """
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the rigid bodies' actor frame relative to the world.
        """
        return self._body_quat_w

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the rigid bodies' center of mass frame relative
        to the world.
        """
        return torch.cat([self.body_lin_vel_w, self.body_ang_vel_w], dim=-1)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame relative to the world.
        """
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame relative to the world.
        """
        return self._body_ang_vel_w

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame relative to the world.
        """
        return self.body_acc_w[..., 0:3]

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame relative to the world.
        """
        return self.body_acc_w[..., 3:6]

    ##
    # Link body properties
    ##

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the rigid bodies' actor frame relative to the world.
        """
        return self.body_pos_w

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the rigid bodies' actor frame  relative to the world.
        """
        return self.body_quat_w

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the rigid bodies' center of mass frame
        relative to the world.
        """
        return torch.cat([self.body_link_lin_vel_w, self.body_link_ang_vel_w], dim=-1)

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame relative to the world.

        For transforming velocities to a new point, see for example:
            https://en.wikipedia.org/wiki/Angular_velocity#Spin_angular_velocity_of_a_rigid_body_or_reference_frame
        """
        link_lin_vel_w = self.body_lin_vel_w + torch.linalg.cross(
            self.body_ang_vel_w,
            math_utils.quat_apply(self._body_quat_w.to_tensor(), -self.com_pos_b),
            dim=-1,
        )
        return link_lin_vel_w

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame relative to the world.
        """
        return self.body_ang_vel_w

    ##
    # Center of mass body properties
    ##

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the rigid bodies' actor frame.
        """
        pos, _ = math_utils.combine_frame_transforms(
            self._body_pos_w.to_tensor(),
            self._body_quat_w.to_tensor(),
            self.com_pos_b,
            self.com_quat_b,
        )
        return pos

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the prinicple axies of inertia of all bodies in simulation world frame.

        Shape is (num_instances, num_bodies, 4). This quantity is the orientation of the rigid bodies' actor frame.
        """
        _, quat = math_utils.combine_frame_transforms(
            self._body_pos_w.to_tensor(),
            self._body_quat_w.to_tensor(),
            self.com_pos_b,
            self.com_quat_b,
        )
        return quat

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the rigid bodies' center of mass frame.
        """
        return torch.cat([self.body_lin_vel_w, self.body_ang_vel_w], dim=-1)

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        return self.body_lin_vel_w

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        return self.body_ang_vel_w

    @property
    def com_pos_b(self) -> torch.Tensor:
        """Center of mass of all of the bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body frame.
        """
        return self._coms_pos_b

    @property
    def com_quat_b(self) -> torch.Tensor:
        """Orientation (w,x,y,z) of the principle axies of inertia of all of the bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body frame.
        """
        return self._coms_quat_b

    ##
    # Backward compatibility.
    ##

    @property
    def joint_limits(self) -> torch.Tensor:
        return self.joint_pos_limits

    @property
    def default_joint_limits(self) -> torch.Tensor:
        return self.default_joint_pos_limits

    @property
    def joint_velocity_limits(self) -> torch.Tensor:
        return self.joint_vel_limits

    @property
    def joint_friction(self) -> torch.Tensor:
        return self.joint_friction_coeff

    @property
    def default_joint_friction(self) -> torch.Tensor:
        return self.default_joint_friction_coeff

    @property
    def fixed_tendon_limit(self) -> torch.Tensor:
        return self.fixed_tendon_pos_limits

    @property
    def default_fixed_tendon_limit(self) -> torch.Tensor:
        return self.default_fixed_tendon_pos_limits
