# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import copy
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectData


class RigidObjectDataSource:
    """Mimic the interface of a `RigidObjectData`, but manage its own tensor data.

    This class is an adaptor for a `RigidObjectData` class, mimicking its full interface. However,
    it holds its own tensor data. The main use case for this class is to be used in the context of
    exporting an MDP to ONNX.
    """

    def __init__(self, rigid_object: RigidObject):
        rigid_object_data: RigidObjectData = rigid_object.data
        self._data = rigid_object_data
        self.device = rigid_object_data.device

        # Pose of the com of each body wrt the actor frame of the body.
        self._coms_pos_b = rigid_object_data._coms_pos_b.clone()
        self._coms_quat_b = rigid_object_data._coms_quat_b.clone()

        self._com_root_pos_b = self._coms_pos_b[:, 0]
        self._com_root_quat_b = self._coms_quat_b[:, 0]

        # Initialize constants.
        self.GRAVITY_VEC_W = rigid_object_data.GRAVITY_VEC_W.clone()
        self.FORWARD_VEC_B = rigid_object_data.FORWARD_VEC_B.clone()

        # The tensors below serve as a single source of truth from which states in different frames can be calculated.
        # We assume having access the velocities in the base frame and positions/orientations in the world frame.

        # Initialize root state tensors.
        self._root_lin_vel_b = rigid_object_data.root_lin_vel_b.clone()
        self._root_ang_vel_b = rigid_object_data.root_ang_vel_b.clone()

        # Initialize tensors by cloning from the original data.
        self._root_pos_w = rigid_object_data.root_pos_w.clone()
        self._root_quat_w = rigid_object_data.root_quat_w.clone()
        self._body_acc_w = rigid_object_data.body_acc_w.clone()

        # Clone all tensors from source class.
        for k, v in vars(rigid_object_data).items():
            if v is None:
                setattr(self, k, v)
            elif isinstance(v, torch.Tensor):
                setattr(self, k, v.clone())
            elif isinstance(v, list):
                setattr(self, k, copy.copy(v))
            else:
                continue

    def update(self, *args, **kwargs):
        """Empty `update` call to implement the interface of `RigidObjectData`."""
        pass

    ##
    # Root state properties.
    ##

    @property
    def root_state_w(self) -> torch.Tensor:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and orientation are of the rigid body's actor frame. Meanwhile, the linear and angular
        velocities are of the rigid body's center of mass frame.
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

        The position, quaternion, and linear/angular velocity are of the rigid body root frame relative to the
        world.
        """
        pose = torch.cat([self.root_pos_w, self.root_quat_w], dim=-1)
        twist = self.root_vel_w.clone()
        twist[:, :3] += torch.linalg.cross(
            twist[:, 3:], math_utils.quat_rotate(self.root_quat_w, -self.com_pos_b[:, 0, :]), dim=-1
        )
        return torch.cat([pose, twist], dim=-1)

    @property
    def root_com_state_w(self) -> torch.Tensor:
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body's center of mass frame
        relative to the world. Center of mass frame is the orientation principle axes of inertia.
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
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame. Shape is (num_instances, 1, 13).

        The position and orientation are of the rigid bodies' actor frame. Meanwhile, the linear and angular
        velocities are of the rigid bodies' center of mass frame.
        """
        return self.root_state_w.view(-1, 1, 13)

    @property
    def body_link_state_w(self) -> torch.Tensor:
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, 1, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """
        return self.root_link_state_w.view(-1, 1, 13)

    @property
    def body_com_state_w(self) -> torch.Tensor:
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """
        return self.root_com_state_w.view(-1, 1, 13)

    @property
    def body_acc_w(self) -> torch.Tensor:
        """Acceleration of all bodies. Shape is (num_instances, 1, 6).

        This quantity is the acceleration of the rigid bodies' center of mass frame.
        """
        return self._body_acc_w

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return math_utils.quat_rotate_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self) -> torch.Tensor:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = math_utils.quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    ##
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body.
        """
        return self._root_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._root_quat_w

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame.
        """
        return torch.cat([self.root_lin_vel_w, self.root_ang_vel_w], dim=-1)

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return math_utils.quat_apply(self.root_quat_w, self.root_lin_vel_b)

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame.
        """
        return math_utils.quat_apply(self.root_quat_w, self.root_ang_vel_b)

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self._root_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self._root_ang_vel_b

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
        return self.root_lin_vel_b

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self.root_ang_vel_b

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the position of the rigid bodies' actor frame.
        """
        return self._root_pos_w.view(-1, 1, 3)

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, 1, 4).

        This quantity is the orientation of the rigid bodies' actor frame.
        """
        return self._root_quat_w.view(-1, 1, 4)

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 6).

        This quantity contains the linear and angular velocities of the rigid bodies' center of mass frame.
        """
        return self.root_vel_w.view(-1, 1, 6)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        return self.root_lin_vel_w.view(-1, 1, 3)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        return self.root_ang_vel_w.view(-1, 1, 3)

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        return self.body_acc_w[..., :3]

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        return self.body_acc_w[..., 3:6]

    #
    # Link body properties
    #

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the position of the rigid bodies' actor frame relative to the world.
        """

        return self.root_pos_w.view(-1, 1, 3)

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, 1, 4).

        This quantity is the orientation of the rigid bodies' actor frame relative to the world.
        """
        return self.root_quat_w.view(-1, 1, 4)

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root rigid body relative to the world.
        """
        twist = self.root_vel_w.clone()
        twist[:, :3] += torch.linalg.cross(
            twist[:, 3:], math_utils.quat_rotate(self.root_quat_w, -self.com_pos_b[:, 0, :]), dim=-1
        )
        return twist.view(-1, 1, 6)

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame relative to the world.
        """
        twist = self.root_vel_w.clone()
        twist[:, :3] += torch.linalg.cross(
            twist[:, 3:], math_utils.quat_rotate(self.root_quat_w, -self.com_pos_b[:, 0, :]), dim=-1
        )
        return twist[:, :3].view(-1, 1, 3)

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame relative to the world.
        """
        return self.root_ang_vel_w.view(-1, 1, 3)

    #
    # Center of mass body properties
    #

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the position of the rigid bodies' center of mass frame.
        """
        pos, _ = math_utils.combine_frame_transforms(
            self.root_pos_w,
            self.root_quat_w,
            self.com_pos_b[:, 0, :],
            self.com_quat_b[:, 0, :],
        )
        return pos.view(-1, 1, 3)

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all bodies in simulation world frame.

        Shape is (num_instances, 1, 4). This quantity is the orientation of the rigid bodies' center of mass frame.
        """
        _, quat = math_utils.combine_frame_transforms(
            self.root_pos_w,
            self.root_quat_w,
            self.com_pos_b[:, 0, :],
            self.com_quat_b[:, 0, :],
        )
        return quat.view(-1, 1, 4)

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 6).

        This quantity contains the linear and angular velocities of the rigid bodies' center of mass frame.
        """
        return self.root_vel_w.view(-1, 1, 6)

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        return self.root_lin_vel_w.view(-1, 1, 3)

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        return self.root_ang_vel_w.view(-1, 1, 3)

    @property
    def com_pos_b(self) -> torch.Tensor:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, 1, 3).

        This quantity is the center of mass location relative to its body's link frame.
        """
        return self._coms_pos_b.view(-1, 1, 3)

    @property
    def com_quat_b(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, 1, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        return self._coms_quat_b.view(-1, 1, 4)


def rigid_object_data_to_dict(
    object_name: str,
    source: RigidObjectDataSource,
    env_id: int,
    device: str,
) -> dict[str, torch.Tensor]:
    assert isinstance(source, RigidObjectDataSource)
    return {
        f"body.{object_name}.pos": source.root_pos_w[env_id].to(device),
        f"body.{object_name}.quat": source.root_quat_w[env_id].to(device),
        f"body.{object_name}.lin_vel": source.root_lin_vel_b[env_id].to(device),
        f"body.{object_name}.ang_vel": source.root_ang_vel_b[env_id].to(device),
    }


def dict_to_rigid_object_data(
    data: dict[str, torch.Tensor],
    object_name: str,
    target: RigidObjectDataSource,
    env_id: int,
) -> None:
    assert isinstance(target, RigidObjectDataSource)
    target._root_pos_w[env_id] = data[f"body.{object_name}.pos"]
    target._root_quat_w[env_id] = data[f"body.{object_name}.quat"]
    target._root_lin_vel_b[env_id] = data[f"body.{object_name}.lin_vel"]
    target._root_ang_vel_b[env_id] = data[f"body.{object_name}.ang_vel"]
