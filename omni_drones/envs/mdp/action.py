import torch
from .mdp_term import MDPTerm

from omni_drones.robots.multirotor import Multirotor, Rotor

from collections import OrderedDict
from typing import Tuple

class ActionFunc(MDPTerm):

    action_shape: torch.Size

    @property
    def action_shape(self):
        raise NotImplementedError

    def apply_action(self, action: torch.Tensor):
        pass


class ActionGroup:
    def __init__(self, action_funcs: OrderedDict[str, ActionFunc]):
        assert isinstance(action_funcs, OrderedDict), "ActionGroup requires an OrderedDict of ActionFuncs."
        self.action_funcs = action_funcs
        action_shapes = {key: func.action_shape for key, func in self.action_funcs.items()}

        try:
            self.action_shape = torch.cat([torch.zeros(shape) for shape in action_shapes.values()], dim=-1).shape
            self.action_split = [shape[-1] for shape in action_shapes.values()]
        except Exception:
            raise ValueError(f"Incompatible action shapes: {action_shapes}")

    def apply_action(self, action: torch.Tensor):
        action_split = torch.split(action, self.action_split, dim=1)
        for action_func, action in zip(self.action_funcs.values(), action_split):
            action_func.apply_action(action)


class RotorCommand(ActionFunc):
    def __init__(self, env, asset_name: str, actuator_name: str):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.rotor: Rotor = self.asset.actuators[actuator_name]

    @property
    def action_shape(self):
        return self.rotor.shape

    def apply_action(self, action: torch.Tensor):
        self.rotor.throttle_target[:] = action.clamp(0., 1.)


from omni_drones.utils.torch import (
    quat_rotate_inverse,
    quat_rotate,
    normalize,
    quaternion_to_rotation_matrix,
    axis_angle_to_matrix,
)

class LeePositionController(ActionFunc):

    def __init__(
        self,
        env: "IsaacEnv",
        asset_name: str,
        xyz_mode: str,
        yaw_mode: str,
        pos_gain: Tuple[float, float, float],
        vel_gain: Tuple[float, float, float],
        attitude_gain: Tuple[float, float, float],
        ang_rate_gain: Tuple[float, float, float],
        inertia: Tuple[float, float, float], # TODO: calculate inertia
    ):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.rotor: Rotor = self.asset.actuators["rotor"]
        self.xyz_mode = xyz_mode
        self.yaw_mode = yaw_mode

        self.action_dim = 4

        rotor_pos_w = self.asset.data.body_pos_w[0, self.rotor.body_ids, :]
        rotor_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w[0].unsqueeze(0),
            rotor_pos_w - self.asset.data.root_pos_w[0].unsqueeze(0)
        )

        arm_lengths = rotor_pos_b.norm(dim=-1)
        rotor_angles = torch.atan2(rotor_pos_b[..., 1], rotor_pos_b[..., 0])

        def get_template(tensor):
            return tensor.flatten(0, -2)[0]

        rotor_direction = get_template(self.rotor.rotor_direction)
        moment_to_force = get_template(self.rotor.km_normalized / self.rotor.kf_normalized)

        print(f'[INFO]: arm_lengths: {arm_lengths.tolist()}')
        print(f'[INFO]: rotor_angles: {rotor_angles.tolist()}')
        print(f'[INFO]: rotor_direction: {rotor_direction.tolist()}')

        with torch.device(self.device):
            self.pos_gain = torch.as_tensor(pos_gain)
            self.vel_gain = torch.as_tensor(vel_gain)
            gravity_dir, gravity_mag = self.asset.env.sim.get_physics_context().get_gravity()
            self.gravity = torch.as_tensor(gravity_dir) * gravity_mag

            I = torch.as_tensor([*inertia, 1]).diag_embed()
            A = torch.stack(
                [
                    torch.sin(rotor_angles) * arm_lengths,
                    -torch.cos(rotor_angles) * arm_lengths,
                    -rotor_direction * moment_to_force,
                    torch.ones(self.rotor.shape[-1])
                ]
            )
            self.mixer = A.T @ (A @ A.T).inverse() @ I
            self.ang_rate_gain = torch.as_tensor(ang_rate_gain) @ I[:3, :3].inverse()
            self.attitude_gain = torch.as_tensor(attitude_gain) @ I[:3, :3].inverse()

        self.mass = get_template(
            self.asset.root_physx_view
            .get_masses()
            .sum(-1, keepdim=True)
            .to(self.device)
        )

    @property
    def action_shape(self):
        return torch.Size([*self.asset.shape, self.action_dim])

    def apply_action(self, action: torch.Tensor):
        batch_shape = action.shape[:-1]
        action = action.reshape(-1, 4)
        target_throttle = self._compute(action)
        self.rotor.throttle_target[:] = target_throttle.reshape(*batch_shape, self.rotor.num_rotors)

    def _compute(self, action: torch.Tensor):
        target_xyz, target_yaw = action.split([3, 1], dim=-1)

        pos = self.asset.data.root_pos_w - self.env.scene.env_origins
        vel = self.asset.data.root_lin_vel_w
        R = quaternion_to_rotation_matrix(self.asset.data.root_quat_w)

        target_pos = pos
        target_vel = 0.
        target_acc = 0.

        if self.xyz_mode == "pos":
            target_pos = target_xyz
        elif self.xyz_mode == "vel":
            target_vel = target_xyz
        elif self.xyz_mode == "acc":
            target_acc = target_xyz
        else:
            raise ValueError(f"Invalid xyz mode: {self.xyz_mode}")

        pos_error = target_pos - pos
        vel_error = target_vel - vel
        acc = (
            + pos_error * self.pos_gain
            + vel_error * self.vel_gain
            + target_acc
            - self.gravity
        )
        target_thrust = self.mass * (acc * R[:, :, 2]).sum(-1, True)

        if self.yaw_mode == "angle":
            b1_des = torch.cat([
                torch.cos(target_yaw),
                torch.sin(target_yaw),
                torch.zeros_like(target_yaw)
            ],dim=-1)
        elif self.yaw_mode == "rate":
            yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).unsqueeze(-1)
            b1_des = torch.cat([
                torch.cos(yaw),
                torch.sin(yaw),
                torch.zeros_like(yaw)
            ],dim=-1)
        else:
            raise ValueError(f"Invalid yaw mode: {self.yaw_mode}")

        b3_des = normalize(acc)
        b2_des = normalize(torch.cross(b3_des, b1_des, 1))
        R_des = torch.stack([
            b2_des.cross(b3_des, 1),
            b2_des,
            b3_des
        ], dim=-1)

        ang_error_matrix = 0.5 * (
            torch.bmm(R.transpose(-2, -1), R_des)
            - torch.bmm(R_des.transpose(-2, -1), R)
        )
        ang_error = torch.stack([
            ang_error_matrix[:, 2, 1],
            ang_error_matrix[:, 0, 2],
            ang_error_matrix[:, 1, 0]
        ],dim=-1)

        ang_vel = self.asset.data.root_ang_vel_b

        if self.yaw_mode == "angle":
            ang_rate_err = 0. - ang_vel
        elif self.yaw_mode == "rate":
            target_ang_vel = torch.zeros_like(ang_vel)
            target_ang_vel[:, 2] = target_yaw.squeeze(1)
            ang_rate_err = torch.bmm(torch.bmm(R_des.transpose(-2, -1), R), target_ang_vel.unsqueeze(2)).squeeze(2) - ang_vel
        else:
            raise ValueError(f"Invalid yaw mode: {self.yaw_mode}")

        ang_acc = (
            + ang_error * self.attitude_gain
            + ang_rate_err * self.ang_rate_gain
            + torch.cross(ang_vel, ang_vel, 1)
        )
        ang_acc_thrust = torch.cat([ang_acc, target_thrust], dim=-1)

        target_thrusts = (self.mixer @ ang_acc_thrust.T).T
        target_throttle = (target_thrusts / self.rotor.kf_normalized).sqrt()
        return target_throttle

    def debug_vis(self):
        rotor_pos_w = self.asset.data.body_pos_w[..., self.rotor.body_ids, :].flatten(0, -2)
        rotor_quat_w = self.asset.data.body_quat_w[..., self.rotor.body_ids, :].flatten(0, -2)

        throttle = torch.zeros(self.rotor.shape + (3,), device=self.device)
        throttle[..., 2] = self.rotor.throttle
        throttle_w = quat_rotate(rotor_quat_w, throttle.flatten(0, -2))
        self.env.debug_draw.vector(
            rotor_pos_w,
            throttle_w,
        )

        rest_throttle = torch.zeros(self.rotor.shape + (3,), device=self.device)
        rest_throttle[..., 2] = 1.0 - self.rotor.throttle
        rest_throttle_w = quat_rotate(rotor_quat_w, rest_throttle.flatten(0, -2))
        self.env.debug_draw.vector(
            rotor_pos_w + throttle_w,
            rest_throttle_w,
            color=(0., 0., 0., .5)
        )


class AttitudeController(ActionFunc):

    def __init__(
        self,
        env: "IsaacEnv",
        asset_name: str,
        z_mode: str,
        yaw_mode: str,
        inertia: Tuple[float, float, float], # TODO: calculate inertia
        ang_rate_gain: Tuple[float, float, float],
        attitude_gain: Tuple[float, float, float],
        vel_gain: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        z_gain: float = 0.0,
    ):
        super().__init__(env)
        self.asset: Multirotor = self.env.scene[asset_name]
        self.rotor: Rotor = self.asset.actuators["rotor"]
        self.z_mode = z_mode
        self.yaw_mode = yaw_mode

        self.action_dim = 4

        rotor_pos_w = self.asset.data.body_pos_w[0, self.rotor.body_ids, :]
        rotor_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w[0].unsqueeze(0),
            rotor_pos_w - self.asset.data.root_pos_w[0].unsqueeze(0)
        )

        arm_lengths = rotor_pos_b.norm(dim=-1)
        rotor_angles = torch.atan2(rotor_pos_b[..., 1], rotor_pos_b[..., 0])

        def get_template(tensor):
            return tensor.flatten(0, -2)[0]

        rotor_direction = get_template(self.rotor.rotor_direction)
        moment_to_force = get_template(self.rotor.km_normalized / self.rotor.kf_normalized)

        print(f'[INFO]: arm_lengths: {arm_lengths.tolist()}')
        print(f'[INFO]: rotor_angles: {rotor_angles.tolist()}')
        print(f'[INFO]: rotor_direction: {rotor_direction.tolist()}')

        with torch.device(self.device):
            self.vel_gain = torch.as_tensor(vel_gain)
            self.z_gain = torch.as_tensor(z_gain)
            gravity_dir, gravity_mag = self.asset.env.sim.get_physics_context().get_gravity()
            self.gravity = torch.as_tensor(gravity_dir) * gravity_mag

            I = torch.as_tensor([*inertia, 1]).diag_embed()
            A = torch.stack(
                [
                    torch.sin(rotor_angles) * arm_lengths,
                    -torch.cos(rotor_angles) * arm_lengths,
                    -rotor_direction * moment_to_force,
                    torch.ones(self.rotor.shape[-1])
                ]
            )
            self.mixer = A.T @ (A @ A.T).inverse() @ I
            self.body_rate_gain = torch.as_tensor(ang_rate_gain) @ I[:3, :3].inverse()
            self.attitude_gain = torch.as_tensor(attitude_gain) @ I[:3, :3].inverse()

        self.mass = get_template(
            self.asset.root_physx_view
            .get_masses()
            .sum(-1, keepdim=True)
            .to(self.device)
        )

    @property
    def action_shape(self):
        return torch.Size([*self.asset.shape, self.action_dim])

    def apply_action(self, action: torch.Tensor):
        batch_shape = action.shape[:-1]
        action = action.reshape(-1, 4)
        target_throttle = self._compute(action)
        self.rotor.throttle_target[:] = target_throttle.reshape(*batch_shape, self.rotor.num_rotors)

    def _compute(self, action: torch.Tensor):
        target_roll, target_pitch, target_yaw, target_z = action.split([1, 1, 1, 1], dim=-1)

        R = quaternion_to_rotation_matrix(self.asset.data.root_quat_w)

        if self.z_mode == "force":
            target_thrust = target_z
        elif self.z_mode == "height":
            pos = self.asset.data.root_pos_w - self.env.scene.env_origins
            vel = self.asset.data.root_lin_vel_w

            target_xy = pos[:, :2]
            target_pos = torch.cat((target_xy, target_z), dim=1)

            xy_gain = torch.zeros(target_xy.size(), device=self.device)
            z_gain = torch.ones(target_z.size(), device=self.device) * self.z_gain
            pos_gain = torch.cat((xy_gain, z_gain), dim=1)

            pos_error = target_pos - pos
            vel_error = 0. - vel
            acc = (
                + pos_error * pos_gain
                + vel_error * self.vel_gain
                - self.gravity
            )
            target_thrust = self.mass * (acc * R[:, :, 2]).sum(-1, True)
        else:
            raise ValueError(f"Invalid z mode: {self.z_mode}")

        if self.yaw_mode == "angle":
            yaw = axis_angle_to_matrix(target_yaw, torch.tensor([0., 0., 1.], device=self.device))
        elif self.yaw_mode == "rate":
            yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).unsqueeze(-1)
            yaw = axis_angle_to_matrix(yaw, torch.tensor([0., 0., 1.], device=self.device))
        else:
            raise ValueError(f"Invalid yaw mode: {self.yaw_mode}")

        roll = axis_angle_to_matrix(target_roll, torch.tensor([1., 0., 0.], device=self.device))
        pitch = axis_angle_to_matrix(target_pitch, torch.tensor([0., 1., 0.], device=self.device))
        R_des = torch.bmm(torch.bmm(yaw,  roll), pitch)

        ang_error_matrix = 0.5 * (
            torch.bmm(R.transpose(-2, -1), R_des)
            - torch.bmm(R_des.transpose(-2, -1), R)
        )
        ang_error = torch.stack([
            ang_error_matrix[:, 2, 1],
            ang_error_matrix[:, 0, 2],
            ang_error_matrix[:, 1, 0]
        ],dim=-1)

        ang_vel = self.asset.data.root_ang_vel_w
        body_rate = self.asset.data.root_ang_vel_b

        if self.yaw_mode == "angle":
            body_rate_err = 0. - body_rate
        elif self.yaw_mode == "rate":
            target_body_rate = torch.zeros_like(body_rate)
            target_body_rate[:, 2] = target_yaw.squeeze(1)
            body_rate_err = torch.bmm(torch.bmm(R_des.transpose(-2, -1), R), target_body_rate.unsqueeze(2)).squeeze(2) - body_rate
        else:
            raise ValueError(f"Invalid yaw mode: {self.yaw_mode}")

        ang_acc = (
            + ang_error * self.attitude_gain
            + body_rate_err * self.body_rate_gain
            + torch.cross(ang_vel, ang_vel, 1)
        )
        ang_acc_thrust = torch.cat([ang_acc, target_thrust], dim=-1)

        target_thrusts = (self.mixer @ ang_acc_thrust.T).T
        target_throttle = (target_thrusts / self.rotor.kf_normalized).sqrt()
        return target_throttle

    def debug_vis(self):
        rotor_pos_w = self.asset.data.body_pos_w[..., self.rotor.body_ids, :].flatten(0, -2)
        rotor_quat_w = self.asset.data.body_quat_w[..., self.rotor.body_ids, :].flatten(0, -2)

        throttle = torch.zeros(self.rotor.shape + (3,), device=self.device)
        throttle[..., 2] = self.rotor.throttle
        throttle_w = quat_rotate(rotor_quat_w, throttle.flatten(0, -2))
        self.env.debug_draw.vector(
            rotor_pos_w,
            throttle_w,
        )

        rest_throttle = torch.zeros(self.rotor.shape + (3,), device=self.device)
        rest_throttle[..., 2] = 1.0 - self.rotor.throttle
        rest_throttle_w = quat_rotate(rotor_quat_w, rest_throttle.flatten(0, -2))
        self.env.debug_draw.vector(
            rotor_pos_w + throttle_w,
            rest_throttle_w,
            color=(0., 0., 0., .5)
        )

