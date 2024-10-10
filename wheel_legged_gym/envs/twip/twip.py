import torch
from isaacgym.torch_utils import quat_rotate_inverse

from wheel_legged_gym.envs.base.legged_robot import LeggedRobot


class Twip(LeggedRobot):

    def _compute_torques(self, actions):
        # actions = [left_wheel, right_wheel]
        action_pos = actions * torch.tensor(
            self.cfg.control.pos_action_scale, device=self.device
        )
        action_vel = actions * torch.tensor(
            self.cfg.control.vel_action_scale, device=self.device
        )
        torques = self.p_gains * (
            action_pos + self.default_dof_pos - self.dof_pos
        ) + self.d_gains * (action_vel - self.dof_vel)

        return torch.clamp(
            torques * self.torques_scale,
            -self.torque_limits,
            self.torque_limits,
        )

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel = (
            self.base_position - self.last_base_position
        ) / self.dt
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.base_lin_vel
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_base_position[:] = self.base_position[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
