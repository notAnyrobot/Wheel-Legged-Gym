from wheel_legged_gym.envs.base.legged_robot import LeggedRobot


class Twip(LeggedRobot):

    def _compute_torques(self, actions):
        # actions = [left_wheel, right_wheel]
        action_pos = actions * self.control.pos_action_scale
        action_vel = actions * self.control.vel_action_scale
        torques = self.p_gains * (
            action_pos + self.default_dof_pos - self.dof_pos
        ) + self.d_gains * (action_vel - self.dof_vel)
