from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class TwipCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_observations = (
            15  # 3 + 3 + 3 + num_joints + num_joints + num_actions
        )
        num_privileged_obs = (
            num_observations + 7 * 11 + 3 + 6 * 5 + 3 + 3
        )  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 2

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        default_joint_angles = {
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        pos_action_scale = [0.0, 0.0]
        vel_action_scale = [1.0, 1.0]
        stiffness = {
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        }
        damping = {
            "left_wheel_joint": 0.5,
            "right_wheel_joint": 0.5,
        }

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/wheelwalker/urdf/twip.urdf"
        name = "twip"
        offset = 0.054
        l1 = 0.25
        l2 = 0.25
        penalize_contacts_on = ["base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.51


class TwipCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "twip"
