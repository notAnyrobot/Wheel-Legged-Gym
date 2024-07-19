# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class WheelWalkerCfg(LeggedRobotCfg):
    
    class env(LeggedRobotCfg.env):
        num_actions = 6

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "left_hip_pitch_joint": 0.0,
            "left_knee_pitch_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_hip_pitch_joint": -0.0,
            "right_knee_pitch_joint": -0.0,
            "right_wheel_joint": 0.0,
            "tail_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        # PD Drive parameters:
        stiffness = {
            "hip_pitch_joint": 40.0, 
            "knee_pitch_joint": 40.0, 
            "wheel_joint": 0
        }  # [N*m*s/rad]
        damping = {
            "hip_pitch_joint": 1.0, 
            "knee_pitch_joint": 1.0, 
            "wheel_joint": 0.5
        }  # [N*m*s/rad]

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/wheelwalker/urdf/wheelwalker_v1.urdf"
        name = "wheelwalker_v1"
        offset = 0.054
        l1 = 0.25
        l2 = 0.25
        penalize_contacts_on = ["base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        
    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.5


class WheelWalkerCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "wheelwalker"
