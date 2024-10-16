import os
from datetime import datetime

import isaacgym
import numpy as np
import torch

from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, task_registry


def test(args):
    # set test wheel_legged_gym arguments
    # args.task = "wheelwalker"
    args.num_envs = 11
    # args.headless = True

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )
    task_registry.save_cfgs(name=args.task)
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    test(args)
