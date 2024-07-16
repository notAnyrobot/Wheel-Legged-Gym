from wheel_legged_gym.learning.algorithms import PPO
from .ppo import ppo_config

configs = {PPO.__name__: ppo_config}

__all__ = ["configs"]