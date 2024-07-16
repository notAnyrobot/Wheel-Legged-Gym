""" A dummy environment for testing purposes. """

from __future__ import annotations

import torch
from wheel_legged_gym.learning.env.vec_env import VecEnv


class DummyEnv(VecEnv):
    """Dummy environment for testing purposes.

    The dummy environment is a simple environment that returns a random observation and reward. The environment
    is used for testing purposes to verify that the wheel_legged_gym is working as expected.
    """

    def __init__(
        self,
        num_obs: int,
        num_privileged_obs: int,
        num_actions: int,
        history_horizon: int = 1,
        **kwargs
    ) -> None:
        """Initialize the dummy environment.

        Args:
            num_obs (int): Number of observations.
            num_privileged_obs (int): Number of privileged observations.
            history_horizon (int): History horizon.
        """
        super().__init__(num_obs, num_privileged_obs, **kwargs)
        self.num_actions = num_actions
        self.obs_history_length = history_horizon
        
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), dtype=torch.float32)
        self.obs_history_buf = torch.zeros((self.num_envs, history_horizon, self.num_obs), dtype=torch.float32)
        self.actions_buf = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.float32)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32)
        self.extras = {
            "observation_history": self.obs_history_buf,
        }

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Return the current observations.

        Returns:
            Tuple[torch.Tensor, dict]: Tuple containing the observations and extras.
        """
        return self.obs_buf, self.extras["observation_history"].clone().detach().reshape(self.num_envs, -1)
    
    def get_privileged_observations(self) -> torch.Tensor:
        """Return the current privileged observations.

        Returns:
            torch.Tensor: The privileged observations.
        """
        return self.privileged_obs_buf

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the dummy environment.

        Returns:
            Tuple[torch.Tensor, dict]: Tuple containing the observations and extras.
        """
        self.obs_buf = torch.rand((self.num_envs, self.num_obs), dtype=torch.float32)
        self.privileged_obs_buf = torch.rand((self.num_envs, self.num_privileged_obs), dtype=torch.float32)
        self.obs_history_buf = torch.rand((self.num_envs, self.obs_history_buf.shape[1], self.num_obs), dtype=torch.float32)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.float32)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32)
        self.extras = {
            "observation_history": self.obs_history_buf,
        }
        
        return self.obs_buf, self.extras


    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the dummy environment.

        Args:
            actions (torch.Tensor): Actions to take in the environment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]: Tuple containing the observations, rewards, resets,
                and extras.
        """
        self.obs_buf = torch.rand((self.num_envs, self.num_obs), dtype=torch.float32)
        self.privileged_obs_buf = torch.rand((self.num_envs, self.num_privileged_obs), dtype=torch.float32)
        self.obs_history_buf = torch.cat((self.obs_buf.unsqueeze(1), self.obs_history_buf[:, :-1, :]), dim=1)
        self.rew_buf = torch.rand(self.num_envs, dtype=torch.float32)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.float32)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32)
        self.extras = {
            "observation_history": self.obs_history_buf,
        }    

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.extras["observation_history"].clone().detach().reshape(self.num_envs, -1),
        ) 
    