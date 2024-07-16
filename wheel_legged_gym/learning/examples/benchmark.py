"""Inference trained policy using dummy environemnt."""

import os
from datetime import datetime
import torch

from wheel_legged_gym.learning.env.vec_env import VecEnv
from wheel_legged_gym.learning.env.dummy_env import DummyEnv
from wheel_legged_gym.learning.algorithms import PPO
from wheel_legged_gym.learning.runners import OnPolicyRunner

from config import configs
from data import DATA_DIR


ALGORITHM = [PPO]
ENVIRONMENTS = ["wheelwalker"]
ENVIRONMENT_KWARGS = [{}]
DEVICE = ["cpu"]
RUNS = 3

RESUME = False
CHECKPOINT_PATH = os.path.join(DATA_DIR, "policy_1.pt")


def run(alg_class, env_name, env_kwargs={}):

    try:
        config = configs[alg_class.__name__][env_name]
    except KeyError:
        print("No configuration found for the specified algorithm and environment. Using default configuration.")
        config = configs["default"]["default"]

    # Get the environment parameters
    env_kwargs = dict(device=DEVICE[0], **config["env_kwargs"])
    
    # Get the algorithm parameters
    runner_kwargs = dict(device=DEVICE[0], **config["runner_kwargs"])
    algorithm_kwargs = dict(**config["algorithm_kwargs"])
    policy_kwargs = dict(device=DEVICE[0], **config["policy_kwargs"])
    training_kwargs = dict(
        runner = runner_kwargs,
        algorithm = algorithm_kwargs,
        policy = policy_kwargs,
    )

    # Create the environment
    num_obs = env_kwargs["observation_count"]
    num_privileged_obs = env_kwargs["privileged_observation_count"]
    history_horizon = env_kwargs["history_horizon"]
    num_action = env_kwargs["action_count"]
    max_episode_length = env_kwargs["max_episode_length"]
    env = DummyEnv(num_obs, num_privileged_obs, num_action, history_horizon,)
    env.max_episode_length = max_episode_length

    # Create the runner
    runner = make_runner(env, training_kwargs)

    # Run the policy
    obs, extras = env.reset()
    done = False
    runner.learn(num_learning_iterations=1000, init_at_random_ep_len=True)
    # while not done:
    #     action, _ = runner.policy.get_action(obs, extras)
    #     obs, extras, done = env.step(action)


def make_runner(env: VecEnv, training_kwargs: dict) -> OnPolicyRunner:
         

    log_root = os.path.join(DATA_DIR, "logs", "benchmark")
    log_dir = os.path.join(
        log_root, datetime.now().strftime('%b%d_%H_%M_%S') + '_' + "wheelwalker"
    )

    runner = OnPolicyRunner(env, training_kwargs, log_dir)
    
    resume = training_kwargs["runner"]["resume"]
    if resume:
        print(f"Loading model from; {CHECKPOINT_PATH}")
        runner.load(CHECKPOINT_PATH)
        
    return runner
         
                 
def main():
    for alg_class in ALGORITHM:
        for i, env_name in enumerate(ENVIRONMENTS):
            
            env_kwargs = ENVIRONMENT_KWARGS[i]
            
            for _ in range(RUNS):
                
                run(alg_class, env_name, env_kwargs)
        
        
if __name__ == "__main__":
    # RESUME = True
    main()