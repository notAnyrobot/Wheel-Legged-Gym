import copy
import numpy as np

default = dict()
default["env_kwargs"] = dict(
    environment_count=1,
    observation_count = 27,
    privileged_observation_count = 27 + 7 * 11 + 3 + 6 * 5 + 3 + 3,
    history_horizon = 5,
    action_count = 6,
)
default["runner_kwargs"] = dict(
    policy_class_name = "ActorCriticSequence",
    algorithm_class_name = "PPO",
    num_steps_per_env = 48,  # per iteration
    save_interval = 100,
    resume = False,
)
default["algorithm_kwargs"] = dict(
    # training params
    value_loss_coef = 1.0,
    use_clipped_value_loss = True,
    clip_param = 0.2,
    entropy_coef = 0.01,
    num_learning_epochs = 5,
    num_mini_batches = 4,  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.0e-3,  # 5.e-4
    schedule = "adaptive",  # could be adaptive, fixed
    gamma = 0.99,
    lam = 0.95,
    desired_kl = 0.005,
    max_grad_norm = 1.0,
    extra_learning_rate = 1e-3,
)
default["policy_kwargs"] = dict(
    init_noise_std = 0.5,
    actor_hidden_dims = [128, 64, 32],
    critic_hidden_dims = [256, 128, 64],
    activation = "elu",  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    # only for ActorCriticSequence
    num_encoder_obs = 5 * 27,
    latent_dim = 3,  # at least 3 to estimate base linear velocity
    encoder_hidden_dims = [128, 64],
)

wheelwalker = copy.deepcopy(default)
wheelwalker["env_kwargs"]["max_episode_length"] = 100
wheelwalker["runner_kwargs"]["resume"] = True

ppo_config = {
    "default": default,
    "wheelwalker": wheelwalker,
}