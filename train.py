import argparse
import os

import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

from env.grid_world import GridWorldEnv
from models.rl_wrapper import CustomTorchModelV2
from utils import plot_metrics


def build_config(env_config: dict, training_config: dict):
    dummy_env = GridWorldEnv(**env_config)

    # PPO training parameters
    ppo_params = dict(
        gamma=training_config["gamma"],
        lr=training_config["lr"],
        grad_clip=training_config["grad_clip"],
        train_batch_size=training_config["train_batch_size"],
        num_epochs=training_config["num_passes"],
        minibatch_size=training_config["minibatch_size"],
        optimizer={"weight_decay": training_config["l2_regularization"]},
        lambda_=0.9,
        entropy_coeff=0.01,
    )

    config = get_default_config(env_config, ppo_params, dummy_env)

    dummy_env.close()
    # config.log_level = "DEBUG"
    return config.build_algo()

def get_default_config(env_config: dict, ppo_params: dict, dummy_env: GridWorldEnv) -> PPOConfig:
    ModelCatalog.register_custom_model("shared_cnn", CustomTorchModelV2)

    config = (
        PPOConfig()
        .environment(
            env="gridworld",
            env_config=env_config,
        )
        .framework("torch")
        .multi_agent(
            policies={
                "shared_policy": PolicySpec(
                    policy_class=None,  # Default to PPO
                    observation_space=dummy_env.observation_space("agent_0"),
                    action_space=dummy_env.action_space("agent_0"),
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )
        .training(
            model={
                "custom_model": "shared_cnn",
            },
            use_gae=True,
            use_critic=True,
            **ppo_params
        )
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto"
        )
        .resources(
            num_gpus=1
        )
        .evaluation(
            evaluation_num_env_runners=0,
            evaluation_interval=None
        )
        .api_stack(
            enable_env_runner_and_connector_v2=False,
            enable_rl_module_and_learner=False,
        )
    )

    return config

def create_model_directories(args: argparse.Namespace):
    i = 0
    model_name = args.model_name
    while os.path.exists(f"./models/saved/{model_name}"):
        i += 1
        model_name = f"{args.model_name}_{i}"

    ckpt_dir = f"./models/ckpt/{model_name}"
    if os.path.exists(ckpt_dir):
        os.rmdir(ckpt_dir)

    save_dir = f"./models/saved/{model_name}"
    os.mkdir(ckpt_dir)

    return model_name, save_dir, ckpt_dir

def train(args: argparse.Namespace, env_config: dict, training_config: dict) -> None:
    print("Training Parameters:")
    print("-"*50)
    print(f"Using device: {args.device}")
    print(f"Model Name: {args.model_name}")
    print(f"Config: {args.config}")
    print("-"*50)

    print("\nBuilding Ray Trainer...\n")

    model_name, save_dir, ckpt_dir = create_model_directories(args)

    trainer = build_config(env_config, training_config)

    policy_id = "shared_policy"
    model = trainer.get_policy(policy_id).model

    print("-"*100 + "\nModel Architecture: ", model)
    print("-"*100 + "\n\nBeginning Training...\n")

    max_rew_epi_count = 0
    target_rew = training_config['target_reward']
    best_score = -np.inf
    data = []
    num_iterations = training_config['num_episodes']
    for i in range(num_iterations):
        result = trainer.train()

        episode_reward_mean = result["env_runners"]['episode_reward_mean']
        episode_len_mean = result["env_runners"]['episode_len_mean']
        print(f"\rIteration {i}/{num_iterations}, total reward = {episode_reward_mean:.2f}, average length: {episode_len_mean}", end="")

        data.append([episode_reward_mean, episode_len_mean])

        if i != 0 and i % 1500 == 0:
            os.mkdir(f"{ckpt_dir}/{(i // 1500)}/")
            trainer.save_checkpoint(f"{ckpt_dir}/{(i // 1500)}/")

        # Stop training if the average reward reaches target for 20 consecutive episodes
        if episode_reward_mean >= target_rew:
            if episode_reward_mean > best_score:
                best_score = episode_reward_mean
            max_rew_epi_count += 1
            if max_rew_epi_count >= 20:
                print("Stopping training - reached target reward.")
                break
        else:
            max_rew_epi_count = 0

    print(f"\nSaving to \"{save_dir}\" using model name: {model_name}")
    trainer.save(save_dir)
    plot_metrics(data, model_name)