import argparse
import os
import shutil

import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from pettingzoo import ParallelEnv

from environment.env_factory import make_env
from models.rl_wrapper import CustomTorchModelV2
from utils import plot_metrics


def build_config(env_config: dict, training_config: dict):
    dummy_env = make_env(env_config)

    # PPO training parameters
    ppo_params = dict(
        gamma=training_config.get("gamma", 0.9),
        lr=training_config.get("lr", 0.0001),
        grad_clip=training_config.get("grad_clip", 1.0),
        train_batch_size=training_config.get("train_batch_size", 8000),
        num_epochs=training_config.get("num_passes", 5),
        minibatch_size=training_config.get("minibatch_size", 800),
        optimizer={"weight_decay": training_config.get("l2_regularization", 0.001)},
        lambda_=training_config.get("lambda_", 0.95),
        entropy_coeff_schedule=training_config.get("entropy_coeff", None),
        clip_param=training_config.get("clip_param", 0.3)
    )

    config = get_default_config(
        env_config,
        ppo_params,
        training_config.get("module_file", "cnn_1conv3linear.py"),
        dummy_env
    )

    dummy_env.close()
    # config.log_level = "DEBUG"
    return config.build_algo()

def get_default_config(env_config: dict, ppo_params: dict, module_file: str, dummy_env: ParallelEnv) -> PPOConfig:
    ModelCatalog.register_custom_model("shared_cnn", CustomTorchModelV2)

    config = (
        PPOConfig()
        .environment(
            env=env_config.get("env_name", "gridworld"),
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
                "custom_model_config": {
                    "module_file": module_file
                },
            },
            use_gae=True,
            use_critic=True,
            **ppo_params
        )
        .env_runners(
            num_env_runners=2,
            num_envs_per_env_runner=2,
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

def create_model_directories(env_config: dict, args: argparse.Namespace):
    default_env_names = ["default", "alt_reward"]

    if env_config['env_name'] in default_env_names:
        if env_config['base_station']:
            experiment_dir = 'experiments\\base-station'
        else:
            experiment_dir = 'experiments\\default-env'
    else:
        experiment_dir = 'experiments\\baseline'

    model_dir = os.path.join(experiment_dir, 'v0')
    i = 1
    while os.path.exists(model_dir):
        model_dir = os.path.join(experiment_dir, f'v{i}')
        i += 1

    ckpt_dir = os.path.join(model_dir, "ckpt")
    save_dir = os.path.join(model_dir, "saved")
    train_metrics_dir = os.path.join(model_dir, "train-metrics")
    test_result_dir = os.path.join(model_dir, "test-results")

    paths = [ckpt_dir, save_dir, train_metrics_dir, test_result_dir]
    for path in paths:
        if os.path.exists(path):
            os.rmdir(path)
        os.makedirs(path)

    shutil.copy(f"config\\{args.config}", f'{model_dir}\\config')

    return ckpt_dir, save_dir, train_metrics_dir, test_result_dir

def train(args: argparse.Namespace, env_config: dict, training_config: dict) -> None:
    print("Training Parameters:")
    print("-"*50)
    print(f"Using device: {args.device}")
    print(f"Module: {training_config['module_file']}")
    print(f"Environment: {env_config['env_name']}")

    ckpt_dir, save_dir, train_metrics_dir, test_result_dir = create_model_directories(env_config, args)

    print(f"Model Path: {save_dir}")
    print("-"*50)

    print("\nBuilding Ray Trainer...\n")

    trainer = build_config(env_config, training_config)

    policy_id = "shared_policy"
    model = trainer.get_policy(policy_id).model

    print("-"*100 + "\nModel Architecture: ", model)
    print("-"*100 + "\n\nBeginning Training...\n")

    max_rew_epi_count = 0
    ckpt_interval = 200
    target_rew = training_config["target_reward"]
    best_score = -np.inf

    num_episodes = training_config["num_episodes"]
    train_batch_size = training_config["train_batch_size"]
    max_steps = env_config["max_steps"]

    data = []
    episodes_elapsed = 0
    num_iterations = int(num_episodes / (train_batch_size / max_steps))
    for i in range(num_iterations):
        result = trainer.train()

        episode_reward_mean = result["env_runners"]["episode_reward_mean"]
        episode_len_mean = result["env_runners"]["episode_len_mean"]
        episodes_elapsed += result["env_runners"]["num_episodes"]
        print(f"\rIteration {i + 1}/{num_iterations}, "
              f"episode: {episodes_elapsed}, "
              f"total reward: {episode_reward_mean:.2f}, "
              f"average length: {episode_len_mean:.2f}", end="")

        data.append([episode_reward_mean, episode_len_mean, episodes_elapsed])

        if i != 0 and i % ckpt_interval == 0:
            os.mkdir(f"{ckpt_dir}/{(i // ckpt_interval)}/")
            trainer.save_checkpoint(f"{ckpt_dir}/{(i // ckpt_interval)}/")

        # Stop training if the average reward reaches target for 20 consecutive iterations
        if episode_reward_mean >= target_rew:
            if episode_reward_mean > best_score:
                best_score = episode_reward_mean
            max_rew_epi_count += 1
            if max_rew_epi_count >= 20:
                print("Stopping training - reached target reward.")
                break
        else:
            max_rew_epi_count = 0

    print(f"\nSaving to \"{save_dir}\"")
    trainer.save(save_dir)
    plot_metrics(data, train_metrics_dir)