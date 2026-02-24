import argparse
import os
import shutil

import numpy as np

from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.models import ModelCatalog

from utils import make_env, plot_metrics
from models.rl_wrapper import CustomTorchModelV2

def build_config(env_config: dict, training_config: dict):
    dummy_env = make_env(env_config)

    # Filter and map parameters for DQN
    dqn_params = {
        "gamma": training_config.get("gamma", 0.90),
        "lr": training_config.get("lr", 0.0001),
        "train_batch_size": training_config.get("train_batch_size", 32),
        "buffer_size": training_config.get("buffer_size", 50000),
        "epsilon": training_config.get("epsilon", [[0, 1.0], [17500000, 0.01], [35000000, 0.01]]),
        "target_network_update_freq": training_config.get("target_network_update_freq", 1500)
    }

    config = get_default_config(
        env_config,
        dqn_params,
        training_config.get("module_file", "dqn_cnn.py"),
        dummy_env
    )

    dummy_env.close()
    return config.build_algo()

def get_default_config(env_config: dict, training_params: dict, module_file: str, dummy_env):
    ModelCatalog.register_custom_model("shared_cnn", CustomTorchModelV2)

    config = (
        DQNConfig()
        .environment(
            env=env_config.get("env_name", "gridworld"),
            env_config=env_config,
        )
        .framework("torch")
        .training(
            gamma=training_params["gamma"],
            lr=training_params["lr"],
            train_batch_size=training_params["train_batch_size"],

            double_q=True, 
            dueling=False,

            replay_buffer_config={
                'type': "MultiAgentPrioritizedReplayBuffer",
                'prioritized_replay_alpha': 0.6,
                'capacity': training_params["buffer_size"]
            },

            epsilon=training_params["epsilon"],

            target_network_update_freq=training_params["target_network_update_freq"],   

            model={
                "custom_model": "shared_cnn",
                "custom_model_config": {
                    "module_file": module_file,
                    "num_agents": env_config.get("num_agents", 5),
                    "disable_preprocessor": True
                },
            },
        )
        .multi_agent(
            policies={
                "shared_policy": (
                    None, 
                    dummy_env.observation_space("agent_0"), 
                    dummy_env.action_space("agent_0"), 
                    {}
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )
        .env_runners(
            num_env_runners=6,
            num_envs_per_env_runner=4,
            rollout_fragment_length="auto", # match 'steps' param in paper
        )
        .resources(
            num_gpus=1
        )
        .evaluation(
            evaluation_num_env_runners=0,
            evaluation_interval=None
        )
        .debugging(
            seed=42
        )
        .api_stack(
            enable_env_runner_and_connector_v2=False,
            enable_rl_module_and_learner=False,
        )
    )
    return config

def create_model_directories(env_config: dict, args: argparse.Namespace):
    env_name = env_config.get('env_name', "gridworld")
    experiment_dir = os.path.abspath(os.path.join("experiments", env_name))

    if env_name != 'gridworld' and env_name != 'baseline':
        raise FileNotFoundError("Please provide a valid environment name")

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

    source_path = os.path.join("config", args.config)
    dest_path = os.path.join(model_dir, "config")
    shutil.copy(source_path, dest_path)

    return ckpt_dir, save_dir, train_metrics_dir, test_result_dir

def train(args: argparse.Namespace, env_config: dict, training_config: dict) -> None:
    print("Training Parameters:")
    print("-"*50)
    print(f"Using device: {args.device}")
    print(f"Module: {training_config['module_file']}")
    print(f"Environment: {env_config['env_name']}")
    print(f"Reward Scheme: {env_config['reward_scheme']}")

    ckpt_dir, save_dir, train_metrics_dir, test_result_dir = create_model_directories(env_config, args)

    print(f"Model Path: {save_dir}")
    print("-"*50)

    print("\nBuilding Ray Trainer...\n")

    model_to_restore = training_config.get("restore_from_model", None)
    if model_to_restore:
        training_config.pop("restore_from_model")

    trainer = build_config(env_config, training_config)

    if model_to_restore:
        model_to_restore = os.path.join("experiments", "gridworld", model_to_restore, "saved")
        model_to_restore = os.path.abspath(model_to_restore)
        trainer.restore(model_to_restore)

    print("-"*100 + "\n\nBeginning Training...\n")

    num_epochs = training_config["num_epochs"]

    max_rew_iter_count = 0
    ckpts = np.linspace(0, num_epochs, num=5, dtype=int)
    ckpt_idx = 0
    target_rew = training_config["target_reward"]
    best_score = -np.inf

    data = []
    episodes_elapsed = 0
    while episodes_elapsed < num_epochs:
        result = trainer.train()

        episode_reward_mean = result["env_runners"]["episode_reward_mean"]
        episode_len_mean = result["env_runners"]["episode_len_mean"]
        episodes_elapsed += result["env_runners"]["num_episodes"]
        current_steps = result["timesteps_total"]
        print(f"\rEpisode: {episodes_elapsed}/{num_epochs}, "
            #   f"episode: {episodes_elapsed}, "
              f"total reward: {episode_reward_mean:.2f}, "
              f"average length: {episode_len_mean:.2f}", end="")

        data.append([episode_reward_mean, episode_len_mean, episodes_elapsed])

        if episodes_elapsed >= ckpts[ckpt_idx]:
            full_ckpt_dir = os.path.join(ckpt_dir, str(ckpt_idx))
            os.makedirs(full_ckpt_dir, exist_ok=True)
            trainer.save_checkpoint(full_ckpt_dir)
            ckpt_idx += 1

        # stop training if the average reward reaches target for 20 consecutive iterations
        if episode_reward_mean >= target_rew:
            if episode_reward_mean > best_score:
                best_score = episode_reward_mean
            max_rew_iter_count += 1
            if max_rew_iter_count >= 20:
                print("Stopping training - reached target reward.")
                break
        else:
            max_rew_iter_count = 0

    print(f"\nSaving to \"{save_dir}\"")
    trainer.save(save_dir)
    plot_metrics(data, train_metrics_dir)
