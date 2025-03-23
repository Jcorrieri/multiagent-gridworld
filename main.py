import argparse

import numpy as np
import torch
from ray.rllib.algorithms import Algorithm, PPO
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

import utils
from env.grid_world import GridWorldEnv
from models.cnn import SharedCNNModel
from utils import build_config


def train(args: argparse.Namespace, env_config: dict) -> None:
    trainer = build_config(env_config)

    print(f"Training Parameters:\n--------------------\n{args}")
    print("Model: ", trainer.get_policy("shared_policy").model)
    print("--------------------\nTraining...")

    max_rew_epi_count = 0
    best_score = -np.inf
    for i in range(args.num_episodes):
        result = trainer.train()

        print(f"Episode {i}: reward = {result['episode_reward_mean']}, length: {result['episode_len_mean']}")

        # Stop training if the average reward reaches 200
        if result["episode_reward_mean"] >= 500:
            if result["episode_reward_mean"] > best_score:
                best_score = result["episode_reward_mean"]
                trainer.save_checkpoint("models/ckpt")
            max_rew_epi_count += 1
            if max_rew_epi_count >= 20:
                print("Stopping training - reached target reward.")
                break
        else:
            max_rew_epi_count = 0

    trainer.save("models/saved/mppo")

def test_one_episode(env: ParallelPettingZooEnv, model: Algorithm, seed: int) -> float:
    observations, _ = env.reset(seed=seed)
    episode_over = False
    total_reward = 0.0

    while not episode_over:
        actions = {}
        for agent_id in env.get_agent_ids():
            if agent_id in observations:
                actions[agent_id] = model.compute_single_action(
                    observations[agent_id],
                    policy_id="shared_policy",
                    explore=False,
                )

        observations, rewards, terminated, truncated, infos = env.step(actions)

        total_reward += sum(rewards.values())

        episode_over = all(terminated.values()) or all(truncated.values())

    return total_reward

def test(args, env_config: dict) -> None:
    # Create the test environment
    env_config["render_mode"] = "human"
    game_env = ParallelPettingZooEnv(GridWorldEnv(**env_config))

    tester = build_config(env_config)
    tester.restore(args.model_path)

    print("Testing...")
    print("Model: ", tester.get_policy("shared_policy").model)

    reward = test_one_episode(game_env, tester, seed=args.seed)

    game_env.close()

    print("Results:\n------------------------------")
    print(f"Reward: {reward}")

def main():
    parser = argparse.ArgumentParser()
    utils.parse_optimizer(parser)
    args = parser.parse_args()
    size = args.size
    num_agents = args.num_agents
    communication_range = args.cr
    max_steps = args.max_steps
    obs_mat_path = args.obs_mat_path

    if num_agents > (size * size):
        raise ValueError('Too many agents for given map size')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    obs_mat = np.loadtxt(obs_mat_path, delimiter=' ', dtype='int')
    obs_mat = [(x, y) for x, y in obs_mat]

    env_config = {
        "render_mode": "rgb_array",
        "size": size,
        "num_agents": num_agents,
        "cr": communication_range,
        "max_steps": max_steps,
        "obs_mat": obs_mat
    }

    # for rllib
    ModelCatalog.register_custom_model("shared_cnn", SharedCNNModel)
    register_env("grid_world", lambda cfg: ParallelPettingZooEnv(GridWorldEnv(**cfg)))

    # env = GridWorldEnv(obs_mat=obs_mat, render_mode="human")
    # # parallel_api_test(env, num_cycles=1_000_000)
    # test_one_episode(env, 42)
    # env.close()

    if not args.test:
        train(args, env_config)
    else:
        test(args, env_config)

if __name__ == "__main__":
    main()
