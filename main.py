import argparse
import os.path
import warnings

import numpy as np
import torch
from ray.rllib.algorithms import Algorithm
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env
from torch.utils.checkpoint import checkpoint

import utils
from env.grid_world import GridWorldEnv
from utils import build_config


def train(args: argparse.Namespace, env_config: dict, old_api_stack: bool = True) -> None:
    trainer = build_config(env_config, old_api_stack)

    if old_api_stack:
        model = trainer.get_policy("shared_policy").model
    else:
        model = trainer.get_module("shared_policy").model

    print("Training Parameters:")
    print("-"*100 + f"\n{args}")
    print("Model: ", model)
    print("-"*100 + "\nTraining...")

    max_rew_epi_count = 0
    best_score = -np.inf
    for i in range(args.num_episodes):
        result = trainer.train()

        episode_reward_mean = result["env_runners"]['episode_reward_mean']
        episode_len_mean = result["env_runners"]['episode_len_mean']
        print("\rEpisode {i}/{args.num_episodes}, total reward = {episode_reward_mean:.2f}, "
              "average length: {episode_len_mean}".format_map(locals()), end="")

        # Stop training if the average reward reaches 200 per agent
        if episode_reward_mean >= (200 * args.num_agents):
            if episode_reward_mean > best_score:
                best_score = episode_reward_mean
                trainer.save_checkpoint("models/ckpt")
            max_rew_epi_count += 1
            if max_rew_epi_count >= 20:
                print("Stopping training - reached target reward.")
                break
        else:
            max_rew_epi_count = 0

    trainer.save(args.model_path)

def test_one_episode(env: ParallelPettingZooEnv, seed: int, model: Algorithm = None) -> float:
    """Training loop for one episode, taken from Ray official documentation"""
    observations, _ = env.reset(seed=seed)
    episode_over = False
    total_reward = 0.0

    while not episode_over:
        actions = {}

        for agent_id, agent_obs in observations.items():
            rl_module = model.get_module(agent_id) # get agent policy

            # Batch the observation (B=1)
            obs_batch = torch.from_numpy(agent_obs).unsqueeze(0)

            # Run inference
            model_outputs = rl_module.forward_inference({"obs": obs_batch})
            action_dist_params = model_outputs["action_dist_inputs"][0].numpy()

            action = int(np.argmax(action_dist_params))

            actions[agent_id] = action

        observations, rewards, terminated, truncated, infos = env.step(actions)

        total_reward += sum(rewards.values())

        episode_over = all(terminated.values()) or all(truncated.values())

    return total_reward

def test(args, env_config) -> None:
    env_config["render_mode"] = "human"
    game_env = ParallelPettingZooEnv(GridWorldEnv(**env_config))

    checkpoint_dir = os.path.abspath(args.model_path)
    tester = Algorithm.from_checkpoint(checkpoint_dir)

    print("Testing Parameters:")
    print("-" * 100 + f"\n{args}")
    print("Model: ", tester.get_policy("shared_policy").model)
    print("-" * 100)

    test_one_episode(game_env, args.seed, tester)

    game_env.close()

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
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
