import argparse
import os.path
import warnings

import numpy as np
import torch
from ray.rllib.algorithms import Algorithm
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from torch.utils.checkpoint import checkpoint

import utils
from env.grid_world import GridWorldEnv
from models.rl_wrappers import CustomTorchModelV2
from utils import build_config, plot_metrics


def train(args: argparse.Namespace, env_config: dict, old_api_stack: bool = True) -> None:
    i = 0
    save_path = f"models/saved/{args.model_name}"
    while os.path.exists(save_path):
        i += 1
        save_path = f"models/saved/{args.model_name}_{i}"

    trainer = build_config(env_config)
    # trainer = Algorithm.from_checkpoint(os.path.abspath(args.model_path))  # transfer learning ..?

    model = trainer.get_policy("shared_policy").model

    print("Training Parameters:")
    print("-"*100 + f"\n{args}")
    print("Model: ", model)
    print("-"*100 + "\nTraining...")

    max_rew_epi_count = 0
    best_score = -np.inf
    data = []
    for i in range(args.num_iterations):
        result = trainer.train()

        episode_reward_mean = result["env_runners"]['episode_reward_mean']
        episode_len_mean = result["env_runners"]['episode_len_mean']
        print("\rIteration {i}/{args.num_iterations}, total reward = {episode_reward_mean:.2f}, "
              "average length: {episode_len_mean}".format_map(locals()), end="")

        data.append([episode_reward_mean, episode_len_mean])

        if i % 1500 == 0:
            os.mkdir(f"{save_path}/{(i // 1500)}/")
            trainer.save_checkpoint(f"{save_path}/{(i // 1500)}/")

        # Stop training if the average reward reaches 460
        if episode_reward_mean >= 460:
            if episode_reward_mean > best_score:
                best_score = episode_reward_mean
            max_rew_epi_count += 1
            if max_rew_epi_count >= 20:
                print("Stopping training - reached target reward.")
                break
        else:
            max_rew_epi_count = 0

    print(f"Saving to {save_path} using model name: {args.model_name if i == 0 else f'{args.model_name}_{i}'}")
    trainer.save(save_path)
    plot_metrics(data, args.model_name if i == 0 else f'{args.model_name}_{i}')

def test_one_episode(test_env: ParallelPettingZooEnv, seed: int | None, model: Algorithm):
    observations, _ = test_env.reset(seed=seed)
    episode_over = False
    total_reward, steps, num_breaks = 0, 0, 0
    while not episode_over:
        actions = {
            agent: model.compute_single_action(
                observations[agent],
                policy_id="shared_policy",
                explore=True
            )
            for agent in observations
        }

        observations, rewards, terminated, truncated, infos = test_env.step(actions)

        total_reward += sum(rewards.values())
        steps += 1
        if infos['agent_0']['connection_broken']:
            num_breaks += 1

        episode_over = terminated.get("__all__", False) or truncated.get("__all__", False)
    return total_reward, steps, num_breaks

def test(args, env_config) -> None:
    game_env = ParallelPettingZooEnv(GridWorldEnv(**env_config))

    checkpoint_dir = os.path.abspath(f"models/saved/{args.model_name}")
    ModelCatalog.register_custom_model("shared_cnn", CustomTorchModelV2)
    tester = Algorithm.from_checkpoint(checkpoint_dir)

    print("Testing Parameters:")
    print("-" * 100 + f"\n{args}")
    print("Model: ", tester.get_policy("shared_policy").model)
    print("-" * 100)

    def pretty_print(title: str, rew: float, stp: int, brk: int):
        print("-"*40)
        print(f"| {title:<36} |")
        print(f"| {'Reward:':<20} {round(rew, 2):>15} |")
        print(f"| {'Steps:':<20} {stp:>15} |")
        print(f"| {'Num Disconnects:':<20} {brk:>15} |")
        print(f"| {'Percentage Connected:':<20} {round(100 * (1 - (brk / stp)), 2):>13}% |")
        print("-"*40)

    env_config["render_mode"] = "human"
    demo_env = ParallelPettingZooEnv(GridWorldEnv(**env_config))

    reward, steps, num_breaks = test_one_episode(demo_env, args.seed, tester)
    demo_env.close()

    pretty_print(f"Metrics for Demo Episode [Seed={args.seed}]", reward, steps, num_breaks)
    print("Running 30 more test episodes...")

    total_reward, total_steps, total_breaks = 0, 0, 0
    for i in range(30):
        print(f"\r{i}/30", end="")
        reward, steps, num_breaks = test_one_episode(game_env, seed=None, model=tester)
        total_reward += reward
        total_steps += steps
        total_breaks += num_breaks
    avg_reward = total_reward / 10
    avg_steps = total_steps // 10
    avg_breaks = total_breaks // 10

    game_env.close()

    pretty_print("Averages Over 10 Test Episodes", avg_reward, avg_steps, avg_breaks)

def main():
    parser = argparse.ArgumentParser()
    utils.parse_optimizer(parser)
    args = parser.parse_args()
    size = args.size
    num_agents = args.num_agents
    communication_range = args.cr
    max_steps = args.max_steps
    obs_mat_path = f"env/obstacle_mats/{args.obs_mat}"

    if num_agents > (size * size):
        raise ValueError('Too many agents for given map size')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    args.device = device

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    obs_mat = np.loadtxt(obs_mat_path, delimiter=' ', dtype='int')
    obs_mat = [(x, y) for x, y in obs_mat]

    env_config = dict(
        render_mode="rgb_array",
        size=size,
        num_agents=num_agents,
        cr=communication_range,
        max_steps=max_steps,
        obs_mat=obs_mat
    )

    # for rllib
    register_env("grid_world", lambda cfg: ParallelPettingZooEnv(GridWorldEnv(**cfg)))

    if not args.test:
        train(args, env_config)
    else:
        test(args, env_config)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
