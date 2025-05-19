import argparse
import os.path
import warnings

import numpy as np
import torch
import yaml
from ray.rllib.algorithms import Algorithm
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from torch.utils.checkpoint import checkpoint

import utils
from env.grid_world import GridWorldEnv
from models.rl_wrappers import CustomTorchModelV2
from utils import build_config, plot_metrics


def train(args: argparse.Namespace, env_config: dict, training_config: dict) -> None:
    i = 0
    model_name = args.model_name
    while os.path.exists(f"./models/saved/{model_name}"):
        i += 1
        model_name = f"{args.model_name}_{i}"

    ckpt_dir = f"./models/ckpt/{model_name}"
    save_dir = f"./models/saved/{model_name}"
    os.mkdir(ckpt_dir)

    trainer = build_config(env_config, training_config)
    # trainer = Algorithm.from_checkpoint(os.path.abspath(args.model_path))  # transfer learning ..?

    model = trainer.get_policy("shared_policy").model

    print("Training Parameters:")
    print("-"*100 + f"\n{args}")
    print("Model: ", model)
    print("-"*100 + "\nTraining...")

    max_rew_epi_count = 0
    best_score = -np.inf
    data = []
    num_iterations = training_config['num_iterations']
    for i in range(num_iterations):
        result = trainer.train()

        episode_reward_mean = result["env_runners"]['episode_reward_mean']
        episode_len_mean = result["env_runners"]['episode_len_mean']
        print(f"\rIteration {i}/{num_iterations}, total reward = {episode_reward_mean:.2f}, "
              "average length: {episode_len_mean}".format_map(locals()), end="")

        data.append([episode_reward_mean, episode_len_mean])

        if i != 0 and i % 1500 == 0:
            os.mkdir(f"{ckpt_dir}/{(i // 1500)}/")
            trainer.save_checkpoint(f"{ckpt_dir}/{(i // 1500)}/")

        # Stop training if the average reward reaches 500
        if episode_reward_mean >= 500:
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

    pretty_print(f"Metrics for Demo Episode", reward, steps, num_breaks)
    print("Running 30 more test episodes...")

    total_reward, total_steps, total_breaks = 0, 0, 0
    num_test_episodes = 30
    for i in range(num_test_episodes):
        print(f"\r{i}/{num_test_episodes}", end="")
        reward, steps, num_breaks = test_one_episode(game_env, seed=args.seed, model=tester)
        total_reward += reward
        total_steps += steps
        total_breaks += num_breaks
    print("")

    avg_reward = total_reward / num_test_episodes
    avg_steps = total_steps // num_test_episodes
    avg_breaks = total_breaks // num_test_episodes

    game_env.close()

    pretty_print(f"Averages Over {num_test_episodes} Test Episodes", avg_reward, avg_steps, avg_breaks)

def main():
    parser = argparse.ArgumentParser()
    utils.parse_optimizer(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    args.device = device

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(f"config/{args.config}", 'r') as f:
        config = yaml.safe_load(f)

    env_config = dict(
        render_mode="rgb_array",
        rw_scheme=config['reward_scheme'],
        **config['environment']
    )

    # for rllib
    register_env("grid_world", lambda cfg: ParallelPettingZooEnv(GridWorldEnv(**cfg)))
    if args.test:
        test(args, env_config)
    else:
        train(args, env_config, config['training'])

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
