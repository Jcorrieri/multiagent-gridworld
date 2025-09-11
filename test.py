import os.path
import time

import pandas as pd
from ray.rllib.algorithms import Algorithm
from ray.rllib.models import ModelCatalog

from environment.envs.baseline import BaselineEnv
from environment.envs.gridworld import GridWorldEnv
from models.rl_wrapper import CustomTorchModelV2
from utils import make_env


def test_one_episode(test_env: GridWorldEnv | BaselineEnv, model: Algorithm, explore: bool):
    observations, _ = test_env.reset()
    episode_over = False
    coverage, total_reward, makespan, num_breaks = 0, 0, 0, 0
    start_ns = time.perf_counter_ns()

    while not episode_over:
        actions = {
            agent: model.compute_single_action(
                observations[agent],
                policy_id="shared_policy",
                explore=explore
            )
            for agent in observations
        } if model else test_env.execute_algorithm()  # for baseline env

        observations, rewards, terminated, truncated, infos = test_env.step(actions)

        coverage = infos['agent_0']['coverage']
        total_reward += sum(rewards.values())
        makespan += 1
        if infos['agent_0']['connection_broken']:
            num_breaks += 1

        # print("\rStep reward:", round(sum(rewards.values()), 2), "Total reward:", round(total_reward, 2), end="")

        episode_over = all(terminated.values()) or all(truncated.values())

    elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2)
    communication_ratio = round((makespan - num_breaks) / makespan * 100, 2)
    coverage = round(coverage, 2)

    return total_reward, makespan, coverage, communication_ratio, elapsed_ms

def build_algo(test_config) -> tuple[Algorithm, str]:
    model = test_config.get('model_version', "v0")
    checkpoint_dir = os.path.join("experiments", "gridworld", model)
    if test_config.get('checkpoint', -1) >= 0:
        checkpoint_dir = os.path.join(checkpoint_dir, "ckpt", str(test_config['checkpoint']))
    else:
        checkpoint_dir = os.path.join(checkpoint_dir, "saved")

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError("Model does not exist, please check \'model_version\' in the config file.")

    ModelCatalog.register_custom_model("shared_cnn", CustomTorchModelV2)
    tester = Algorithm.from_checkpoint(checkpoint_dir)

    return tester, os.path.join("gridworld", model)

def test(env_config, test_config) -> None:
    env_config["seed"] = test_config.get("seed", 42)
    if test_config.get("render", False):
        env_config["render_mode"] = "human"

    is_baseline = env_config['env_name'].strip().lower() == "baseline"
    if is_baseline:
        tester, model_dir = None, "baseline"
    else:
        tester, model_dir = build_algo(test_config)

    print("Testing Parameters:")
    print("-"*50)
    print(f"Seed: {env_config['seed']}")
    print(f"Environment: {env_config['env_name']}")
    print(f"Reward Scheme: {env_config['reward_scheme']}")
    print(f"Model Version: {test_config['model_version']}")
    print("-"*50)

    num_maps = 50
    num_episodes_per_map = test_config.get("num_episodes_per_map", 10)
    num_episodes = num_maps * num_episodes_per_map

    csv_data = []
    if num_episodes > 0:
        print(f"Running {num_episodes} test episodes")
        game_env = make_env(env_config)

        for i in range(num_episodes):
            print(f"\r{i}/{num_episodes}", end="", flush=True)
            reward, makespan, coverage, communication_ratio, elapsed_ms = test_one_episode(
                game_env, tester, test_config.get("explore", False)
            )
            csv_data.append({
                "Episode": i + 1,
                "Makespan": makespan,
                "Coverage": coverage,
                "Communication_Ratio": communication_ratio,
                "Inference_Time_ms": elapsed_ms,
            })
        print("")

        game_env.close()

        num_agents = env_config["num_agents"]
        comm_range = env_config["cr"]

        metrics_path = os.path.join("experiments", model_dir, "test-results", f"{num_agents}_robots_{comm_range}_cr.csv")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        columns = ["Episode", "Makespan", "Coverage", "Communication_Ratio", "Inference_Time_ms"]
        df = pd.DataFrame(csv_data, columns=columns)

        df.to_csv(metrics_path, index=False, mode='w', header=True)

        print(f"Results saved to {metrics_path}")