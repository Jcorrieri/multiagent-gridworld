import os.path

import pandas as pd
from ray.rllib.algorithms import Algorithm
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog

from utils import make_env
from models.rl_wrapper import CustomTorchModelV2


def test_one_episode(test_env: ParallelPettingZooEnv, model: Algorithm, explore: bool):
    observations, _ = test_env.reset()
    episode_over = False
    coverage, total_reward, steps, num_breaks = 0, 0, 0, 0
    while not episode_over:
        actions = {
            agent: model.compute_single_action(
                observations[agent],
                policy_id="shared_policy",
                explore=explore
            )
            for agent in observations
        }

        observations, rewards, terminated, truncated, infos = test_env.step(actions)

        coverage = infos['agent_0']['coverage']
        total_reward += sum(rewards.values())
        steps += 1
        if infos['agent_0']['connection_broken']:
            num_breaks += 1

        episode_over = all(terminated.values()) or all(truncated.values())
    return total_reward, steps, num_breaks, coverage

def test(env_config, test_config) -> None:
    env_config["seed"] = test_config.get("seed", 42)
    if test_config.get("render", False):
        env_config["render_mode"] = "human"

    model = test_config.get("model_path", "default-env/v0")
    checkpoint_dir = os.path.join("experiments", model)
    if test_config.get("checkpoint", -1) >= 0:
        checkpoint_dir = os.path.join(checkpoint_dir, f"ckpt/{test_config['checkpoint']}")
    else:
        checkpoint_dir = os.path.join(checkpoint_dir, "saved")

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError("Model path does not exist, please check \'model_path\' in the config file.")

    print("Testing Parameters:")
    print("-"*50)
    print(f"Seed: {env_config['seed']}")
    print(f"Environment: {env_config['env_name']}")
    print(f"Reward Scheme: {env_config['reward_scheme']}")
    print("-"*50)

    ModelCatalog.register_custom_model("shared_cnn", CustomTorchModelV2)
    tester = Algorithm.from_checkpoint(checkpoint_dir)

    num_episodes = 50 * test_config.get("num_episodes_per_map", 10)

    if num_episodes > 0:
        print(f"Running {num_episodes} test episodes")
        game_env = ParallelPettingZooEnv(make_env(env_config))

        epis_connected, total_reward, total_steps, total_breaks, total_coverage = 0, 0, 0, 0, 0
        for i in range(num_episodes):
            print(f"\r{i}/{num_episodes}", end="")
            reward, steps, num_breaks, coverage = test_one_episode(game_env, tester, test_config.get("explore", False))
            total_reward += reward
            total_coverage += coverage
            total_steps += steps
            epis_connected += 1 if (num_breaks == 0) else 0
            total_breaks += num_breaks
        print("")

        avg_reward = round(total_reward / num_episodes, 2)
        avg_steps = round(total_steps / num_episodes, 2)
        avg_breaks = round(total_breaks / num_episodes, 2)
        avg_coverage = round(total_coverage / num_episodes, 2)

        game_env.close()

        title = f"Averages Over {num_episodes} Test Episodes"
        comm_ratio = epis_connected / num_episodes * 100
        percent_connected = round(100 * (1 - (avg_breaks / avg_steps)), 2)

        print("-"*40)
        print(f"| {title:<36} |")
        print(f"| {'Reward:':<20} {avg_reward:>15} |")
        print(f"| {'Steps:':<20} {avg_steps:>15} |")
        print(f"| {'Coverage:':<20} {avg_coverage:>15}% |")
        print(f"| {'Num Disconnects:':<20} {avg_breaks:>15} |")
        print(f"| {'Percentage Connected:':<20} {percent_connected:>13}% |")
        print(f"| {'Communication Ratio:':<20} {comm_ratio:>15}%")
        print("-"*40)

        csv_data = {
            "Num_Robots": [env_config["num_agents"]],
            "Avg_Reward": [avg_reward],
            "Avg_Num_Disconnects": [avg_breaks],
            "Avg_Connection_Percent": [percent_connected],
            "Communication_Ratio": [comm_ratio],
            "Avg_Duration_Steps": [avg_steps],
            "Avg_Coverage_Percent": [avg_coverage]
        }

        metrics_dir = os.path.join("experiments", model, "test-results/results.csv")

        header = True
        if os.path.exists(metrics_dir):
            header = False

        df = pd.DataFrame(csv_data)
        df.to_csv(metrics_dir, index=False, mode='a', header=header)

        print(f"Results saved to {metrics_dir}/results.csv")