import os.path

from ray.rllib.algorithms import Algorithm
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog

from env.grid_world import GridWorldEnv
from models.rl_wrapper import CustomTorchModelV2


def test_one_episode(test_env: ParallelPettingZooEnv, model: Algorithm):
    observations, _ = test_env.reset()
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

        episode_over = all(terminated.values()) or all(truncated.values())
    return total_reward, steps, num_breaks

def test(args, env_config, test_config) -> None:
    print("Testing Parameters:")
    print("-"*50)
    print(f"Using device: {args.device}")
    print(f"Config: {args.config}")
    print("-"*50)

    env_config["seed"] = env_config.get("seed", 42)

    checkpoint_dir = os.path.abspath(f"experiments/default-env/v1/saved")

    ModelCatalog.register_custom_model("shared_cnn", CustomTorchModelV2)
    tester = Algorithm.from_checkpoint(checkpoint_dir)

    policy_net = "shared_policy"

    print("Model: ", tester.get_policy(policy_net).model)
    print("-" * 50)

    def pretty_print(title: str, rew: float, stp: int, brk: int):
        print("-"*40)
        print(f"| {title:<36} |")
        print(f"| {'Reward:':<20} {round(rew, 2):>15} |")
        print(f"| {'Steps:':<20} {stp:>15} |")
        print(f"| {'Num Disconnects:':<20} {brk:>15} |")
        print(f"| {'Percentage Connected:':<20} {round(100 * (1 - (brk / stp)), 2):>13}% |")
        print("-"*40)

    demo_env = ParallelPettingZooEnv(GridWorldEnv(env_config))

    reward, steps, num_breaks = test_one_episode(demo_env, tester)
    demo_env.close()

    pretty_print(f"Metrics for Demo Episode", reward, steps, num_breaks)

    num_episodes = test_config.get("num_episodes_per_map", 10)
    # num_episodes = 50 * test_config.get("num_episodes_per_map", 10)
    epis_connected = 0
    if num_episodes > 1:
        print(f"Running {num_episodes} more test episodes...")
        game_env = ParallelPettingZooEnv(GridWorldEnv(env_config))

        total_reward, total_steps, total_breaks = 0, 0, 0
        for i in range(num_episodes):
            print(f"\r{i}/{num_episodes}", end="")
            reward, steps, num_breaks = test_one_episode(game_env, tester)
            total_reward += reward
            total_steps += steps
            epis_connected += 1 if (num_breaks == 0) else 0
            total_breaks += num_breaks
        print("")

        avg_reward = round(total_reward / num_episodes, 2)
        avg_steps = round(total_steps / num_episodes, 2)
        avg_breaks = round(total_breaks / num_episodes, 2)

        game_env.close()

        pretty_print(f"Averages Over {num_episodes} Test Episodes", avg_reward, avg_steps, avg_breaks)

        comm_ratio = epis_connected / num_episodes * 100
        print(epis_connected)
        print(f"Communication Ratio: {round(comm_ratio)}%")