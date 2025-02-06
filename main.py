import argparse
import random
import sys

import gymnasium
import torch
from stable_baselines3.common.logger import Logger

import utils
from models.benchmarks import FrontierPolicy, RandomPolicy
from models.dqn_sb import CustomDQN
from models.ppo import CustomPPO
from train import train


def test_step(env, model, seed):
    obs, info = env.reset(seed=seed)
    episode_over = False
    total_reward = 0.0
    while not episode_over:
        actions = model(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        episode_over = terminated or truncated
    return total_reward

def test(args, model):
    # test models and compare
    print("Testing...")
    test_seed = args.seed + random.randint(1, 10000)
    game_env = gymnasium.make('gymnasium_env/' + args.env_name, render_mode="human", size=args.size,
                              num_agents=args.num_agents, cr=args.cr)
    results = [[model, test_step(game_env, model, test_seed)]]

    baselines = [FrontierPolicy(args), RandomPolicy(args)]
    for policy in baselines:
        results.append([policy, test_step(args.env, policy, test_seed)])

    print("Results:\n------------------------------")
    for r in results:
        print("Model:", r[0])
        print("Total Reward: {reward:0.2f}/{max_reward}".format(reward=r[1], max_reward=args.size**2 + args.size))
        print("------------------------------")


def main():
    parser = argparse.ArgumentParser()
    utils.parse_optimizer(parser)
    args = parser.parse_args()
    size = args.size
    num_agents = args.num_agents
    cr = args.cr  # communication range

    if num_agents > (size * size):
        raise ValueError('Too many agents for given map size')
    env = gymnasium.make('gymnasium_env/'+args.env_name, size=size, render_mode='rgb_array', num_agents=num_agents, cr=cr)
    args.env = env

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # TODO -- replace w generic model loading
    if args.model == 'PPO':
        model = CustomPPO(args)
    elif args.model == 'DQN':
        model = CustomDQN(args)
    else:
        model = FrontierPolicy(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if not args.test:
        print("Training Parameters:\n--------------------\n{}".format(args))
        print("Model:\n{}".format(model))
        print("--------------------\nTraining...")
        logger = Logger(folder=None, output_formats=[utils.MinimalLogger(sys.stdout)])
        if type(model) is CustomPPO or type(model) is CustomDQN:
            model.learn(logger)
        else:
            train(args, model)
    else:
        test(args, model)

if __name__ == "__main__":
    main()
