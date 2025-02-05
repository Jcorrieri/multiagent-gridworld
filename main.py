import argparse
import random

import gymnasium
import numpy as np
import torch

import utils
from models.baselines import FrontierPolicy, RandomPolicy
from models.ppo import CustomPPO

def train_one_epoch(env, model):
    episode_over = False
    obs, info = env.reset()

    episode_reward = 0.0
    while not episode_over:
        actions = model(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        episode_reward = reward
        episode_over = terminated or truncated
    return episode_reward

def train(args, model):
    total_rewards = []
    for episode in range(args.num_episodes):
        print("\repisode: {}/{}".format(episode + 1, args.num_episodes), end="")
        total_rewards.append(train_one_epoch(args.env, model))
    print("\n--------------------")
    print("Mean Reward: {mean:.2f}, Std: {std:.2f} \nMin Reward: {min:.2f}, Max Reward: {max:.2f}"
          .format(mean=np.mean(total_rewards), std=np.std(total_rewards), min=np.min(total_rewards),
                  max=np.max(total_rewards)))

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
    render_mode = args.render_mode

    if num_agents > (size * size):
        raise ValueError('Too many agents for given map size')
    env = gymnasium.make('gymnasium_env/'+args.env_name, render_mode=render_mode, size=size, num_agents=num_agents, cr=cr)
    args.env = env

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = CustomPPO(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not args.test:
        print("Training Parameters:\n--------------------\n{}".format(args))
        print("Model:\n{}".format(model))
        print("--------------------\nTraining...")
        if type(model) is CustomPPO:
            model.train_ppo()
        else:
            train(args, model)
    else:
        test(args, model)

if __name__ == "__main__":
    main()
