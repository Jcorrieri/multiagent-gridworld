import argparse
import gymnasium
import numpy as np
import torch

import gymnasium_env
import utils

from models import BasicRandomPolicy, FrontierPolicy


def train_one_epoch(env, model):
    episode_over = False
    observation, info = env.reset()

    episode_reward = 0.0
    while not episode_over:
        actions = model(observation)
        # Step the environment with actions
        observation, reward, terminated, truncated, info = env.step(actions)

        episode_reward = reward
        # Determine if the episode is over
        episode_over = terminated or truncated
    return episode_reward

def train(args, env, model):
    total_rewards = []
    for episode in range(args.num_episodes):
        print("\repisode: {}/{}".format(episode + 1, args.num_episodes), end="")
        total_rewards.append(train_one_epoch(env, model))
    print("\n--------------------")
    print("Mean Reward: {mean:.2f}, Std: {std:.2f} \nMin Reward: {min:.2f}, Max Reward: {max:.2f}"
          .format(mean=np.mean(total_rewards), std=np.std(total_rewards), min=np.min(total_rewards),
                  max=np.max(total_rewards)))

def main():
    parser = argparse.ArgumentParser()
    utils.parse_optimizer(parser)
    args = parser.parse_args()

    size = args.size
    num_agents = args.num_agents
    if num_agents > (size * size):
        raise ValueError('Too many agents for given map size')

    cr = args.cr  # communication range
    render_mode = args.render_mode

    env = gymnasium.make('gymnasium_env/'+args.env_name, render_mode=render_mode, size=size, num_agents=num_agents, cr=cr)
    args.env = env

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.model == 'basic-random':
        model = BasicRandomPolicy(args)
    elif args.model == 'greedy':
        model = FrontierPolicy(args)
    else:
        raise ValueError('Invalid Model')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not args.test:
        print("Training Parameters:\n--------------------\n{}".format(args))
        print("Model:\n{}".format(model))
        print("--------------------\nTraining...")
        train(args, env, model)

if __name__ == "__main__":
    main()
