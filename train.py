import numpy as np


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