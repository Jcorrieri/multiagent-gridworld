from argparse import Namespace

import numpy as np
import torch.nn as nn
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import utils
from gymnasium_env.envs.grid_world import Actions


class CustomPPO(nn.Module):
    def __init__(self, args: Namespace):
        super(CustomPPO, self).__init__()
        self.model = PPO.load("PPO_test") if args.test else None
        self.args = args

    def forward(self, x):
        # flatten obs (x)
        flat_map = x['map'].ravel()
        flat_agents = x['agents'].ravel()
        obs = np.concatenate((flat_agents, flat_map))

        action, _states = self.model.predict(np.array(obs))
        return action

    def train_ppo(self):
        train_env = FlattenObservation(self.args.env)
        check_env(train_env, warn=True)
        model = PPO("MlpPolicy", train_env, verbose=1)
        model.learn(total_timesteps=25000)
        model.save("PPO_test")
        self.model = model

class RandomPolicy(nn.Module):
    def __init__(self, args: Namespace):
        super(RandomPolicy, self).__init__()
        self.args = args

    def forward(self, x):
        return self.args.env.action_space.sample()

class FrontierPolicy(nn.Module):
    def __init__(self, args: Namespace):
        super(FrontierPolicy, self).__init__()
        self.args = args

    def forward(self, x):
        obs = x
        moves = np.array([Actions.no_op.value for _ in range(self.args.num_agents)], dtype=int)

        unvisited_nodes = [
            (i, j) for i in range(len(obs['map'])) for j in range(len(obs['map'][i])) if not obs['map'][i][j]
        ]
        # assuming shortest distance is available
        for i, agent in enumerate(obs['agents']):
            # calculate closest non visited point
            target = utils.find_closest_frontier(agent, unvisited_nodes)
            if not target:
                moves[i] = Actions.no_op.value
                continue

            distance = np.linalg.norm(target - agent)
            neighbors = utils.get_neighbors(agent, self.args.size)
            for neighbor in neighbors:
                direction = utils.direction_to_action(neighbor - agent)
                if np.linalg.norm(target - neighbor) < distance:
                    moves[i] = direction
        return moves