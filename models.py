import random
from argparse import Namespace

import numpy as np
import torch.nn as nn

import utils
from gymnasium_env.envs.grid_world import Actions


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()


class BasicRandomPolicy(nn.Module):
    def __init__(self, args: Namespace):
        super(BasicRandomPolicy, self).__init__()
        self.args = args

    def forward(self, x):
        moves = np.array([random.randint(0, 4) for _ in range(self.args.num_agents)])

        return moves


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
            # calculate closest nonvisited point
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