from argparse import Namespace
import numpy as np
import torch.nn as nn
import utils
from gymnasium_env.envs.grid_world import Actions


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

        grid, agents = np.zeros((self.args.size, self.args.size), dtype=int), []
        i = 0
        while i < self.args.num_agents * 2:
            agents.append(np.array([x[i], x[i+1]]))
            i += 2

        r, c = 0, 0
        for i in range((self.args.num_agents*2), len(obs)):
            grid[r, c] = obs[i]
            if c == self.args.size - 1:
                c = 0
                r += 1
            else:
                c += 1

        unvisited_nodes = [(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if not grid[i][j]]
        # assuming shortest distance is available
        for i, agent in enumerate(agents):
            # calculate closest non visited point
            target = utils.find_closest_frontier(agent, unvisited_nodes)
            if not target:
                moves[i] = Actions.no_op.value
                continue
            target = np.array(target)
            distance = np.linalg.norm(target - agent)
            neighbors = utils.get_neighbors(agent, self.args.size)
            for neighbor in neighbors:
                direction = utils.direction_to_action(neighbor - agent)
                if np.linalg.norm(target - neighbor) < distance:
                    moves[i] = direction
        return moves