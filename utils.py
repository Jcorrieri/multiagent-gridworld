import sys

import numpy as np
from scipy.spatial import cKDTree
from stable_baselines3.common.logger import HumanOutputFormat

from gymnasium_env.envs.grid_world import Actions

class MinimalLogger(HumanOutputFormat):
    def write(self, key_values, key_excluded, step=0):
        # Extract relevant information
        steps = key_values.get("time/total_timesteps", "N/A")
        mean_reward = key_values.get("rollout/ep_rew_mean", "N/A")
        loss = key_values.get("train/loss", "N/A")

        # Handle formatting for numeric values
        steps_str = f"{steps}" if steps != "N/A" else "N/A"
        mean_reward_str = f"{float(mean_reward):.2f}" if mean_reward != "N/A" else "N/A"
        loss_str = f"{float(loss):.2f}" if loss != "N/A" else "N/A"

        # Print on a single line
        sys.stdout.write(f"\rSteps: {steps_str} | Mean Reward: {mean_reward_str} | Loss: {loss_str}")
        sys.stdout.flush()

    def close(self):
        pass  # Override base method without needing extra logic

def direction_to_action(direction):
    direction_map = {
        (1, 0): Actions.right.value,
        (0, 1): Actions.up.value,
        (-1, 0): Actions.left.value,
        (0, -1): Actions.down.value,
        (0, 0): Actions.no_op.value,
    }
    return direction_map[tuple(direction)]

def find_closest_frontier(agent_loc, unvisited_nodes):
    tree = cKDTree(unvisited_nodes)
    distance, index = tree.query(agent_loc)
    return unvisited_nodes[index]

def get_neighbors(node, size):
    top, bottom = node + np.array([0, 1]), node + np.array([0, -1])
    left, right = node + np.array([-1, 0]), node + np.array([1, 0])
    potential_neighbors = list([top, bottom, left, right])

    neighbors = []
    for tile in potential_neighbors:
        if 0 <= tile[0] < size and 0 <= tile[1] < size:
            neighbors.append(tile)

    return neighbors

def parse_optimizer(parser):
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--env_name', type=str, default='GridWorld-v0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='ppo')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--render_mode', type=str, default='rgb_array')
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--cr', type=int, default=3)
