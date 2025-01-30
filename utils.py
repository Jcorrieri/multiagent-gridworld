import numpy as np
from scipy.spatial import cKDTree

from gymnasium_env.envs.grid_world import Actions

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
    # Convert unvisited nodes into a list of tuples for KDTree
    tree = cKDTree(unvisited_nodes)

    # Query the tree to find the closest node to the robot's position
    distance, index = tree.query(agent_loc)

    # Return the closest node
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
    parser.add_argument('--model', type=str, default='greedy')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--render_mode', type=str, default='rgb_array')
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--cr', type=int, default=3)
