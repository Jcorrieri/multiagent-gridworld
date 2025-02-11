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

def compute_frontier_scores(visited):
    size = visited.shape[0]
    frontier_scores = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(size):
            raw_score = 0
            neighbors = get_neighbors((i, j), size)

            for neighbor in neighbors:
                x, y = neighbor
                if not visited[x, y]:
                    raw_score += 1

            frontier_scores[i, j] = raw_score / len(neighbors)

    return frontier_scores

def parse_optimizer(parser):
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--model', type=str, default='PPO')
    parser.add_argument('--env_name', type=str, default='GridWorld-v0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--size', type=int, default=16)
    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--cr', type=int, default=6)
