import os.path

import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt


def parse_optimizer(parser):
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

def plot_metrics(metrics: [[float, float]], path: str):
    mean_rewards = [m[0] for m in metrics]
    mean_lengths = [m[1] for m in metrics]
    episode = [m[2] for m in metrics]

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(episode, mean_rewards, label="Mean Reward", color='blue')
    axs[0].set_ylabel("Mean Reward")
    axs[0].set_title("Training Progress")
    axs[0].grid(True)

    axs[1].plot(episode, mean_lengths, label="Mean Episode Length", color='green')
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Mean Episode Length")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(path, "metrics_plot.png"))
    print(f"Saved training plot to {path}/metrics_plot.png")

def generate_obstacles(grid_size=25, obstacle_density=0.15, max_attempts=100, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def flood_fill(grid, start):
        visited = np.zeros_like(grid, dtype=bool)
        queue = deque([start])
        visited[start] = True
        count = 1
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if not visited[nr, nc] and grid[nr, nc] == 0:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
                        count += 1
        return count

    def is_connected(grid):
        free_positions = np.argwhere(grid == 0)
        if len(free_positions) == 0:
            return False
        start = tuple(free_positions[0])
        filled = flood_fill(grid, start)
        return filled == len(free_positions)

    total_tiles = grid_size * grid_size
    max_obstacles = int(obstacle_density * total_tiles)

    for attempt in range(max_attempts):
        grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        obstacles_placed = 0
        while obstacles_placed < max_obstacles:
            shape_type = random.choice(["point", "rect", "line"])
            r, c = random.randint(0, grid_size - 2), random.randint(0, grid_size - 1)

            if shape_type == "point":
                if grid[r, c] == 0:
                    grid[r, c] = 1
                    obstacles_placed += 1

            elif shape_type == "rect":
                h, w = random.randint(1, 3), random.randint(1, 3)
                r2, c2 = min(r + h, grid_size), min(c + w, grid_size)
                subgrid = grid[r:r2, c:c2]
                free_space = (subgrid == 0)
                num_new = np.sum(free_space)
                if obstacles_placed + num_new <= max_obstacles:
                    subgrid[free_space] = 1
                    obstacles_placed += num_new

            elif shape_type == "line":
                length = random.randint(3, 6)
                if random.random() < 0.5:  # horizontal
                    c2 = min(c + length, grid_size)
                    line = grid[r, c:c2]
                    free_space = (line == 0)
                    num_new = np.sum(free_space)
                    if obstacles_placed + num_new <= max_obstacles:
                        line[free_space] = 1
                        obstacles_placed += num_new
                else:  # vertical
                    r2 = min(r + length, grid_size)
                    line = grid[r:r2, c]
                    free_space = (line == 0)
                    num_new = np.sum(free_space)
                    if obstacles_placed + num_new <= max_obstacles:
                        line[free_space] = 1
                        obstacles_placed += num_new

        if is_connected(grid):
            grid[grid_size - 3 :, :] = 0
            return grid

    raise RuntimeError("Failed to generate a valid obstacle map after multiple attempts.")

def save_obstacle_map(grid, filename):
    with open(filename, 'w') as f:
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 1:
                    f.write(f"{r} {c}\n")

def gen_train_test_split():
    for i in range(50):
        grid = generate_obstacles()
        save_obstacle_map(grid, f'env/obstacle_mats/training/mat{i}')
    for i in range(2, 50):
        grid = generate_obstacles()
        save_obstacle_map(grid, f'env/obstacle_mats/testing/mat{i}')
