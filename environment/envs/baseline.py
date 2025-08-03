import os
from collections import deque, Counter
from copy import copy

import networkx as nx
import numpy as np
import pygame
from scipy.ndimage import convolve
from numpy import ndarray

import environment.rewards
from environment.envs.gridworld import GridWorldEnv, Actions


class BaselineEnv(GridWorldEnv):
    metadata = {
        "name": "baseline",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 24
    }

    def __init__(self, env_params, **kwargs):
        super().__init__(env_params, **kwargs)
        self.frontiers = []

    def get_frontiers(self):
        # Define 4-connectivity kernel (Von Neumann neighborhood)
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        obstacle_mask = np.zeros_like(self.visited_tiles, dtype=bool)
        obstacle_mask[self.obs_mat[:, 0], self.obs_mat[:, 1]] = True

        visited = (self.visited_tiles == 1) & (~obstacle_mask)

        visited_neighbor_count = convolve(visited.astype(np.uint8), kernel, mode='constant', cval=0)

        frontier_mask = (self.visited_tiles == 0) & (~obstacle_mask) & (visited_neighbor_count > 0)

        self.frontiers = np.argwhere(frontier_mask)

    def wavefront_distance_from_frontier(self, obstacles: list[tuple]):
        h, w = self.size, self.size
        dist_map = np.full((h, w), fill_value=np.inf, dtype=np.float32)

        q = deque()
        for x, y in self.frontiers:
            dist_map[x, y] = 0
            q.append((x, y))

        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        while q:
            x, y = q.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    if (nx, ny) not in obstacles and dist_map[nx, ny] > dist_map[x, y] + 1:
                        dist_map[nx, ny] = dist_map[x, y] + 1
                        q.append((nx, ny))

        return dist_map

    def execute_config(self, movements) -> list:
        new_positions = []
        for i, agent in enumerate(self.agents):
            action = movements[i]
            direction = self._action_to_direction[action]
            current_position = self.agent_locations[i]
            proposed_position = tuple(current_position + direction)
            new_positions.append(proposed_position)

        return new_positions

    def compute_fitness(self, config: ndarray, obstacles: list[tuple], dist_map: ndarray[tuple[int, int]]) -> float:
        config_fitness = 0.0

        new_positions = self.execute_config(config)

        if self.base_station:
            arrays_list = [np.array((24, 0))]
        else:
            arrays_list = []

        for t in new_positions:
            arrays_list.append(np.array(t))

        self._build_adj_matrix(arrays_list)
        G = nx.from_numpy_array(self.adj_matrix)
        connected = nx.is_connected(G)

        pos_counts = Counter(new_positions)
        colliding_positions = {pos for pos, count in pos_counts.items() if count > 1}

        for position in new_positions:
            utility = 0.0

            out_of_bounds_r = position[0] < 0 or position[0] >= self.size
            out_of_bounds_c = position[1] < 0 or position[1] >= self.size
            in_obstacle = position in obstacles

            agent_collision = position in colliding_positions

            invalid_move = out_of_bounds_r or out_of_bounds_c or in_obstacle or agent_collision

            if invalid_move or not connected:
                utility += -999999.0
            else:
                utility -= dist_map[position[0], position[1]]

            config_fitness += utility

        return config_fitness

    def get_max_config(self) -> ndarray:
        k = 100
        obstacles = [(x, y) for x, y in self.obs_mat]
        self.get_frontiers()
        dist_map = self.wavefront_distance_from_frontier(obstacles)

        # generate a population
        configurations = [np.full(self.num_agents, Actions.no_op.value)]
        for i in range(1, k):
            # generate a config change
            new_moves = np.random.default_rng().integers(low=0, high=5, size=self.num_agents)
            configurations.append(new_moves)

        # compute fitness; find maximum
        config_max = configurations[0]
        max_fitness = self.compute_fitness(config_max, obstacles, dist_map)
        for i in range(1, k):
            config_fitness = self.compute_fitness(configurations[i], obstacles, dist_map)
            if config_fitness > max_fitness:
                config_max = configurations[i]
                max_fitness = config_fitness

        return config_max

    def _render_frame(self):
        """Render the current state of the environment"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Multi-Agent Area Coverage")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background

        # Calculate grid cell size
        pix_square_size = self.window_size / self.size

        # Draw visited cells
        visited_indices = np.argwhere(self.visited_tiles > 0)
        for idx in visited_indices:
            pygame.draw.rect(
                canvas,
                (185, 235, 245),
                pygame.Rect(
                    pix_square_size * np.flip(idx), # flip coords for pygame rendering
                    (pix_square_size, pix_square_size),
                    ),
            )

        # draw frontier tiles
        for tile in self.frontiers:
            pygame.draw.rect(
                canvas,
                (240, 240, 0), # Black for obstacles
                pygame.Rect(
                    pix_square_size * np.flip(tile),  # flip coords for pygame rendering
                    (pix_square_size, pix_square_size),
                    ),
            )

        for obstacle in self.obs_mat:
            pygame.draw.rect(
                canvas,
                (0, 0, 0), # Black for obstacles
                pygame.Rect(
                    pix_square_size * np.flip(obstacle),  # flip coords for pygame rendering
                    (pix_square_size, pix_square_size),
                    ),
            )

        if self.use_local_fov: # fog of war
            indices = np.argwhere(self.visibility_mask == 0)
            for idx in indices:
                pygame.draw.rect(
                    canvas,
                    (192, 192, 192, 0.65),
                    pygame.Rect(
                        pix_square_size * np.flip(idx), # flip coords for pygame rendering
                        (pix_square_size, pix_square_size),
                        ),
                )

        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (192, 192, 192),  # Light gray
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (192, 192, 192),  # Light gray
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        # Draw communication links between agents
        G = nx.from_numpy_array(self.adj_matrix)
        for a, b in G.edges():
            (ax, ay), (bx, by) = self.agent_locations[a], self.agent_locations[b]
            pygame.draw.line(
                canvas,
                (255, 0, 0),  # Red for communication links
                ((ay + 0.5) * pix_square_size, (ax + 0.5) * pix_square_size),
                ((by + 0.5) * pix_square_size, (bx + 0.5) * pix_square_size),
                width=2,
            )

        # Draw base station if it exists
        if self.base_station:
            station_offset = 1

            top_left_x = (0 + 0.55) * pix_square_size - pix_square_size / 3
            top_left_y = ((self.size - 1) + 0.55) * pix_square_size - pix_square_size / 3

            # Draw station
            pygame.draw.rect(
                canvas,
                (105, 105, 105),
                pygame.Rect(top_left_x, top_left_y, pix_square_size / 1.5, pix_square_size / 1.5)
            )

            # Draw agent ID (centered)
            font = pygame.font.SysFont(None, int(pix_square_size * 0.7))
            text = font.render("B", True, (0, 0, 0))
            text_rect = text.get_rect(center=((0 + 0.5) * pix_square_size, ((self.size - 1) + 0.5) * pix_square_size))
            canvas.blit(text, text_rect)
        else:
            station_offset = 0

        # Draw agents
        for i, (x, y) in enumerate(self.agent_locations[:self._num_agents - station_offset]):
            # Draw agent circle
            pygame.draw.circle(
                canvas,
                self._get_agent_color(i),
                ((y + 0.5) * pix_square_size, (x + 0.5) * pix_square_size), # swap coords for pygame rendering
                pix_square_size / 3,
                )

            # Draw agent ID
            font = pygame.font.SysFont(None, int(pix_square_size / 2))
            text = font.render(str(i), True, (255, 255, 255))
            text_rect = text.get_rect(center=((y + 0.5) * pix_square_size, (x + 0.5) * pix_square_size))
            canvas.blit(text, text_rect)

        # Display coverage percentage
        coverage = np.sum(self.visited_tiles > 0) / self.max_coverage * 100
        font = pygame.font.SysFont(None, 30)
        text = font.render(f"Coverage: {coverage:.1f}% | Step: {self.timestep}", True, (0, 150, 0))
        canvas.blit(text, (10, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

if __name__ == "__main__":
    reward_scheme = environment.rewards.Default()

    env = BaselineEnv({
        'render_mode': "human",
        'map_dir_path': '../obstacle-mats/testing',
        'base_station': False,
        'fov': 25,
        'num_agents': 6,
        'reward_scheme': reward_scheme
    })

    # unit test -- default env
    obs, _ = env.reset()
    episode_over = False
    r = 0.0
    while not episode_over:
        actions_dict = {f'agent_{i}': int(val) for i, val in enumerate(env.get_max_config())}
        observations, rewards, terminated, truncated, infos = env.step(actions_dict)
        r += sum(rewards.values())
        print("\rStep reward:", round(sum(rewards.values()), 2), "Total reward:", round(r, 2), end="")
        episode_over = all(terminated.values()) or all(truncated.values())

    # parallel_api_test(env, num_cycles=1_000_000)
    env.close()