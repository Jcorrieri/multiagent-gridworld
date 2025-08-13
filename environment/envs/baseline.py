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
        self.meeting_point = None
        self.recovery_cost_map = None
        self.in_deadlock_recovery = False
        self.frontiers = []
        self.s = kwargs.get("S", 20)
        self.epsilon = kwargs.get("epsilon", 2)
        self.hist_points = 0
        self.minx = np.zeros(env_params['num_agents'], dtype=int)
        self.maxx = np.zeros(env_params['num_agents'], dtype=int)
        self.miny = np.zeros(env_params['num_agents'], dtype=int)
        self.maxy = np.zeros(env_params['num_agents'], dtype=int)

    def deadlock_recovery(self):
        self.in_deadlock_recovery = True
        
        meet_radius = self.num_agents * 1.0
        num_neighbors = 4
        within_meet = {agent: False for agent in self.agents}

        moves = []

        for agent in self.agents:
            location = np.array(self.agent_locations[agent])
            if np.linalg.norm(location - self.meeting_point) < meet_radius:
                within_meet[agent] = True

            best_move = Actions.no_op.value
            min_cost = 999999999
            for i in range(num_neighbors):
                direction = self._action_to_direction[i]
                (x, y) = location + direction

                out_of_bounds_r = x < 0 or x >= self.size
                out_of_bounds_c = y < 0 or y >= self.size
                out_of_bounds = out_of_bounds_r or out_of_bounds_c

                if out_of_bounds:
                    continue

                cost = self.recovery_cost_map[x, y]
                if cost < min_cost:
                    min_cost = cost
                    best_move = i
            moves.append(best_move)

        if all(within_meet.values()):
            self.in_deadlock_recovery = False

        return moves

    def detect_deadlock(self) -> bool:
        frontier_set = set(map(tuple, self.frontiers))
        # check for every robot
        for i, agent in enumerate(self.agents):
            position = self.agent_locations[agent]
            # if a robot hits a frontier then reset
            if position in frontier_set:
                self.minx[i] = self.maxx[i] = position[0]
                self.miny[i] = self.maxy[i] = position[1]
                self.hist_points = 0
                return False
            else: # update the changes in x and y
                if position[0] < self.minx[i]:
                    self.minx[i] = position[0]
                if position[1] < self.miny[i]:
                    self.miny[i] = position[1]
                if position[0] > self.maxx[i]:
                    self.maxx[i] = position[0]
                if position[1] > self.maxy[i]:
                    self.maxy[i] = position[1]

                # check whether last frontier visit is older than S
                if self.hist_points > self.s:
                    # if no progress is made
                    if (self.maxx[i] - self.minx[i]) < self.epsilon and (self.maxy[i] - self.miny[i]) < self.epsilon:
                        return True # deadlock
                    else:
                        # otherwise reset
                        self.minx[i] = self.maxx[i] = position[0]
                        self.miny[i] = self.maxy[i] = position[1]
                        self.hist_points = 0
        self.hist_points += 1 # synchronous movement means agents move all at once each step
        return False

    def get_frontiers(self):
        # Define 4-connectivity kernel (Von Neumann neighborhood)
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        obstacle_mask = self.obs_mat == 1

        visited = (self.visited_tiles == 1) & (~obstacle_mask)

        visited_neighbor_count = convolve(visited.astype(np.uint8), kernel, mode='constant', cval=0)

        frontier_mask = (self.visited_tiles == 0) & (~obstacle_mask) & (visited_neighbor_count > 0)

        self.frontiers = np.argwhere(frontier_mask)

    def wavefront_distance_from_frontier(self, points: list[ndarray]):
        h, w = self.size, self.size
        dist_map = np.full((h, w), fill_value=np.inf, dtype=np.float32)

        q = deque()
        for (x, y) in points:
            dist_map[x, y] = 0
            q.append((x, y))

        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        while q:
            x, y = q.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    if self.obs_mat[nx, ny] == 0 and dist_map[nx, ny] > dist_map[x, y] + 1:
                        dist_map[nx, ny] = dist_map[x, y] + 1
                        q.append((nx, ny))

        return dist_map

    def execute_config(self, movements) -> list:
        new_positions = []
        for i, agent in enumerate(self.agents):
            action = movements[i]
            direction = self._action_to_direction[action]
            (curr_x, curr_y) = self.agent_locations[agent]
            proposed_position = (curr_x + direction[0], curr_y + direction[1])

            new_positions.append(proposed_position)

        return new_positions

    def compute_fitness(self, config: ndarray, dist_map: ndarray[tuple[int, int]]) -> float:
        config_fitness = 0.0

        new_positions = self.execute_config(config)

        arrays_list = []

        for t in new_positions:
            arrays_list.append(t)

        if self.base_station:
            arrays_list = [(self.size - 1, 0)]

        np_arraylist = list(map(np.array, arrays_list))
        self._build_adj_matrix(np_arraylist)
        G = nx.from_numpy_array(self.adj_matrix)
        connected = nx.is_connected(G)

        pos_counts = Counter(new_positions)
        colliding_positions = {pos for pos, count in pos_counts.items() if count > 1}

        for position in new_positions:
            utility = 0.0

            out_of_bounds_r = position[0] < 0 or position[0] >= self.size
            out_of_bounds_c = position[1] < 0 or position[1] >= self.size
            out_of_bounds = out_of_bounds_r or out_of_bounds_c

            in_obstacle = True
            if not out_of_bounds:
                in_obstacle = self.obs_mat[position[0], position[1]] == 1

            agent_collision = position in colliding_positions

            invalid_move = out_of_bounds or in_obstacle or agent_collision or not connected

            if invalid_move:
                utility += -999999.0
            else:
                utility -= dist_map[position[0], position[1]]

            config_fitness += utility

        return config_fitness

    def get_max_config(self) -> ndarray:
        k = 50
        self.get_frontiers()
        dist_map = self.wavefront_distance_from_frontier(self.frontiers)

        # generate a population
        configurations = [np.full(self.num_agents, Actions.no_op.value)]
        for i in range(1, k):
            # generate a config change
            new_moves = np.random.default_rng().integers(low=0, high=5, size=self.num_agents)
            configurations.append(new_moves)

        # compute fitness; find maximum
        config_max = configurations[0]
        max_fitness = self.compute_fitness(config_max, dist_map)
        for i in range(1, k):
            config_fitness = self.compute_fitness(configurations[i], dist_map)
            if config_fitness > max_fitness:
                config_max = configurations[i]
                max_fitness = config_fitness

        return config_max

    def execute_algorithm(self) -> dict[str, int]:
        if self.in_deadlock_recovery:
            print("DEADLOCK!!!!")
            new_config = self.deadlock_recovery()
        elif self.detect_deadlock():
            rand = np.random.default_rng().integers(low=0, high=self.num_agents)
            self.meeting_point = np.array(self.agent_locations[f"agent_{rand}"])
            self.recovery_cost_map = self.wavefront_distance_from_frontier([self.meeting_point])

            new_config = self.deadlock_recovery()
        else:
            new_config = self.get_max_config()

        return {f'agent_{i}': int(val) for i, val in enumerate(new_config)}

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed, options)
        self.hist_points = 0
        self.in_deadlock_recovery = False
        self.minx = np.zeros(self.num_agents, dtype=int)
        self.maxx = np.zeros(self.num_agents, dtype=int)
        self.miny = np.zeros(self.num_agents, dtype=int)
        self.maxy = np.zeros(self.num_agents, dtype=int)
        self.frontiers = []
        return observations, infos

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

        for obstacle in np.argwhere(self.obs_mat == 1):
            pygame.draw.rect(
                canvas,
                (0, 0, 0), # Black for obstacles
                pygame.Rect(
                    pix_square_size * np.flip(obstacle),  # flip coords for pygame rendering
                    (pix_square_size, pix_square_size),
                    ),
            )

        if self.use_local_fov: # fog of war
            indices = np.argwhere(self.visible_tiles == 0)
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
            a_loc_key = f"agent_{a}"
            b_loc_key = f"agent_{b}"
            if a == self._num_agents:
                a_loc_key = "base_station"
            elif b == self._num_agents:
                b_loc_key = "base_station"

            (ax, ay), (bx, by) = self.agent_locations[a_loc_key], self.agent_locations[b_loc_key]
            pygame.draw.line(
                canvas,
                (255, 0, 0),  # Red for communication links
                ((ay + 0.5) * pix_square_size, (ax + 0.5) * pix_square_size),
                ((by + 0.5) * pix_square_size, (bx + 0.5) * pix_square_size),
                width=2,
            )

        # Draw agents and base station if enabled
        for i, (agent, (x, y)) in enumerate(self.agent_locations.items()):
            if agent == "base_station":
                color = (85, 85, 85)
                text = "B"
            else:
                color = self._get_agent_color(i)
                text = str(i)

            # Draw agent circle
            pygame.draw.circle(
                canvas,
                color,
                ((y + 0.5) * pix_square_size, (x + 0.5) * pix_square_size), # swap coords for pygame rendering
                pix_square_size / 3,
                )

            # Draw agent ID
            font = pygame.font.SysFont(None, int(pix_square_size / 2))
            text_content = font.render(text, True, (255, 255, 255))
            text_rect = text_content.get_rect(center=((y + 0.5) * pix_square_size, (x + 0.5) * pix_square_size))
            canvas.blit(text_content, text_rect)

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
        actions_dict = env.execute_algorithm()
        observations, rewards, terminated, truncated, infos = env.step(actions_dict)
        r += sum(rewards.values())
        print("\rStep reward:", round(sum(rewards.values()), 2), "Total reward:", round(r, 2), end="")
        episode_over = all(terminated.values()) or all(truncated.values())

    # parallel_api_test(env, num_cycles=1_000_000)
    env.close()