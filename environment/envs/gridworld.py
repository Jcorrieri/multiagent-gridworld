import functools
import os
from copy import copy
from enum import Enum

import gymnasium
import networkx as nx
import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID

import environment.rewards
from environment.rewards import RewardScheme, Default


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    no_op = 4


class GridWorldEnv(ParallelEnv):
    metadata = {
        "name": "gridworld",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 24
    }

    def __init__(self, env_params, **kwargs):
        self.size = env_params.get("size", 25)
        self.window_size = 512
        self.rng = np.random.default_rng(env_params.get("seed", None))

        self._num_agents = env_params.get("num_agents", 5)
        self.base_station = env_params.get("base_station", False)
        self.total_num_robots = (self._num_agents + 1) if self.base_station else self._num_agents
        self.agent_locations: dict[str, tuple[int, int]] = {}
        self.agents = [f"agent_{i}" for i in range(self._num_agents)]  # never contains the base station

        self.cr = env_params.get("cr", 10)
        self.fov_range = env_params.get("fov", 25)
        self.use_local_fov = self.fov_range < 25
        self.max_steps = env_params.get("max_steps", 1000)
        self.max_coverage = 0
        self.timestep = 0

        self.reward_scheme: RewardScheme = env_params.get("reward_scheme", Default())

        self.visited_tiles = np.zeros((self.size, self.size), dtype=int)
        self.visible_tiles = np.zeros((self.size, self.size), dtype=int)
        self.adj_matrix = np.zeros((self.total_num_robots, self.total_num_robots), dtype=int)
        self.obs_mat = np.zeros((self.size, self.size), dtype=int)

        self.num_maps = 50
        self.map_dir_path = env_params.get("map_dir_path")
        self.map_indices = self.rng.permutation(np.arange(self.num_maps)).tolist()

        self._action_to_direction = {
            Actions.right.value: np.array([0, 1]),
            Actions.up.value: np.array([-1, 0]),
            Actions.left.value: np.array([0, -1]),
            Actions.down.value: np.array([1, 0]),
            Actions.no_op.value: np.array([0, 0]),
        }

        self.render_mode = env_params.get("render_mode", "rgb_array")

        self.window = None
        self.clock = None

    # For Ray RLlib
    @property
    def observation_spaces(self):
        return {agent: self.observation_space(agent) for agent in self.agents}

    # For Ray RLlib
    @property
    def action_spaces(self):
        return {agent: self.action_space(agent) for agent in self.agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """Return observation space for a specific agent"""
        # 12x12xC observation space with binary values
        return spaces.Box(low=0, high=1, shape=(self.size, self.size, 4), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """Return action space for a specific agent"""
        return spaces.Discrete(5)  # 5 actions: right, up, left, down, no-op

    def _build_adj_matrix(self, locations: list[np.ndarray]):
        """Build adjacency matrix based on communication range"""
        self.adj_matrix = np.zeros((self.total_num_robots, self.total_num_robots), dtype=np.int64)
        for i in range(self.total_num_robots):
            for j in range(i + 1, self.total_num_robots):
                dist = np.linalg.norm(locations[i] - locations[j])
                if dist <= self.cr:
                    self.adj_matrix[i][j] = 1
                    self.adj_matrix[j][i] = 1

    def generate_observation(self, agent):
        channels = 4

        obs = np.zeros((self.size, self.size, channels), dtype=np.float32)

        # Layer 0: Obstacle Map
        if self.use_local_fov:
            visible_obs_mask = (self.obs_mat == 1) & (self.visible_tiles == 1)
            obs[visible_obs_mask, 0] = 1.0
        else:
            obs[self.obs_mat, 0] = 1.0

        # Layer 1: Agent's own position
        (row, col) = self.agent_locations[agent]
        obs[row, col, 1] = 1.0

        # Layer 2: Other agents' positions including base station if enabled
        for agent_key, (row, col) in filter(lambda kv : kv[0] != agent, self.agent_locations.items()):
            obs[row, col, 2] = 1.0

        # Layer 3: Coverage Map
        mask = self.visited_tiles == 1
        obs[mask, 3] = 1.0

        return obs

    def update_visibility(self):
        for agent in self.agents:
            center_row, center_col = self.agent_locations[agent]

            row_start = max(0, center_row - self.fov_range)
            row_end = min(self.size, center_row + self.fov_range + 1)  # +1 to account for exclusive Python slicing
            col_start = max(0, center_col - self.fov_range)
            col_end = min(self.size, center_col + self.fov_range + 1)

            rr, cc = np.meshgrid(range(row_start, row_end), range(col_start, col_end), indexing='ij')
            locations = np.stack((rr, cc), axis=-1).reshape(-1, 2)  # all visible tiles

            self.visible_tiles[locations[:, 0], locations[:, 1]] = 1

            unvisited_obstacle_mask = (self.visible_tiles == 1) & (self.obs_mat == 1)

            self.visited_tiles[unvisited_obstacle_mask] = 1.0

    def locations_to_ndarray_list(self) -> list[np.ndarray[int, int]]:
        locations_to_ndarray_list = []

        for i in range(self._num_agents):
            location = np.array(self.agent_locations[f"agent_{i}"])
            locations_to_ndarray_list.append(location)
        if self.base_station:
            location = self.agent_locations["base_station"]
            locations_to_ndarray_list.append(location)

        return locations_to_ndarray_list

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.visited_tiles = np.zeros((self.size, self.size), dtype=int)
        self.visible_tiles = np.zeros((self.size, self.size), dtype=int)
        self.obs_mat = np.zeros((self.size, self.size), dtype=int)
        self.timestep = 0

        if not self.map_indices:
            self.map_indices = self.rng.permutation(np.arange(self.num_maps)).tolist()

        mat_idx = self.map_indices.pop()
        map_path = os.path.join(self.map_dir_path, f'mat{mat_idx}')
        obstacle_points = np.loadtxt(map_path, delimiter=' ', dtype='int')
        self.obs_mat[obstacle_points[:, 0], obstacle_points[:, 1]] = 1

        self.max_coverage = self.size**2

        # spawn agents
        base_station_offset = 0
        if self.base_station:
            base_station_offset = 1
            self.agent_locations["base_station"] = (self.size - 1, 0)

        for i, agent in enumerate(self.agents):
            self.agent_locations[agent] = (self.size - 1, i + base_station_offset)

        self.visited_tiles[self.size - 1, :self.total_num_robots] = 1

        self._build_adj_matrix(self.locations_to_ndarray_list())

        if self.use_local_fov:
            self.update_visibility()
        else:
            self.visited_tiles[self.obs_mat == 1] = 1.0

        observations = {}
        infos = {}
        for agent in self.agents:
            observations[agent] = self.generate_observation(agent)
            infos[agent] = {
                "coverage": np.sum(self.visited_tiles > 0) / self.max_coverage * 100,
                "step": self.timestep,
                "connection_broken": False,
            }

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos

    def step(self, actions):
        """Execute one step for all agents"""
        self.timestep += 1

        rewards = {agent: 0.0 for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}

        obstacle_coords = np.argwhere(self.obs_mat == 1)
        occupied_positions = set(map(tuple, obstacle_coords))
        for position in self.agent_locations.values():
            occupied_positions.add(position)

        visited_count = np.sum(self.visited_tiles > 0)  # needed for proper coverage calculation

        # Determine new positions
        collisions = {agent: False for agent in self.agents}
        for agent in self.agents:
            action = actions[agent]
            direction = self._action_to_direction[action]
            (curr_x, curr_y) = self.agent_locations[agent]

            proposed_position = (curr_x + direction[0], curr_y + direction[1])

            row, col = proposed_position
            out_of_bounds = row < 0 or row >= self.size or col < 0 or col >= self.size
            valid_move = action == Actions.no_op.value or (not out_of_bounds and proposed_position not in occupied_positions)

            if valid_move:
                if self.visited_tiles[row, col] == 0:
                    visited_count += 1

                old_position = self.agent_locations[agent]
                occupied_positions.remove(old_position)

                self.agent_locations[agent] = proposed_position
                occupied_positions.add(proposed_position)
            else:
                # If collision, don't update positions
                collisions[agent] = True

        self._build_adj_matrix(self.locations_to_ndarray_list())
        G = nx.from_numpy_array(self.adj_matrix)
        connected = nx.is_connected(G)

        coverage = visited_count / self.max_coverage
        prev_coverage = np.sum(self.visited_tiles > 0) / self.max_coverage

        step_info = {
            "coverage": coverage * 100,
            "prev_coverage": prev_coverage * 100,
            "timestep": self.timestep,
            "connected": connected,
            "collisions": collisions,
            "graph": G,
        }

        # Calculate rewards
        self.reward_scheme.calculate_rewards(rewards, step_info, self)

        for (row, col) in self.agent_locations.values():
            self.visited_tiles[row, col] = 1

        if self.use_local_fov:
            self.update_visibility()

        observations = {}
        infos = {}
        for agent in self.agents:
            observations[agent] = self.generate_observation(agent)
            infos[agent] = {
                "coverage": step_info['coverage'],
                "step": step_info['timestep'],
                "connection_broken": not step_info['connected'],
            }

        if visited_count == self.max_coverage:
            for agent in self.agents:
                rewards[agent] += self.reward_scheme.get_terminated()
            terminated = {agent: True for agent in self.agents}

        if self.timestep >= self.max_steps:
            truncated = {agent: True for agent in self.agents}

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminated, truncated, infos

    def _get_agent_color(self, agent_idx):
        """Get unique color for each agent"""
        colors = [
            (0, 0, 0),  # Black
            (255, 0, 0),  # Red
            (0, 0, 255),  # Blue
            (0, 255, 0),  # Green
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),  # Maroon
            (0, 128, 0),  # Dark Green
            (0, 0, 128),  # Navy
        ]
        return colors[agent_idx % len(colors)]

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

    def close(self):
        """Close the environment"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    reward_scheme = environment.rewards.Coverage()

    env = GridWorldEnv({
        'render_mode': "human",
        'map_dir_path': '../obstacle-mats/testing',
        'base_station': False,
        'fov': 25,
        'num_agents': 5,
        'reward_scheme': reward_scheme
    })

    # unit test -- default env
    obs, _ = env.reset()
    episode_over = False
    r = 0.0
    while not episode_over:
        vals = np.random.default_rng().integers(low=0, high=5, size=env.num_agents)
        actions_dict = {f'agent_{i}': int(val) for i, val in enumerate(vals)}
        observations, rewards, terminated, truncated, infos = env.step(actions_dict)
        r += sum(rewards.values())
        print("\rStep reward:", round(sum(rewards.values()), 2), "Total reward:", round(r, 2), end="")
        episode_over = all(terminated.values()) or all(truncated.values())

    # parallel_api_test(env, num_cycles=1_000_000)
    env.close()