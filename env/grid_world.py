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
from pettingzoo.test import parallel_api_test
from pettingzoo.utils.env import AgentID

import utils


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
        "render_fps": 60
    }

    def __init__(self, env_params, **kwargs):       
        self.size = env_params.get("size", 25)
        self.window_size = 512
        self.rng = np.random.default_rng(env_params.get("seed", 42))

        self._num_agents = env_params.get("num_agents", 5)
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents = []

        self.cr = env_params.get("cr", 10)
        self.fov_range = env_params.get("fov", 25)
        self.max_steps = env_params.get("max_steps", 1000)
        self.max_coverage = 0
        self.timestep = 0

        # reward
        reward_scheme = env_params.get("reward_scheme", {})
        self.new_tile_visited = reward_scheme.get("new_tile_visited", 2.0)
        self.old_tile_visited = reward_scheme.get("old_tile_visited", -0.1)
        self.obstacle_penalty = reward_scheme.get("obstacle", -1.0)
        self.termination_bonus = reward_scheme.get("terminated", 480)

        self.visited_tiles = np.zeros((self.size, self.size), dtype=int)
        self.visibility_mask = np.zeros((self.size, self.size), dtype=int)
        self.adj_matrix = np.zeros((self._num_agents, self._num_agents), dtype=int)
        self.agent_locations = np.zeros((self._num_agents, 2), dtype=int)

        self.num_maps = 50
        self.map_dir_path = env_params.get("map_dir_path")
        self.map_indices = self.rng.permutation(np.arange(self.num_maps)).tolist()
        self.obs_mat = None

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

    def _build_adj_matrix(self):
        """Build adjacency matrix based on communication range"""
        self.adj_matrix = np.zeros((self._num_agents, self._num_agents), dtype=np.int64)
        for i in range(self._num_agents):
            for j in range(i + 1, self._num_agents):
                dist = np.linalg.norm(self.agent_locations[i] - self.agent_locations[j])
                if dist <= self.cr:
                    self.adj_matrix[i][j] = 1
                    self.adj_matrix[j][i] = 1

    def _generate_observation(self, agent_idx):
        channels = 5 if self.fov_range < self.size else 4  # add visibility mask if using local fovs

        obs = np.zeros((self.size, self.size, channels), dtype=np.float32)

        # Layer 0: Obstacle Map
        obs[self.obs_mat[:, 0], self.obs_mat[:, 1], 0] = 1.0

        # Layer 1: Agent's own position
        agent_pos = self.agent_locations[agent_idx]
        obs[agent_pos[0], agent_pos[1], 1] = 1.0

        # Layer 2: Other agents' positions
        for i, pos in enumerate(self.agent_locations):
            if i != agent_idx:
                obs[pos[0], pos[1], 2] = 1.0

        # Layer 3: Coverage Map
        mask = self.visited_tiles == 1
        obs[mask, 3] = 1.0

        # Layer 4 (Optional): Visibility Mask
        if self.fov_range < self.size:
            mask = self.visibility_mask == 1
            obs[mask, 4] = 1.0

        return obs

    def _generate_spawns(self):
        self.agent_locations[:, 0] = self.size - 1
        self.agent_locations[:, 1] = np.arange(self._num_agents)
        self.visited_tiles[self.size - 1, :self._num_agents] = 1

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.visited_tiles = np.zeros((self.size, self.size), dtype=int)
        self.visibility_mask = np.zeros((self.size, self.size), dtype=int)
        self.agent_locations = np.zeros((self._num_agents, 2), dtype=int)
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        if not self.map_indices:
            self.map_indices = self.rng.permutation(np.arange(self.num_maps)).tolist()

        mat_idx = self.map_indices.pop()
        map_path = os.path.join(self.map_dir_path, f'mat{mat_idx}')  # TODO -- replace w real map indices
        self.obs_mat = np.loadtxt(map_path, delimiter=' ', dtype='int')

        self.visited_tiles[self.obs_mat[:, 0], self.obs_mat[:, 1]] = 1.0  # count obstacle tiles as visited

        self.max_coverage = self.size**2 - len(self.obs_mat)

        self._generate_spawns()
        self._build_adj_matrix()

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._generate_observation(i)
            infos[agent] = {
                "coverage": np.sum(self.visited_tiles > 0) / self.max_coverage,
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

        previous_locations = self.agent_locations.copy()

        occupied_positions = set([(x, y) for x, y in self.obs_mat])
        for pos in previous_locations:
            occupied_positions.add(tuple(pos))

        # Determine new positions
        new_positions = []
        for i, agent in enumerate(self.agents):
            action = actions[agent]
            direction = self._action_to_direction[action]
            proposed_position = np.clip(previous_locations[i] + direction, 0, self.size - 1)

            # Check if proposed position is already claimed
            pos_tuple = tuple(proposed_position)
            if pos_tuple not in occupied_positions:
                new_positions.append(proposed_position)
                occupied_positions.add(pos_tuple)
            else:
                # If collision, don't move
                new_positions.append(previous_locations[i])
                occupied_positions.add(tuple(previous_locations[i]))

        self.agent_locations = np.array(new_positions)

        self._build_adj_matrix()
        G = nx.from_numpy_array(self.adj_matrix)
        connected = nx.is_connected(G)

        # Calculate rewards
        for i, agent in enumerate(self.agents):
            current_pos = self.agent_locations[i]
            previous_pos = previous_locations[i]

            # Check if the agent moved
            if not np.array_equal(current_pos, previous_pos):
                if self.visited_tiles[current_pos[0], current_pos[1]] == 0:
                    rewards[agent] += self.new_tile_visited
                else:
                    rewards[agent] += self.old_tile_visited

                self.visited_tiles[current_pos[0], current_pos[1]] = 1
            else:
                # collision or no-op
                rewards[agent] += self.obstacle_penalty

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._generate_observation(i)
            infos[agent] = {
                "coverage": np.sum(self.visited_tiles > 0) / self.max_coverage,
                "step": self.timestep,
                "connection_broken": not connected,
            }

        all_visited = np.sum(self.visited_tiles > 0) == self.max_coverage
        if all_visited:
            for agent in self.agents:
                rewards[agent] += self.termination_bonus
            terminated = {agent: True for agent in self.agents}
            self.agents = []

        if self.timestep >= self.max_steps:
            truncated = {agent: True for agent in self.agents}
            self.agents = []

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
        for i in range(self.size):
            for j in range(self.size):
                if self.visited_tiles[i, j] > 0:
                    # Light blue for visited cells
                    pygame.draw.rect(
                        canvas,
                        (185, 235, 245),
                        pygame.Rect(
                            pix_square_size * np.array([j, i]), # flip coords for pygame rendering
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

        # Draw agents
        for i, (x, y) in enumerate(self.agent_locations):
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
        coverage = len(self.visited_tiles) / self.max_coverage * 100
        font = pygame.font.SysFont(None, 30)
        text = font.render(f"Coverage: {coverage:.1f}% | Step: {self.timestep}", True, (0, 0, 0))
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
    env = GridWorldEnv({'render_mode': "human", 'map_dir_path': './obstacle-mats/testing'})

    for i in range(100):
        obs, _ = env.reset()
        episode_over = False
        while not episode_over:
            vals = np.random.default_rng().integers(low=0, high=5, size=5)
            actions_dict = {f'agent_{i}': int(val) for i, val in enumerate(vals)}
            observations, rewards, terminated, truncated, infos = env.step(actions_dict)
            episode_over = all(terminated.values()) or all(truncated.values())

    # parallel_api_test(env, num_cycles=1_000_000)
    env.close()