import functools
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


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    no_op = 4


class GridWorldEnv(ParallelEnv):
    metadata = {
        "name": "GridWorldEnv_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 12
    }

    def __init__(self, render_mode=None, size=12, num_agents=3, cr=3, max_steps=1000, map_name=None, rw_scheme=None):
        self.size = size
        self.window_size = 512
        self.rng = None

        self._num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]

        self.cr = cr
        self.max_steps = max_steps
        self.max_coverage = 0
        self.timestep = 0

        # reward
        self._new_tile_connected: float = rw_scheme['new_tile_connected']
        self._new_tile_disconnected: float = rw_scheme['new_tile_disconnected']
        self._old_tile_connected: float = rw_scheme['old_tile_connected']
        self._old_tile_disconnected: float = rw_scheme['old_tile_disconnected']
        self._obs_penalty: float = rw_scheme['obstacle']
        self._termination_bonus: float = rw_scheme['terminated']

        self.grid = np.zeros((size, size), dtype=np.int8)
        self._adj_matrix = np.zeros((num_agents, num_agents), dtype=int)
        self._agent_locations = np.zeros((num_agents, 2), dtype=int)

        self._map_name = map_name
        self._obs_mat = None

        self._action_to_direction = {
            Actions.right.value: np.array([0, 1]),
            Actions.up.value: np.array([-1, 0]),
            Actions.left.value: np.array([0, -1]),
            Actions.down.value: np.array([1, 0]),
            Actions.no_op.value: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _build_adj_matrix(self):
        """Build adjacency matrix based on communication range"""
        self._adj_matrix = np.zeros((self._num_agents, self._num_agents), dtype=np.int64)
        for i in range(self._num_agents):
            for j in range(i + 1, self._num_agents):
                dist = np.linalg.norm(self._agent_locations[i] - self._agent_locations[j])
                if dist <= self.cr:
                    self._adj_matrix[i][j] = 1
                    self._adj_matrix[j][i] = 1

    def _generate_observation(self, agent_idx):
        obs = np.zeros((self.size, self.size, 4), dtype=np.float32)

        # Layer 0: Obstacle Map
        obs[:, :, 0] = (self.grid < 0).astype(np.float32)

        # Layer 1: Agent's own position
        agent_pos = self._agent_locations[agent_idx]
        obs[agent_pos[0], agent_pos[1], 1] = 1.0

        # Layer 2: Other agents' positions
        for i, pos in enumerate(self._agent_locations):
            if i != agent_idx:
                obs[pos[0], pos[1], 2] = 1.0

        # Layer 3: Binary coverage map (1 = visited, 0 = unvisited, -1 = obstacle (count as visited))
        obs[:, :, 3] = (self.grid != 0).astype(np.float32)

        return obs

    def _generate_spawns(self, occupied_positions: set):
        placed_agents = []
        for i in range(self._num_agents):
            while True:
                if len(placed_agents) == 0:  # place first agent
                    x, y = self.rng.integers(0, self.size, 2)
                else:
                    # place others within range of another agent
                    ref_agent = placed_agents[0]

                    delta_arr = np.array(self.rng.integers(-self.cr, self.cr + 1, 2))
                    new_agent = ref_agent + delta_arr

                    if np.linalg.norm(ref_agent - new_agent) >= self.cr:
                        continue

                    x, y = new_agent
                    x, y = min(max(x, 0), self.size - 1), min(max(y, 0), self.size - 1)

                if (x, y) not in occupied_positions:
                    self._agent_locations[i] = np.array([x, y])
                    occupied_positions.add((x, y))
                    placed_agents.append(np.array([x, y]))
                    break


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
        # 12x12x4 observation space with binary values
        return spaces.Box(low=0, high=1, shape=(self.size, self.size, 4), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """Return action space for a specific agent"""
        return spaces.Discrete(5)  # 5 actions: right, up, left, down, no-op

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)

        if self._obs_mat is None:
            obs_mat = np.loadtxt(f"env/obstacle_mats/{self._map_name}", delimiter=' ', dtype='int')
            self._obs_mat = [(x, y) for x, y in obs_mat]

        self.max_coverage = self.size**2 - len(self._obs_mat)

        for i, (row, col) in enumerate(self._obs_mat):
            self.grid[row, col] = -1

        self._generate_spawns(set(self._obs_mat))

        for pos in self._agent_locations:
            self.grid[pos[0], pos[1]] = 1

        self._build_adj_matrix()

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._generate_observation(i)
            infos[agent] = {"coverage": np.sum(self.grid > 0) / self.max_coverage}

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos

    def step(self, actions):
        """Execute one step for all agents"""
        self.timestep += 1

        previous_locations = self._agent_locations.copy()

        rewards = {agent: 0.0 for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}

        occupied_positions = set(self._obs_mat)
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

        self._agent_locations = np.array(new_positions)

        self._build_adj_matrix()
        G = nx.from_numpy_array(self._adj_matrix)
        connected = nx.is_connected(G)

        # Calculate rewards
        for i, agent in enumerate(self.agents):
            current_pos = self._agent_locations[i]
            previous_pos = previous_locations[i]

            # Check if the agent moved
            if not np.array_equal(current_pos, previous_pos):
                if self.grid[current_pos[0], current_pos[1]] == 0 and connected:
                    rewards[agent] += self._new_tile_connected
                elif self.grid[current_pos[0], current_pos[1]] == 0:
                    rewards[agent] += self._new_tile_disconnected
                elif connected:
                    rewards[agent] += self._old_tile_connected
                else:
                    rewards[agent] += self._old_tile_disconnected

                self.grid[current_pos[0], current_pos[1]] = 1
            else:
                # collision or no-op
                rewards[agent] += self._obs_penalty

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._generate_observation(i)
            infos[agent] = {
                "coverage": np.sum(self.grid > 0) / self.max_coverage,
                "step": self.timestep,
                "connection_broken": not connected
            }

        all_visited = np.sum(self.grid > 0) == self.max_coverage
        if all_visited:
            for agent in self.agents:
                rewards[agent] += self._termination_bonus
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
                if self.grid[i, j] > 0:
                    # Light blue for visited cells
                    pygame.draw.rect(
                        canvas,
                        (185, 235, 245),
                        pygame.Rect(
                            pix_square_size * np.array([j, i]), # flip coords for pygame rendering
                            (pix_square_size, pix_square_size),
                        ),
                    )

        for (x, y) in self._obs_mat:
            pygame.draw.rect(
                canvas,
                (0, 0, 0), # Black for obstacles
                pygame.Rect(
                    pix_square_size * np.array([y, x]),  # flip coords for pygame rendering
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
        G = nx.from_numpy_array(self._adj_matrix)
        for a, b in G.edges():
            (ax, ay), (bx, by) = self._agent_locations[a], self._agent_locations[b]
            pygame.draw.line(
                canvas,
                (255, 0, 0),  # Red for communication links
                ((ay + 0.5) * pix_square_size, (ax + 0.5) * pix_square_size),
                ((by + 0.5) * pix_square_size, (bx + 0.5) * pix_square_size),
                width=2,
            )

        # Draw agents
        for i, (x, y) in enumerate(self._agent_locations):
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
        coverage = np.sum(self.grid > 0) / self.max_coverage * 100
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
    env = GridWorldEnv()
    parallel_api_test(env, num_cycles=1_000_000)
    env.close()