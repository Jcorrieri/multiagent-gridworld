from enum import Enum
import gymnasium as gym
import networkx as nx
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    no_op = 4


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(self, render_mode=None, size=5, num_agents=2, cr=3):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.cr = cr
        self.num_agents = num_agents
        self.visited = np.zeros((size, size), dtype=bool)
        self._edge_list = []
        self._adj_matrix = np.zeros((self.num_agents, self.num_agents), dtype=np.float32)
        self._agent_locations = np.zeros((num_agents, 2), dtype=int)
        self._num_tile_visits = np.zeros((size, size), dtype=int)

        self.observation_space = spaces.Dict({
            "agents": spaces.Box(low=0, high=size-1, shape=(num_agents, 2), dtype=int),
            "map": spaces.Box(low=0, high=1, shape=(size, size), dtype=bool),
            "adj_matrix": spaces.Box(low=0, high=1, shape=(num_agents, num_agents), dtype=np.float32),
            # "edges": spaces.Tuple((
            #     spaces.MultiDiscrete([num_agents, num_agents]),
            # ))
        })

        # We have 5 actions for each agent.
        self.action_space = spaces.MultiDiscrete([5] * num_agents)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
            Actions.no_op.value: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_edges(self):
        edges = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(self._agent_locations[i] - self._agent_locations[j])
                if dist <= self.cr:
                    edges.append((i, j))  # Add undirected edge
                    edges.append((j, i))
        return edges

    def _get_obs(self):
        return {
            "agents": self._agent_locations,
            "map": self.visited,
            "adj_matrix": self._adj_matrix,
        }

    def _get_info(self):
        return {
            "coverage": np.sum(self.visited) / (self.size * self.size)
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.visited = np.zeros((self.size, self.size), dtype=bool)
        self._num_tile_visits = np.zeros((self.size, self.size), dtype=int)

        occupied_positions = set()
        # Randomize locations for all agents + ensure no overlap
        for i in range(self.num_agents):
            while True:
                loc = tuple(self.np_random.integers(0, self.size, size=2))
                if loc not in occupied_positions:
                    self._agent_locations[i] = loc
                    occupied_positions.add(loc)
                    break

        for loc in self._agent_locations:
            self.visited[loc[0], loc[1]] = True
            self._num_tile_visits[loc[0], loc[1]] += 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, actions):
        reward = 0.0
        occupied_positions = set(tuple(loc) for loc in self._agent_locations)
        new_agent_locations = []

        # propose movements
        for i, action in enumerate(actions):
            direction = self._action_to_direction[action]
            proposed_position = np.clip(
                self._agent_locations[i] + direction, 0, self.size - 1
            )

            if tuple(proposed_position) in occupied_positions:
                # Collision detected, keep the agent in its current position (no-op)
                new_agent_locations.append(self._agent_locations[i])
            else:
                # No collision, move to the new position
                new_agent_locations.append(proposed_position)
                occupied_positions.remove(tuple(self._agent_locations[i]))
                occupied_positions.add(tuple(proposed_position))

        self._agent_locations = np.array(new_agent_locations)

        # Mark the new position as visited + calc rewards
        for loc in self._agent_locations:
            if self.visited[loc[0], loc[1]]:
                dynamic_penalty = float(self._num_tile_visits[loc[0], loc[1]] * 0.5)
                reward -= dynamic_penalty
            else:
                reward += 1.0
            self.visited[loc[0], loc[1]] = True
            self._num_tile_visits[loc[0], loc[1]] += 1

        G = nx.Graph()
        G.add_nodes_from(range(self.num_agents))

        self._edge_list = self._get_edges()
        G.add_edges_from(self._edge_list)

        self._adj_matrix = nx.to_numpy_array(G, nodelist=G.nodes, dtype=np.float32)

        if not nx.is_connected(G):
            reward -= 5.0

        terminated = bool(np.all(self.visited))
        if terminated:
            reward += self.size**2

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the visited locations
        for i in range(self.size):
            for j in range(self.size):
                if self.visited[i][j]:
                    pygame.draw.rect(
                        canvas,
                        (87, 189, 209),
                        pygame.Rect(
                            pix_square_size * np.array([i, j]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Agents
        for loc in self._agent_locations:
            pygame.draw.circle(
                canvas,
                0,
                (loc + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # test edges
        edges = self._edge_list
        for a, b in edges:
            pygame.draw.line(
                canvas,
                (255, 0, 0),
                self._agent_locations[a] * pix_square_size + pix_square_size / 2,
                self._agent_locations[b] * pix_square_size + pix_square_size / 2,
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()