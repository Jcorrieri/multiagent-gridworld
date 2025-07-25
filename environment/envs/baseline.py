import os
from copy import copy

import networkx as nx
import numpy as np
from numpy import ndarray

from environment.envs.gridworld import GridWorldEnv


class BaselineEnv(GridWorldEnv):
    metadata = {
        "name": "baseline",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 24
    }

    def execute_config(self, movements) -> list:
        new_positions = []
        for i, agent in enumerate(self.agents):
            action = movements[i]
            direction = self._action_to_direction[action]
            current_position = self.agent_locations[i]
            proposed_position = tuple(current_position + direction)
            new_positions.append(proposed_position)

        return new_positions

    def compute_fitness(self, config: ndarray, obstacles: list[tuple]) -> float:
        config_fitness = 0.0

        new_positions = self.execute_config(config)

        arrays_list = [np.array(t) for t in new_positions]

        self._build_adj_matrix(arrays_list)
        G = nx.from_numpy_array(self.adj_matrix)
        connected = nx.is_connected(G)

        for i, position in enumerate(new_positions):
            utility = 0.0

            out_of_bounds_r = position[0] < 0 or position[0] >= self.size
            out_of_bounds_c = position[1] < 0 or position[1] >= self.size
            in_obstacle = position in obstacles

            agent_collision = new_positions.count(position) > 1

            if out_of_bounds_r or out_of_bounds_c or in_obstacle or agent_collision or not connected:
                utility += -3.0  # impossible
            else:
                pass # TODO -- calculate manhattan distance to nearest unexplored cell

            config_fitness += utility

        return config_fitness

    def base_station(self):
        k = 50
        obstacles = [(x, y) for x, y in self.obs_mat]

        # generate a population
        configurations = []
        for i in range(k):
            # generate a config change
            new_moves = np.random.default_rng().integers(low=0, high=5, size=5)
            configurations.append(new_moves)

        # compute fitness; find maximum
        config_max = configurations[0]
        max_fitness = self.compute_fitness(config_max, obstacles)
        for i in range(1, k):
            config_fitness = self.compute_fitness(configurations[i], obstacles)
            if config_fitness > max_fitness:
                config_max = configurations[k]
                max_fitness = config_fitness

        self.execute_config(config_max)

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
        map_path = os.path.join(self.map_dir_path, f'mat{mat_idx}')
        self.obs_mat = np.loadtxt(map_path, delimiter=' ', dtype='int')

        # self.visited_tiles[self.obs_mat[:, 0], self.obs_mat[:, 1]] = 1.0  # count obstacle tiles as visited

        self.max_coverage = self.size**2 - len(self.obs_mat)

        self._generate_spawns()
        self._build_adj_matrix()

        if self.use_local_fov:
            self._generate_local_obs()

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._generate_observation(i)
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

        previous_locations = self.agent_locations.copy()

        occupied_positions = set([(x, y) for x, y in self.obs_mat])
        for pos in previous_locations:
            occupied_positions.add(tuple(pos))

        # Determine new positions
        new_positions = []
        collisions = {agent: False for agent in self.agents}
        for i, agent in enumerate(self.agents):
            action = actions[agent]
            direction = self._action_to_direction[action]
            proposed_position = np.clip(previous_locations[i] + direction, 0, self.size - 1)

            # Check if proposed position is already claimed
            pos_tuple = tuple(proposed_position)
            if pos_tuple not in occupied_positions:
                new_positions.append(proposed_position)
                occupied_positions.remove(tuple(previous_locations[i]))
                occupied_positions.add(pos_tuple)
            else:
                # If collision, don't move
                new_positions.append(previous_locations[i])
                collisions[agent] = True

        if self.base_station:
            new_positions.append((self.size - 1,0))
        self.agent_locations = np.array(new_positions)

        self._build_adj_matrix()
        G = nx.from_numpy_array(self.adj_matrix)
        connected = nx.is_connected(G)

        coverage = np.sum(self.visited_tiles > 0) / self.max_coverage

        step_info = {
            "coverage": coverage * 100,
            "timestep": self.timestep,
            "connected": connected,
            "collisions": collisions,
            "graph": G,
        }

        # Calculate rewards
        self._calc_rewards(rewards, step_info)

        if self.use_local_fov:
            self._generate_local_obs()

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._generate_observation(i)
            infos[agent] = {
                "coverage": step_info['coverage'],
                "step": step_info['timestep'],
                "connection_broken": not step_info['connected'],
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

if __name__ == "__main__":
    reward_scheme = {
        'new_tile_visited_connected': 4.0,
        'old_tile_visited_connected': -0.1,
        'new_tile_visited_disconnected': -0.5,
        'old_tile_visited_disconnected': -0.8,
        'obstacle_penalty': -1.0,
        'terminated': 200
    }

    env = GridWorldEnv({
        'render_mode': "human",
        'map_dir_path': '../obstacle-mats/testing',
        'base_station': True,
        'fov': 25,
        'reward_scheme': reward_scheme
    })

    # unit test -- default env
    obs, _ = env.reset()
    episode_over = False
    while not episode_over:
        vals = np.random.default_rng().integers(low=0, high=5, size=5)
        actions_dict = {f'agent_{i}': int(val) for i, val in enumerate(vals)}
        observations, rewards, terminated, truncated, infos = env.step(actions_dict)
        episode_over = all(terminated.values()) or all(truncated.values())

    # parallel_api_test(env, num_cycles=1_000_000)
    env.close()