from enum import Enum
import numpy as np

from environment.envs.gridworld import GridWorldEnv


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    no_op = 4


class GridWorldEnvV2(GridWorldEnv):
    def __init__(self, env_params, **kwargs):
        super().__init__(env_params, **kwargs)

    def _calc_rewards(self, rewards: dict[str: float], connected: bool, collisions: [bool]):
        if connected:
            explorers = []
            maintainers = []

            for i, agent in enumerate(self.agents):
                current_pos = self.agent_locations[i]

                if collisions[agent]:
                    rewards[agent] += getattr(self, 'obstacle_penalty')

                if self.visited_tiles[current_pos[0], current_pos[1]] == 0:
                    explorers.append(agent)
                else:
                    maintainers.append(agent)

                self.visited_tiles[current_pos[0], current_pos[1]] = 1

            exploration_reward = len(explorers) * getattr(self, 'new_tile_visited')
            for agent in explorers:
                rewards[agent] += exploration_reward

            if explorers:
                maintenance_reward = len(explorers) * getattr(self, 'old_tile_maintainer')
                for agent in maintainers:
                    rewards[agent] += maintenance_reward
            else:
                for agent in maintainers:
                    rewards[agent] += getattr(self, 'old_tile_stagnant')

        else:
            for agent in self.agents:
                rewards[agent] += getattr(self, 'disconnected')

if __name__ == "__main__":
    env = GridWorldEnvV2({'render_mode': "human", 'map_dir_path': './obstacle-mats/testing', 'base_station': True})

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