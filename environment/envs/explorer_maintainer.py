import numpy as np
from environment.envs.gridworld import GridWorldEnv


class ExplorerMaintainerEnv(GridWorldEnv):
    def __init__(self, env_params, **kwargs):
        super().__init__(env_params, **kwargs)

    def _calc_rewards(self, rewards: dict[str: float], step_info):
        explorers = []
        maintainers = []

        for i, agent in enumerate(self.agents):
            current_pos = self.agent_locations[i]

            if step_info['collisions'][agent]:
                rewards[agent] += getattr(self, 'obstacle_penalty')

            if self.visited_tiles[current_pos[0], current_pos[1]] == 0:
                explorers.append(agent)
            else:
                maintainers.append(agent)

            self.visited_tiles[current_pos[0], current_pos[1]] = 1

        if step_info['connected']:
            explorer_reward = getattr(self, 'explorer') + (step_info['coverage'] / 100)
            for agent in explorers:
                rewards[agent] += explorer_reward

            if explorers:
                for agent in maintainers:
                    rewards[agent] += getattr(self, 'maintainer_percentage') * explorer_reward
            else:
                for agent in maintainers:
                    rewards[agent] += getattr(self, 'stagnation_penalty')

        else:
            for agent in self.agents:
                rewards[agent] += getattr(self, 'disconnected')

if __name__ == "__main__":
    reward_scheme = {
        'explorer': 1.0,
        'maintainer_percentage': 0.5,
        'stagnation_penalty': -0.1,
        'disconnected': -0.1,
        'obstacle_penalty': 0.0,
        'terminated': 100
    }

    env = ExplorerMaintainerEnv({
        'render_mode': "human",
        'map_dir_path': '../obstacle-mats/testing',
        'base_station': True,
        'fov': 3,
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