import networkx as nx
import numpy as np
from environment.envs.gridworld import GridWorldEnv


class ComponentBasedEnv(GridWorldEnv):
    def __init__(self, env_params, **kwargs):
        super().__init__(env_params, **kwargs)

    def _calc_rewards(self, rewards: dict[str: float], step_info):
        G: nx.Graph = step_info['graph']
        components = nx.connected_components(G)

        for agent in self.agents:
            rewards[agent] += getattr(self, 'timestep_penalty')

if __name__ == "__main__":
    reward_scheme = {
        'explorer': 1.0,
        'coverage_bonus': 1.0,
        'maintainer_percentage': 0.5,
        'timestep_penalty': -0.01,
        'termination_bonus': 100
    }

    env = ComponentBasedEnv({
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