from environment.envs import gridworldv2, gridworld
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env


def register_custom_envs(name: str):
    register_env(name, lambda cfg: ParallelPettingZooEnv(gridworld.GridWorldEnv(cfg)))
    register_env(name, lambda cfg: ParallelPettingZooEnv(gridworldv2.GridWorldEnv(cfg)))

def make_env(env_config: dict):
    name = env_config.get('env_name', 'gridworld')

    if name == "gridworld":
        return gridworld.GridWorldEnv(env_config)
    elif name == "gridworldv2":
        return gridworldv2.GridWorldEnv(env_config)
    # elif name == "baseline":
    #     pass