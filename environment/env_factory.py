from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env

from environment.envs.gridworld import GridWorldEnv
from environment.envs.gridworldv2 import GridWorldEnvV2


def register_custom_envs():
    register_env("default", lambda cfg: ParallelPettingZooEnv(GridWorldEnv(cfg)))
    register_env("alt_reward", lambda cfg: ParallelPettingZooEnv(GridWorldEnvV2(cfg)))

def make_env(env_config: dict):
    name = env_config.get('env_name', 'default')

    if name == "default":
        return GridWorldEnv(env_config)
    elif name == "alt_reward":
        return GridWorldEnvV2(env_config)
    # elif name == "baseline":
    #     pass