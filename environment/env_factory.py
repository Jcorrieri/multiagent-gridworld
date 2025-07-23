from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env

from environment.envs.gridworld import GridWorldEnv
from environment.envs.explorer_maintainer import ExplorerMaintainerEnv
from environment.envs.component_based import ComponentBasedEnv


def register_custom_envs():
    register_env("default", lambda cfg: ParallelPettingZooEnv(GridWorldEnv(cfg)))
    register_env("alt_reward", lambda cfg: ParallelPettingZooEnv(ExplorerMaintainerEnv(cfg)))

def make_env(env_config: dict):
    name = env_config.get('env_name', 'default')

    if name == "default":
        return GridWorldEnv(env_config)
    elif name == "explorer_maintainer":
        return ExplorerMaintainerEnv(env_config)
    elif name == "component_based":
        pass
    elif name == "baseline":
        pass