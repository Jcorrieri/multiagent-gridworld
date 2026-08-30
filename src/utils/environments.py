from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env as register_ray_env

from environment.envs.baseline import BaselineEnv
from environment.envs.gridworld import GridWorldEnv
from environment.rewards import Components
from environment.rewards import Coverage
from environment.rewards import Default
from environment.rewards import ExplorerMaintainer
from environment.rewards import RewardScheme


def register_envs():
    register_ray_env("gridworld", lambda cfg: ParallelPettingZooEnv(GridWorldEnv(cfg)))


def make_reward_scheme(module) -> RewardScheme:
    if module == "coverage":
        return Coverage()
    elif module == "explorer_maintainer":
        return ExplorerMaintainer()
    elif module == "components":
        return Components()
    else:
        return Default()


def make_env(env_config: dict):
    name = env_config.get('env_name', 'default')
    if name == "baseline":
        return BaselineEnv(env_config)
    else:
        return GridWorldEnv(env_config)
