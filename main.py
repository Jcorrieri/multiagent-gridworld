import argparse
import warnings

import torch
import yaml
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env

from env.grid_world import GridWorldEnv
from test import test
from train import train
from utils import parse_optimizer


def main():
    parser = argparse.ArgumentParser()
    parse_optimizer(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(f"config/default", 'r') as f:
        config = yaml.safe_load(f)

    # for rllib
    register_env("gridworld", lambda cfg: ParallelPettingZooEnv(GridWorldEnv(cfg)))

    if args.test:
        map_dir_path = "./env/obstacle-mats/testing"
    else:
        map_dir_path = "./env/obstacle-mats/training"

    env_config = dict(
        map_dir_path=map_dir_path,
        render_mode="rgb_array",
        reward_scheme=config['reward_scheme'],
        **config['environment']
    )

    if args.test:
        test(env_config, config['testing'])
    else:
        train(args, env_config, config['training'])

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
