import argparse
import warnings
import torch
import yaml

from test import test
from train import train
from utils import parse_optimizer, register_envs, make_reward_scheme


def main():
    parser = argparse.ArgumentParser()
    parse_optimizer(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    with open(f"config/{args.config}", 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['testing'].get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['testing'].get("seed", 42))

    if args.test:
        map_dir_path = "environment/obstacle-mats/testing"
    else:
        map_dir_path = "environment/obstacle-mats/training"

    reward_scheme_module = config['rewards']['module']
    reward_scheme_params = config['rewards']['params']
    reward_scheme = make_reward_scheme(reward_scheme_module, reward_scheme_params)

    register_envs()

    env_config = dict(
        map_dir_path=map_dir_path,
        render_mode="rgb_array",
        reward_scheme=reward_scheme,
        **config['environment']
    )

    if args.test:
        test(env_config, config['testing'])
    else:
        train(args, env_config, config['training'])

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
