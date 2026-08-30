import argparse
import warnings
import torch
import yaml
from pathlib import Path
from typing import Literal

from test import test
from train import train
from utils.environments import make_reward_scheme, register_envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["train", "test"], required=True)
    parser.add_argument('--config', type=str, default='default')
    args = parser.parse_args()

    mode = args.mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    config_path = Path("config") / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config[mode].get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['testing'].get("seed", 42))

    map_dir_path = Path("maps") / f"{mode}ing"

    reward_scheme_module = config['environment']['reward_scheme']
    reward_scheme = make_reward_scheme(reward_scheme_module)
    config['environment']['reward_scheme'] = reward_scheme

    register_envs()

    env_config = dict(
        map_dir_path=map_dir_path,
        render_mode="rgb_array",
        **config['environment']
    )

    if args.mode == "train":
        train(args, env_config, config['training'])
    else:
        test(env_config, config['testing'])

if __name__ == "__main__":
    main()
