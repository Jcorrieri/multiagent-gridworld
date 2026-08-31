import yaml
import torch
from pathlib import Path
from argparse import ArgumentParser

from utils.environments import register_envs


def init_script(type: str = "training"):
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='default')
    args = parser.parse_args()

    config_path = Path("config") / args.config
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    seed = config[type].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    register_envs()
    env_config = dict(
        map_dir_path=Path("maps") / type,
        render_mode="rgb_array",
        **config['environment']
    )

    return env_config, config[type], config_path

