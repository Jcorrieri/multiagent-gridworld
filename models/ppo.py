import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from utils import MinimalLogger


# create larger grid environments incrementally
class CurriculumCallback(BaseCallback):
    def __init__(self, check_freq, grid_size_start=5, grid_size_max=25,
                 success_threshold=0.95, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.grid_size_start = grid_size_start
        self.grid_size_current = grid_size_start
        self.grid_size_max = grid_size_max
        self.success_threshold = success_threshold
        self.successes = 0

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if self.successes / self.check_freq > self.success_threshold:
                if self.grid_size_current < self.grid_size_max:
                    self.grid_size_current += 1
                    self.training_env.env_method("update_grid_size",
                                                 self.grid_size_current)
                    self.successes = 0

        return True


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.flatten_dim = int(np.prod(observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class CustomPPO(nn.Module):
    def __init__(self, args: Namespace):
        super(CustomPPO, self).__init__()
        self.model = PPO.load("PPO_test") if args.test else None
        self.args = args

    def forward(self, x):
        action, _states = self.model.predict(x)
        return action

    def train_ppo(self):
        check_env(self.args.env, warn=True)
        model = PPO("MlpPolicy", self.args.env, verbose=0,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2)

        total_timesteps = 200000
        curriculum_callback = CurriculumCallback(check_freq=10000,
                                                 grid_size_start=self.args.size,
                                                 grid_size_max=20,
                                                 success_threshold=0.95)

        new_logger = Logger(folder=None, output_formats=[MinimalLogger(sys.stdout)])
        model.set_logger(new_logger)
        model.learn(total_timesteps=total_timesteps)
        model.save("PPO_test")
        self.model = model