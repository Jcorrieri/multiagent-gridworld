from argparse import Namespace

import torch
import torch.nn as nn
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.Wrappers import CnnWrapper, TQDMCallback


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        assert len(observation_space.shape) == 3, "Expected 3D input (C,H,W)"
        assert observation_space.shape[0] == 3, "Expected 3 channels: map, frontier, agent"

        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            # Block 1: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: Conv2d -> BatchNorm2d -> ReLU
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 3: Conv2d -> BatchNorm2d -> ReLU
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten()
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.fc = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(observations))


class CustomPPO(nn.Module):
    def __init__(self, args: Namespace, default: bool = False):
        super(CustomPPO, self).__init__()
        self.args = args
        if args.test:
            self.model = PPO.load(self.args.model_path)
        elif default:
            self.model = PPO("MultiInputPolicy", args.env, verbose=0, ent_coef=0.01, device = args.device)
        else:
            env = CnnWrapper(args.env, args.size)
            policy_kwargs = dict(
                features_extractor_class=CustomCNNFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            )
            self.model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, device=args.device,
                             learning_rate=3e-4,
                             n_steps=2048,
                             batch_size=64,
                             n_epochs=10,
                             gamma=0.99,
                             gae_lambda=0.95,
                             clip_range=0.2,
                             vf_coef=0.5,
                             max_grad_norm=0.5)

    def forward(self, x):
        action, _states = self.model.predict(x)
        return action

    def learn(self):
        total_timesteps = 200000
        # curriculum_callback = CurriculumCallback(check_freq=10000,
        #                                          grid_size_start=self.args.size,
        #                                          grid_size_max=20)
        self.model.learn(total_timesteps=total_timesteps, callback=TQDMCallback(total_timesteps))
        self.model.save(self.args.model_path)