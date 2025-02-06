from argparse import Namespace

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomPPOExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, grid_size: int, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.grid_size = grid_size
        self.kernel_size = 5
        self.stride = 1

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=grid_size,
                kernel_size=self.kernel_size,
                stride=self.stride
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=grid_size,
                out_channels=grid_size,
                kernel_size=self.kernel_size,
                stride=self.stride
            ),
            nn.ReLU(),
            nn.Flatten()
        )

        def convert_to_size(size, kernel_size=5, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        self.width = convert_to_size(convert_to_size(grid_size))
        self.height = convert_to_size(convert_to_size(grid_size))

        self.linear = nn.Sequential(
            nn.Linear(grid_size * self.width * self.height, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )

    def create_agent_channel(self, robot_positions, batch_size):
        agent_channel = torch.zeros((batch_size, self.grid_size, self.grid_size),
                                    device=robot_positions.device)
        for b in range(batch_size):
            pos = robot_positions[b]  # Shape: (n_robots, 2)
            agent_channel[b, pos[:, 0].long(), pos[:, 1].long()] = 1

        return agent_channel

    def forward(self, observations):
        batch_size = observations['map'].shape[0]
        grid_channel = observations['map'].float()
        agent_channel = self.create_agent_channel(observations['agents'], batch_size)
        x = torch.stack([grid_channel, agent_channel], dim=1)
        cnn_features = self.cnn(x)
        features = self.linear(cnn_features)

        return features


class CustomPPO(nn.Module):
    def __init__(self, args: Namespace):
        super(CustomPPO, self).__init__()
        self.args = args
        if args.test:
            self.model = PPO.load("models/saved/Custom_PPO")
        else:
            policy_kwargs = dict(
                features_extractor_class=CustomPPOExtractor,
                features_extractor_kwargs=dict(grid_size=args.size,features_dim=64),
            )
            self.model = PPO("MultiInputPolicy", self.args.env, policy_kwargs=policy_kwargs, verbose=0,
                             learning_rate=3e-4,
                             n_steps=2048,
                             batch_size=64,
                             n_epochs=10,
                             gamma=0.99,
                             gae_lambda=0.95,
                             clip_range=0.2)

    def forward(self, x):
        action, _states = self.model.predict(x)
        return action

    def learn(self, logger: Logger):
        total_timesteps = 100000
        # curriculum_callback = CurriculumCallback(check_freq=10000,
        #                                          grid_size_start=self.args.size,
        #                                          grid_size_max=20)
        self.model.set_logger(logger)
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save("models/saved/Custom_PPO")