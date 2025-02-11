from argparse import Namespace

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import DQN

class CustomDQNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        # Define your custom CNN layers here, based on observation space
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Linear layers to map CNN features to output dimension
        self.fc = nn.Sequential(
            nn.Linear(64 * observation_space.shape[1] * observation_space.shape[2], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.fc(x)


class TestDQN(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, n_actions: int):
        super(TestDQN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = CustomDQNExtractor(observation_space)

        self.q_network = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(observation)
        q_values = self.q_network(features)
        return q_values


class CustomDQN(nn.Module):
    def __init__(self, args: Namespace):
        super(CustomDQN, self).__init__()
        self.args = args
        if args.test:
            self.model = DQN.load("models/saved/Custom_PPO")
        else:
            policy_kwargs = {
                "q_net_class": TestDQN,
                "features_extractor_class": CustomDQNExtractor,
                "features_extractor_kwargs": {"features_dim": 64},  # Specify your desired feature size
            }
            self.model = DQN("CnnPolicy", args.env, policy_kwargs=policy_kwargs, verbose=1)

    def forward(self, x):
        action, _states = self.model.predict(x)
        return action

    def learn(self, logger: Logger):
        total_timesteps = 200000
        # curriculum_callback = CurriculumCallback(check_freq=10000,
        #                                          grid_size_start=self.args.size,
        #                                          grid_size_max=20)
        self.model.set_logger(logger)
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save("models/saved/Custom_PPO")