from argparse import Namespace

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomPPOExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        grid_shape = observation_space['map'].shape
        self.n = grid_shape[0]
        num_agents = observation_space['agents'].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(2, self.n, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.n, self.n, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        test_input = torch.zeros(1, 2, self.n, self.n)
        cnn_output_dim = self.cnn(test_input).view(1, -1).size(1)

        self.agent_fc = nn.Sequential(
            nn.Linear(num_agents * 2, 32),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, observations: dict) -> torch.Tensor:
        grid_map = observations['map'].float().unsqueeze(1)
        robot_positions = observations['agents'].float()

        robot_positions_map = torch.zeros_like(grid_map)
        for idx, agent_pos in enumerate(robot_positions):
            x, y = agent_pos.int()
            robot_positions_map[0, 0, x, y] = 1

        combined_map = torch.cat([grid_map, robot_positions_map], dim=1)
        cnn_features = self.cnn(combined_map)
        agent_locations = observations['agents'].float().view(observations['agents'].size(0), -1)
        agent_features = self.agent_fc(agent_locations)
        combined_features = torch.cat([cnn_features, agent_features], dim=1)

        final_features = self.fc(combined_features)

        return final_features


class CustomPPO(nn.Module):
    def __init__(self, args: Namespace):
        super(CustomPPO, self).__init__()
        self.args = args
        if args.test:
            self.model = PPO.load("models/saved/Custom_PPO")
        else:
            policy_kwargs = dict(
                features_extractor_class=CustomPPOExtractor,
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

    # def set_env(self, env):
    #     self.model.set_env(env)