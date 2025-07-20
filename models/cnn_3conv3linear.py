import torch
import torch.nn as nn


class ActorCriticCNNModel(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super().__init__()

        self.obs_space = obs_space

        h, w, c = obs_space.shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces spatial dimension

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.conv(dummy)
            flattened_size = out.flatten(1).size(1)

        self.actor_head = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        if obs.ndim == 4 and obs.shape[1] != self.obs_space.shape[-1]:
            obs = obs.permute(0, 3, 1, 2)

        x = self.conv(obs)

        logits = self.actor_head(x)
        value = self.critic_head(x)

        return logits, value