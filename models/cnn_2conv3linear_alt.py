import torch
import torch.nn as nn


class ActorCriticCNNModel(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super().__init__()

        h, w, c = obs_space.shape

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.conv(dummy)
            flattened_size = out.flatten(1).size(1)

        self.linear = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.actor_head = nn.Linear(256, num_outputs)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, obs):
        if obs.ndim == 4 and obs.shape[1] != self.obs_space.shape[-1]:
            obs = obs.permute(0, 3, 1, 2)

        x = self.conv(obs)
        x = self.shared_fc(x)

        logits = self.actor_head(x)
        value = self.critic_head(x)

        return logits, value