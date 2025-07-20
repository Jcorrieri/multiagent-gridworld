import torch
import torch.nn as nn


class ActorCriticCNNModel(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super().__init__()

        self.obs_space = obs_space

        h, w, c = obs_space.shape  # HWC

        # MATLAB
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.conv(dummy)
            flattened_size = out.flatten(1).size(1)

        self.actor_head = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        if obs.ndim == 4 and obs.shape[1] != self.obs_space.shape[-1]:  # PettingZoo obs are [B, H, W, C]
            obs = obs.permute(0, 3, 1, 2)  # Torch expects [B, C, H, W]

        x = self.conv(obs)

        logits = self.actor_head(x)
        value = self.critic_head(x)

        return logits, value