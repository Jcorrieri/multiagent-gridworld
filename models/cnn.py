import torch
import torch.nn as nn


class ActorCriticCNNModel(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super().__init__()

        h, w, c = obs_space.shape  # HWC

        kernel_size = 3
        stride = 2

        self.conv = nn.Sequential(
            nn.ConstantPad2d(1, -1),  # Pad 1 pixel border with -1s (like MATLAB)
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=kernel_size, stride=stride, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=1),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.conv(dummy)
            flattened_size = out.flatten(1).size(1)

        self.linear = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(256, num_outputs)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, obs):
        obs = obs.float()
        if obs.ndim == 4 and obs.shape[1] != 3:
            obs = obs.permute(0, 3, 1, 2)
        x = self.conv(obs)
        x = self.linear(x)
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value