import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticCNNModel(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super().__init__()
        self.obs_space = obs_space
        h, w, c = obs_space.shape  # HWC

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=5, stride=1, padding=2),  # keep HxW
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # keep HxW
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # NEW: GAP instead of Flatten
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # [B,64,H,W] -> [B,64,1,1]

        # Heads now take a 64-d vector (since last conv has 64 channels)
        in_feat = 64
        self.actor_head = nn.Sequential(
            nn.Linear(in_feat, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(in_feat, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        if obs.ndim == 4 and obs.shape[1] != self.obs_space.shape[-1]:  # [B,H,W,C] -> [B,C,H,W]
            obs = obs.permute(0, 3, 1, 2)

        x = self.conv(obs)                   # [B,64,H,W]
        g = self.gap(x).flatten(1)           # [B,64]
        logits = self.actor_head(g)          # [B,num_outputs]
        value  = self.critic_head(g)         # [B,1]
        return logits, value
