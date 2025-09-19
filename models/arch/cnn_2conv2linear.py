import torch
import torch.nn as nn


class ActorCriticCNNModel(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super().__init__()

        self.obs_space = obs_space

        h, w, c = 25, 25, 4  # HWC

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.conv(dummy)
            flattened_size = out.flatten(1).size(1)

        self.actor_head = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    @staticmethod
    def _to_nchw(x):
            # RLlib usually passes [B,H,W,C]; convert to [B,C,H,W]
            return x.permute(0, 3, 1, 2) if x.ndim == 4 else x

    def forward(self, obs):
        if not hasattr(obs, "ndim"):
            a_obs = obs["actor"]
            c_obs = obs["critic"]

        if isinstance(obs, dict):
            a_obs = self._to_nchw(obs["actor"])
            c_obs = self._to_nchw(obs["critic"])
        else:
            x = self._to_nchw(obs)
            a_obs = c_obs = x

        # Actor
        z_pi = self.conv(a_obs)
        logits = self.actor_head(z_pi)

        # Critic
        z_v = self.conv(c_obs)
        value = self.critic_head(z_v)

        return logits, value