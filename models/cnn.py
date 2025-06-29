import torch
import torch.nn as nn


class CentralizedCriticCNNModel(nn.Module):
    def __init__(self, obs_space, num_outputs, critic_obs_dim):
        super().__init__()

        self.obs_space = obs_space

        h, w, c = obs_space.shape  # HWC

        # Actor (local obs) CNN
        self.actor_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flattened_size = self.actor_conv(dummy).flatten(1).size(1)

        self.actor_linear = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(256, num_outputs)

        # Critic (centralized obs) MLP — input is a flat tensor
        self.critic_net = nn.Sequential(
            nn.Linear(critic_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._value_out = None

    def forward(self, obs, critic_obs=None):
        if obs.ndim == 4 and obs.shape[1] != self.obs_space.shape[-1]:
            obs = obs.permute(0, 3, 1, 2)  # Convert [B, H, W, C] to [B, C, H, W]

        # Actor branch
        x = self.actor_conv(obs)
        x = self.actor_linear(x)
        logits = self.actor_head(x)

        # Centralized critic branch
        if critic_obs is not None:
            if not isinstance(critic_obs, torch.Tensor):
                critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=obs.device)
            critic_obs = critic_obs.view(critic_obs.size(0), -1)  # Flatten to [B, D]
            self._value_out = self.critic_net(critic_obs)
        else:
            self._value_out = torch.zeros(obs.size(0), 1, device=obs.device)

        return logits

    def value_function(self):
        return self._value_out.view(-1)


class ActorCriticCNNModel(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super().__init__()

        self.obs_space = obs_space

        h, w, c = obs_space.shape  # HWC

        # MATLAB
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     nn.Flatten()
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten()
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
        torch.autograd.set_detect_anomaly(True)

        if obs.ndim == 4 and obs.shape[1] != self.obs_space.shape[-1]:  # PettingZoo obs are [B, H, W, C]
            obs = obs.permute(0, 3, 1, 2)  # Torch expects [B, C, H, W]

        x = self.conv(obs)
        x = self.linear(x)

        logits = self.actor_head(x)
        value = self.critic_head(x)

        return logits, value