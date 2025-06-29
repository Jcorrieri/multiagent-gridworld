import torch
import torch.nn as nn


class CentralizedCriticCNNModel(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super().__init__()

        self.obs_space = obs_space

        h, w, c = obs_space.shape  # HWC

        # actor (default) cnn
        self.actor_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_actor = torch.zeros(1, c, h, w)  # c = 4 (or whatever obs_space.shape[2])
            actor_flattened_size = self.actor_conv(dummy_actor).flatten(1).size(1)

        self.actor_head = nn.Sequential(
            nn.Linear(actor_flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

        # new critic cnn
        self.critic_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_critic = torch.zeros(1, 3, h, w)  # explicitly 3 channels
            critic_flattened_size = self.critic_conv(dummy_critic).flatten(1).size(1)

        self.critic_head = nn.Sequential(
            nn.Linear(critic_flattened_size, 256),
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
        logits = self.actor_head(x)

        # Centralized critic branch
        if critic_obs is not None:
            if critic_obs.ndim == 4 and critic_obs.shape[1] != 3:  # 3 = num channels
                critic_obs = critic_obs.permute(0, 3, 1, 2)

            print("TORCH FOWARD SHAPE", critic_obs.shape)

            if not isinstance(critic_obs, torch.Tensor):
                critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=obs.device)
            global_x = self.critic_conv(critic_obs)
            self._value_out = self.critic_head(global_x)
        else:
            self._value_out = torch.zeros(obs.shape[0], 1, device=obs.device)

        return logits

    def value_function(self):
        return self._value_out.view(-1)


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

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     nn.Flatten()
        # )

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
        if obs.ndim == 4 and obs.shape[1] != self.obs_space.shape[-1]:  # PettingZoo obs are [B, H, W, C]
            obs = obs.permute(0, 3, 1, 2)  # Torch expects [B, C, H, W]

        x = self.conv(obs)
        x = self.linear(x)

        logits = self.actor_head(x)
        value = self.critic_head(x)

        return logits, value