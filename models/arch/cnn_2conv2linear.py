import torch
import torch.nn as nn


class ActorCriticCNNModel(nn.Module):
    def __init__(self, obs_space, num_outputs):
        super().__init__()

        self.obs_space = obs_space

        # h, w, c = obs_space["agent_0"]["actor"].shape  # HWC
        h, w, c = 25, 25, 4  # NOTE: hardcoded for now due to dict preprocessing bug

        self.actor_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.actor_conv(dummy)
            actor_flattened_size = out.flatten(1).size(1)

        self.actor_linear = nn.Sequential(
            nn.Linear(actor_flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)  # output layer
        )  

        ch, cw, cc = 25, 25, 4  # NOTE: hardcoded for now due to dict preprocessing bug

        self.critic_conv = nn.Sequential(
            nn.Conv2d(in_channels=cc, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, cc, ch, cw)
            out = self.critic_conv(dummy)
            critic_flattened_size = out.flatten(1).size(1)

        self.critic_linear = nn.Sequential(
            nn.Linear(critic_flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # output layer
        )

    def forward(self, obs):
        # obs is a dict: {"actor": ..., "critic": ...}
        actor_obs = obs["actor"]
        critic_obs = obs["critic"]

        # Actor branch
        if actor_obs.ndim == 4 and actor_obs.shape[1] != self.obs_space.shape[-1]:
            actor_obs = actor_obs.permute(0, 3, 1, 2)
        x_actor = self.actor_conv(actor_obs)
        logits = self.actor_linear(x_actor)

        # Critic branch
        if critic_obs.ndim == 4 and critic_obs.shape[1] != self.obs_space.shape[-1]:
            critic_obs = critic_obs.permute(0, 3, 1, 2)
        x_critic = self.critic_conv(critic_obs)
        value = self.critic_linear(x_critic)

        return logits, value