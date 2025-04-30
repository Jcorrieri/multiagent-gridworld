from collections import defaultdict

import torch
import torch.nn as nn
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from models.cnn import ActorCriticCNNModel


class AgentRewardAndLengthLogger(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.agent_episode_rewards = defaultdict(list)  # agent_id → list of total episode rewards
        self.episode_lengths = []  # list of episode lengths

    def on_episode_end(self, *, worker, episode, **kwargs):
        # Track reward per agent
        for agent_id in episode.agent_rewards:
            total_reward = sum(r for (_, r) in episode.agent_rewards[agent_id])
            self.agent_episode_rewards[agent_id].append(total_reward)

        # Track episode length
        self.episode_lengths.append(episode.length)


class CustomTorchModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.network = ActorCriticCNNModel(obs_space, num_outputs)
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        device = next(self.parameters()).device
        obs = input_dict["obs"].float().to(device)
        logits, value = self.network(obs)
        self._value_out = value
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out.squeeze(1)