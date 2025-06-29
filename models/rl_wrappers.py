import numpy as np
import torch.nn as nn
from gymnasium.spaces import Box
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from models.cnn import CentralizedCriticCNNModel, ActorCriticCNNModel


class CentralizedCriticWrappedModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.base = CentralizedCriticCNNModel(obs_space, num_outputs)

    def forward(self, input_dict, state, seq_lens):
        device = next(self.parameters()).device
        obs = input_dict["obs"].float().to(device)
        critic_obs = input_dict.get("global_state", None)
        return self.base(obs, critic_obs), state

    def value_function(self):
        return self.base.value_function()


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