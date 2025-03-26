from typing import Any

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
import torch.nn as nn

from models.cnn import ActorCriticCNNModel


class CustomTorchModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.network = ActorCriticCNNModel(obs_space, num_outputs)
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        logits, value = self.network(obs)
        self._value_out = value.view(-1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out


class CustomRLModule(TorchRLModule):
    def setup(self):
        self.model = ActorCriticCNNModel(
            obs_space=self.observation_space,
            num_outputs=self.action_space.n
        )

    def _forward(self, batch, **kwargs) -> dict[str, Any]:
        obs_batch = batch[Columns.OBS]
        logits, value = self.model(obs_batch)
        self._value_out = value.view(-1)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: self._value_out,
        }

    def value_function(self):
        return self._value_out