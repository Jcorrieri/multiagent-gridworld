import torch.nn as nn
import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from models.cnn import CentralizedCriticCNNModel, ActorCriticCNNModel

class CentralizedCriticCallback(DefaultCallbacks):
    def on_postprocess_trajectory(
            self,
            *,
            worker,
            episode,
            agent_id,
            policy_id,
            policies,
            postprocessed_batch,
            original_batches,
            **kwargs,
    ):
        obs_self = postprocessed_batch["obs"]
        all_obs = [obs_self]

        for other_id, (other_policy_id, other_policy, other_batch) in original_batches.items():
            if other_id == agent_id:
                continue
            all_obs.append(other_batch["obs"])

        centralized_obs = np.concatenate(all_obs, axis=-1)
        postprocessed_batch["critic_obs"] = centralized_obs


class CentralizedCriticWrappedModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        critic_obs_dim = model_config["custom_model_config"]["critic_obs_dim"]

        self.base = CentralizedCriticCNNModel(obs_space, num_outputs, critic_obs_dim)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        critic_obs = input_dict.get("critic_obs", None)
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