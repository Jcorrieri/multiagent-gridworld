import importlib
import os.path

import torch.nn as nn
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from models.cnn_1conv3linear import ActorCriticCNNModel


class CustomTorchModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        module_file = kwargs.get("module_file", "cnn_1conv3linear.py")
        if ".py" not in module_file:
            module_file += ".py"

        models_dir = os.path.abspath("models")
        module_path = os.path.join(models_dir, module_file)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "ActorCriticCNNModel"):
            raise NotImplementedError("ActorCriticCNNModel must be the name of your PyTorch module.")

        self.network = module.ActorCriticCNNModel(obs_space, num_outputs)
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