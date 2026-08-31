import importlib
from pathlib import Path

import torch.nn as nn
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CustomTorchModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)       

        module_file = kwargs['module_file']
        if ".py" not in module_file:
            module_file += ".py"
        # get torch model from config
        module_path = Path(__file__).parent / "arch" / module_file
        module_name = module_path.stem
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "ActorCriticCNNModel"):
            raise NotImplementedError("ActorCriticCNNModel must be the name of your PyTorch module.")

        self.network = module.ActorCriticCNNModel(obs_space, num_outputs)
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        logits, value = self.network(obs)
        self._value_out = value
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out.squeeze(1)
