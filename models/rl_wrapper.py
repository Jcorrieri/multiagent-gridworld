import importlib
import os.path

import torch.nn as nn
import torch
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CustomTorchModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Get configuration
        custom_config = model_config.get('custom_model_config', {})
        module_file = kwargs.get("module_file", custom_config.get('module_file'))
        num_agents = custom_config.get('num_agents', 1)
        
        if ".py" not in module_file:
            module_file += ".py"
        
        # Dynamically load the model architecture
        models_dir = os.path.abspath("models/arch")
        module_path = os.path.join(models_dir, module_file)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "DQNModel"):
            raise NotImplementedError("DQNModel must be the name of your PyTorch module.")

        # Initialize the network with agent information
        self.network = module.DQNModel(obs_space, num_outputs, num_agents=num_agents)
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        
        if isinstance(obs, dict):
            agent_id_batch = obs["agent_id"] 
        else:
            # Fallback if agent_id isn't in this specific dict
            agent_id_batch = torch.zeros((obs.shape[0], self.network.num_agents), device=obs.device)
            agent_id_batch[:, 0] = 1 

        agent_indices = torch.argmax(agent_id_batch, dim=-1)
            
        q_values, state_value = self.network(obs, agent_id=agent_indices)
        
        self._value_out = state_value
        return q_values, state

    @override(TorchModelV2)
    def value_function(self):
        if self._value_out is None:
            return torch.zeros(1)
        return self._value_out.squeeze(1)
