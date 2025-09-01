
import importlib
import os.path
from typing import List, Tuple

import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


def _flatten_state(h: torch.Tensor, c: torch.Tensor) -> List[torch.Tensor]:
    # h,c: [num_layers, B, hidden] -> [B, num_layers*hidden]
    h_flat = h.permute(1, 0, 2).reshape(h.shape[1], -1).contiguous()
    c_flat = c.permute(1, 0, 2).reshape(c.shape[1], -1).contiguous()
    return [h_flat, c_flat]


def _unflatten_state(h_flat: torch.Tensor, c_flat: torch.Tensor, num_layers: int, hidden_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # h_flat,c_flat: [B, num_layers*hidden] -> [num_layers, B, hidden]
    B = h_flat.shape[0]
    h = h_flat.view(B, num_layers, hidden_size).permute(1, 0, 2).contiguous()
    c = c_flat.view(B, num_layers, hidden_size).permute(1, 0, 2).contiguous()
    return h, c


class CustomTorchModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        module_file = kwargs['module_file']
        if ".py" not in module_file:
            module_file += ".py"
        # get torch model from config
        models_dir = os.path.abspath("models/arch")
        module_path = os.path.join(models_dir, module_file)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Prefer LSTM class if present; else fall back to CNN
        if hasattr(module, "ActorCriticCNNLSTM"):
            Net = module.ActorCriticCNNLSTM
        elif hasattr(module, "ActorCriticCNNModel"):
            Net = module.ActorCriticCNNModel
        else:
            raise NotImplementedError("Expected ActorCriticCNNLSTM or ActorCriticCNNModel in the module.")

        self.network = Net(obs_space, num_outputs)
        self._value_out = None

        # Sequence/rnn config
        self.max_seq_len = model_config.get("max_seq_len", 16)
        self._is_recurrent = hasattr(self.network, "lstm")
        if self._is_recurrent:
            self._num_layers = self.network.lstm.num_layers
            self._hidden_size = self.network.lstm.hidden_size

    # ---- RLlib RNN hooks ----
    @override(TorchModelV2)
    def get_initial_state(self):
        if not self._is_recurrent:
            return []
        # Return two tensors (h, c), each shape [hidden*num_layers]
        h0, c0 = self.network.get_initial_state(batch_size=1, device=next(self.parameters()).device)
        h0f, c0f = _flatten_state(h0, c0)  # [1, HN]
        return [h0f.squeeze(0), c0f.squeeze(0)]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        device = next(self.parameters()).device
        obs = input_dict["obs"].float().to(device)

        if not self._is_recurrent:
            logits, value = self.network(obs)
            self._value_out = value
            return logits, state

        # Deduce B and T from seq_lens / obs
        B = int(seq_lens.shape[0]) if hasattr(seq_lens, "shape") else int(len(seq_lens))
        T = int(obs.shape[0] // B)

        # Build [B, T, ...] obs for our network
        obs_seq = obs.view(B, T, *obs.shape[1:])

        # Dones (if available) for state resets; reshape to [B, T]
        dones = input_dict.get("dones", None)
        if dones is not None:
            dones = dones.to(device=device).view(B, T)

        # Rebuild LSTM state if provided, else zeros
        if state and len(state) == 2 and state[0].numel() > 0:
            h_flat, c_flat = state
            h0, c0 = _unflatten_state(h_flat.to(device), c_flat.to(device), self._num_layers, self._hidden_size)
        else:
            h0, c0 = self.network.get_initial_state(B, device=device)

        logits, value, (hn, cn) = self.network(obs_seq, (h0, c0), dones=dones)

        # Flatten back to [B*T, ...] for RLlib
        if logits.dim() == 3:
            logits = logits.reshape(B * T, -1)
            value  = value.reshape(B * T, 1)

        # Cache value for value_function()
        self._value_out = value

        # Return new flattened states [B, hidden*num_layers]
        hnf, cnf = _flatten_state(hn, cn)
        return logits, [hnf, cnf]

    @override(TorchModelV2)
    def value_function(self):
        # _value_out is [B*T, 1]
        return self._value_out.squeeze(-1)
