import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class SharedCNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        h, w, c = obs_space.shape  # HWC
        self.cnn = nn.Sequential(
            nn.ConstantPad2d(1, -1),  # Pad 1 pixel border with -1s (like MATLAB)
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(16 * (h // 4) * (w // 4), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(256, num_outputs)
        self.value_head = nn.Linear(256, 1)

        self._value_out = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        # Permute if obs is NHWC → NCHW
        if x.ndim == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        features = self.cnn(x)
        self._value_out = self.value_head(features).squeeze(1)
        logits = self.policy_head(features)
        return logits, state

    @override(ModelV2)
    def value_function(self):
        return self._value_out
