import torch
import torch.nn as nn

class DQNModel(nn.Module):
    def __init__(self, obs_space, num_outputs, num_agents=1):
        super().__init__()
        
        h, w, c = 25, 25, 4
        self.num_agents = num_agents
        self.num_outputs = num_outputs

        # Shared CNN layer
        self.shared_conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.shared_conv(dummy)
            flattened_size = out.flatten(1).size(1)

        self.shared_fc = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Individual advantage heads for each agent
        self.individual_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, self.num_outputs)
            ) for _ in range(self.num_agents)
        ])

        # Centralized value head
        self.value_head = nn.Sequential(
            nn.Linear(512, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def _to_nchw(self, x):
        # RLlib usually passes [B,H,W,C]; convert to [B,C,H,W]
        return x.permute(0, 3, 1, 2) if x.dim() == 4 else x

    def forward(self, obs, agent_id):   
        if isinstance(obs, dict):
            x = self._to_nchw(obs["critic"]) # here, full obs is expected
        else:
            x = self._to_nchw(obs)
        
        features = self.shared_fc(self.shared_conv(x)) # [Batch, 512]
        
        all_advantages = torch.stack([head(features) for head in self.individual_heads], dim=1) 
                
        batch_indices = torch.arange(agent_id.size(0), device=agent_id.device)
        advantage = all_advantages[batch_indices, agent_id] # [Batch, Num_Outputs]

        state_value = self.value_head(features)
        q_values = state_value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values, state_value