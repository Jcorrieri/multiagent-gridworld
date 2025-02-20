import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from torch_geometric.data import Data
from torch_geometric.nn.conv import GATConv, GCNConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, in_channels=27, hidden_dim=128, out_dim=64):
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim=out_dim)

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        g = build_graph(data)

        x, edge_index = g.x, g.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.fc(x)

        return x


def build_graph(observation):
    if len(observation["agents"].shape) == 3:
        observation = {key: observation[key].squeeze(0) for key in observation}

    agent_locations = observation['agents']
    grid_map = observation['map']
    matrix = observation['adj_matrix']

    # Node Features: Create a local view for each agent
    node_features = []
    for loc in agent_locations:
        x, y = loc
        x, y = int(x.item()), int(y.item())
        # Local crop from grid map around the agent
        local_view = grid_map[max(x - 2, 0):x + 3, max(y - 2, 0):y + 3].flatten()
        local_view_tensor = torch.tensor(local_view, dtype=torch.float)

        # Agent-level attributes (can be extended)
        agent_state = torch.tensor([x, y], dtype=torch.float)

        # Combine agent state and local map
        node_feature = torch.cat([agent_state, local_view_tensor])
        node_features.append(node_feature)

    x = torch.stack(node_features)

    edge_index = torch.nonzero(matrix, as_tuple=False).t().contiguous()
    src_nodes, dst_nodes = edge_index
    edge_attr = torch.norm(agent_locations[src_nodes] - agent_locations[dst_nodes], dim=1, p=2).unsqueeze(1)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph