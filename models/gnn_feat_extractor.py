from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim)

        # Number of agents from observation space
        self.num_agents = observation_space.spaces["agents"].shape[0]
        self.map_size = observation_space.spaces["map"].shape[0]

        # Encoder for robot positions (x, y coordinates)
        self.robot_encoder = nn.Linear(2, 32)  # Position features - increased dimension

        # Map processing with CNN
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.ReLU(),
        )

        # Global map feature processing
        self.global_map_features = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Graph Convolutional Layers (taking position + local map features)
        self.conv1 = GCNConv(32 + 32, 64)  # Position + local map features
        self.conv2 = GCNConv(64, 64)

        # Final output layer (combines node features with global map features)
        self.final_layer = nn.Linear(self.num_agents * 64 + 32, features_dim)

        # Final output dimension
        self._features_dim = features_dim

    def forward(self, obs_dict):
        # Extract inputs
        agents_pos = obs_dict["agents"].float()  # [batch_size, num_agents, 2]
        adj_matrix = obs_dict["adj_matrix"].float()  # [batch_size, num_agents, num_agents]
        map_data = obs_dict["map"].float()  # [batch_size, map_size, map_size]

        # Add batch dimension if missing
        if agents_pos.dim() == 2:
            agents_pos = agents_pos.unsqueeze(0)
        if adj_matrix.dim() == 2:
            adj_matrix = adj_matrix.unsqueeze(0)
        if map_data.dim() == 2:
            map_data = map_data.unsqueeze(0)

        batch_size = agents_pos.shape[0]

        # Process map with CNN
        map_input = map_data.unsqueeze(1)  # [batch_size, 1, map_size, map_size]
        map_features = self.map_encoder(map_input)  # [batch_size, 32, map_size, map_size]

        # Global map features
        global_map_feature = map_features.mean(dim=[2, 3])  # [batch_size, 32]
        global_map_feature = self.global_map_features(global_map_feature)  # [batch_size, 32]

        # Encode robot positions
        pos_features = self.robot_encoder(agents_pos)  # [batch_size, num_agents, 32]

        # Extract local map features for each robot
        x_idx = agents_pos[:, :, 0].long()
        y_idx = agents_pos[:, :, 1].long()
        local_map_features = map_features[torch.arange(batch_size)[:, None], :, x_idx, y_idx]

        # Combine position and local map features
        node_features = torch.cat([pos_features, local_map_features], dim=2)  # [batch_size, num_agents, 32+32]
        node_features = node_features.view(batch_size * self.num_agents, -1)  # [batch_size * num_agents, 64]

        # Assuming adj_matrix has shape [batch_size, num_agents, num_agents]
        batch_size, num_agents, _ = adj_matrix.shape

        # Process each batch separately
        all_edges = []
        for b in range(batch_size):
            # Get adjacency matrix for this batch
            batch_adj = adj_matrix[b]

            # Convert to sparse format
            batch_edges, _ = dense_to_sparse(batch_adj)

            # Offset the indices to account for batch position
            batch_offset = b * num_agents
            batch_edges = batch_edges + batch_offset

            # Add to our collection
            all_edges.append(batch_edges)

        # Combine all edges into a single edge_index tensor
        # This creates a block-diagonal structure
        edge_index = torch.cat(all_edges, dim=1)

        # Apply GNN layers
        x = F.relu(self.conv1(node_features, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Reshape to [batch_size, num_agents, 64]
        x = x.view(batch_size, self.num_agents, 64)

        # Flatten node features and combine with global map features
        x_flat = x.reshape(batch_size, -1)  # [batch_size, num_agents * 64]
        combined = torch.cat([x_flat, global_map_feature], dim=1)  # [batch_size, num_agents * 64 + 32]

        # Final output
        return self.final_layer(combined)  # [batch_size, features_dim]