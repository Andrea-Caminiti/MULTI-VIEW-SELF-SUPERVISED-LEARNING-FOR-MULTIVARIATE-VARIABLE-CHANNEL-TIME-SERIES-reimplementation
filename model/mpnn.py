import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

class MPNN_GAT(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=64, num_heads=4, num_layers=3):
        """
        Args:
            embedding_dim: Feature size of each node.
            hidden_dim: Hidden size of the MPNN layers.
            num_heads: Number of attention heads in GAT.
            num_layers: Number of GAT layers.
        """
        super(MPNN_GAT, self).__init__()
        self.convs = nn.ModuleList([
            GATConv(embedding_dim, hidden_dim, heads=num_heads, concat=False) for _ in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, embeddings):
        """
        Args:
            embeddings: Tensor of shape (batch_size, num_channels, embedding_dim)
        
        Returns:
            Tensor of shape (batch_size, embedding_dim) - Combined representation
        """
        batch_size, num_channels, _ = embeddings.shape
        device = embeddings.device

       
        edge_index = torch.combinations(torch.arange(num_channels, device=device), r=2).T  # Pairs
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make bidirectional

        
        combined_outputs = []
        for i in range(batch_size):
            x = embeddings[i]  # shape: (num_channels, embedding_dim)

            for conv in self.convs:
                x = conv(x, edge_index)
                x = torch.relu(x) 

            
            combined_outputs.append(x.mean(dim=0))

        return self.readout(torch.stack(combined_outputs))


class MPNN_CONV(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=64, num_layers=3):
        """
        Args:
            embedding_dim: Feature size of each node.
            hidden_dim: Hidden size of the MPNN layers.
            num_layers: Number of message passing steps.
        """
        super(MPNN_CONV, self).__init__()
        self.convs = nn.ModuleList([GCNConv(embedding_dim, hidden_dim) for _ in range(num_layers)])
        self.readout = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, embeddings):
        """
        Args:
            embeddings: Tensor of shape (batch_size, num_channels, embedding_dim)
        
        Returns:
            Tensor of shape (batch_size, embedding_dim) - Combined representation
        """
        batch_size, num_channels, _ = embeddings.shape
        device = embeddings.device
        
        
        edge_index = torch.combinations(torch.arange(num_channels, device=device), r=2).T  # Pairs
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make bidirectional

        
        combined_outputs = []
        for i in range(batch_size):
            x = embeddings[i]  # shape: (num_channels, embedding_dim)
            for conv in self.convs:
                x = conv(x, edge_index)
                x = torch.relu(x)
            
            
            combined_outputs.append(x.mean(dim=0))

        return self.readout(torch.stack(combined_outputs))
