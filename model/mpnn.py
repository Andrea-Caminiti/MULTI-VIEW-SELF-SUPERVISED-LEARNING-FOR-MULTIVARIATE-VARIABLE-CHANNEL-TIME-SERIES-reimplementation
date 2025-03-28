import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class MPNN_CONV(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=128, num_layers=1):
       
        super(MPNN_CONV, self).__init__()
        self.convs = nn.ModuleList([GCNConv(embedding_dim, hidden_dim) for _ in range(num_layers)])
        self.readout = nn.Linear(hidden_dim, embedding_dim)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, embeddings, b, ch):
        device = embeddings.device
        # --------------- Taken from the repo of the paper --------------------------
        message_from = torch.arange(b*ch).unsqueeze(1).repeat(1, (ch-1)).view(-1)
        message_to = torch.arange(b*ch).view(b, ch).unsqueeze(1).repeat(1, ch, 1)
        idx = ~torch.eye(ch).view(1, ch, ch).repeat(b, 1, 1).bool()
        message_to = message_to[idx].view(-1)
        # -----------------------------------------------------------------------------
        edge_index = torch.stack((message_from, message_to)).to(device)
        
        for conv in self.convs:
            embeddings = conv(embeddings, edge_index)
            embeddings = self.activation(embeddings)
        return embeddings
    
    
    def rout(self, x):
        res = self.readout(x) # Shape: (batch_size, num_channels, reduced_time_steps_ embedding_dim)
        return res
