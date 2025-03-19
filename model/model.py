import torch
import torch.nn as nn
from model.encoder import Encoder
from model.mpnn import MPNN_CONV, MPNN_GAT

class Model(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=64, num_layers=3, attention=False, attention_heads=None, classes=5):
        """
        Args:
            embedding_dim: Size of the feature representations.
            hidden_dim: Hidden dimension for the MPNN.
            num_message_passing_steps: Number of iterations in message passing.
        """

        assert (not attention and not attention_heads) or (attention and attention_heads)

        super(Model, self).__init__()

        # Single-channel encoder
        self.encoder = Encoder(embedding_dim=embedding_dim)
        
        # MPNN for combining multiple channels
        if attention: 
            self.mpnn = MPNN_GAT(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_heads=attention_heads, num_layers=num_layers)
        else: 
            self.mpnn = MPNN_CONV(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        self.classifier = nn.Linear(embedding_dim, classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_channels, time_steps)

        Returns:
            Tensor of shape (batch_size, embedding_dim) - The final learned representation
        """
        num_channels = x.shape[1]
        
        channel_embeddings = torch.stack([self.encoder(x[:, i, :].unsqueeze(1)) for i in range(num_channels)], dim=1) # shape: (batch_size, num_channels, embedding_dim, reduced_time_steps)

        channel_embeddings = channel_embeddings.mean(dim=-1)  # Shape: (batch_size, num_channels, embedding_dim)

        res = self.mpnn(channel_embeddings)  # Shape: (batch_size, embedding_dim)

        return res
    
    def classify(self, x):
        res = self.classifier(x)
        res = torch.argmax(res, 1)
        return res

    def freeze_encoder(self):
        for par in self.encoder.parameters():
            par.requires_grad = False
    def freeze_mpnn(self):
        for par in self.mpnn.parameters():
            par.requires_grad = False
    def freeze_classifier(self):
        for par in self.classifier.parameters():
            par.requires_grad = False
