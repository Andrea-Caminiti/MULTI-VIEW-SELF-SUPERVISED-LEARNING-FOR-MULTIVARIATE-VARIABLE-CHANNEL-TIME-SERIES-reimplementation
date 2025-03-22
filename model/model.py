import torch
import torch.nn as nn
from model.encoder import Encoder
from model.mpnn import MPNN_CONV, MPNN_GAT

class Model(nn.Module):
    def __init__(self, orig_channels = 6, time_l = 31, embedding_dim=64, hidden_dim=64, num_layers=3, attention=False, attention_heads=None, classes=5, view_strat = 'split', finetune = False):
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
            
        self.embedding_dim = embedding_dim
        self.view_strat = view_strat
        self.finetune = finetune
        
        self.flatten = nn.Flatten()
        self.e_dim = embedding_dim * time_l
        self.classifier = nn.Linear(self.e_dim, classes)
        
        if view_strat == 'split':
            self.channelreduction = nn.Linear(in_features=orig_channels, out_features=1)
        else:
            self.channelreduction = nn.Linear(2, 1)
            
        
    def forward(self, x):
        pass
    
    def encode(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_channels, time_steps)

        Returns:
            Tensor of shape (batch_size, embedding_dim) - The final learned representation
        """
        num_channels = x.shape[1]
        channel_embeddings = torch.stack([self.encoder(x[:, i, :].unsqueeze(1)) for i in range(num_channels)], dim=1).squeeze() # shape: (num_channels, embedding_dim, reduced_time_steps)

        return channel_embeddings
    
    def message(self, x): # Shape: (batch_size, num_channels, embedding_dim)
        res = self.mpnn(x)  # Shape: (batch_size, embedding_dim, )

        return res
    
    
    def classify(self, x):
        
        res = self.channelreduction(x)
        res = self.flatten(res)
        res = self.classifier(res)
        return res
    
    def update_classifier(self, channels, classes):
        if self.finetune:
            self.channelreduction = nn.Linear(in_features=channels, out_features=1)
            self.classifier = nn.Linear(self.e_dim, classes)
        else:
            self.classifier = nn.Linear(self.embedding_dim, classes)
        

    def freeze_encoder(self):
        for par in self.encoder.parameters():
            par.requires_grad = False
    def freeze_mpnn(self):
        for par in self.mpnn.parameters():
            par.requires_grad = False
    def freeze_classifier(self):
        for par in self.classifier.parameters():
            par.requires_grad = False
