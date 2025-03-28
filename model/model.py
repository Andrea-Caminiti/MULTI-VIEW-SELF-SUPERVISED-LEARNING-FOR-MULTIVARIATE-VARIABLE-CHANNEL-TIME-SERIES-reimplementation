import torch
import torch.nn as nn
from model.encoder import Encoder
from model.mpnn import MPNN_CONV

class Model(nn.Module):
    def __init__(self, orig_channels = 6, time_l = 31, embedding_dim=128, hidden_dim=128, num_layers=1, classes=5, view_strat = 'split', finetune = False):
        
        super(Model, self).__init__()

        self.encoder = Encoder(embedding_dim=embedding_dim)
        
        self.mpnn = MPNN_CONV(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers)
            
        self.embedding_dim = embedding_dim
        self.view_strat = view_strat
        self.finetune = finetune
        self.flatten = nn.Flatten()
        
        if view_strat == 'split':
            self.channelreduction = nn.Linear(in_features=orig_channels, out_features=1)
            self.e_dim = embedding_dim * time_l
            self.classifier = nn.Linear(self.e_dim, classes)
        else:             
            self.proj = nn.Linear(time_l, 1)
            self.channelreduction = nn.Linear(2, 1)
            self.classifier = nn.Linear(embedding_dim, classes)
        
    def forward(self, x):
        pass
    
    def encode(self, x):
        channel_embeddings = self.encoder(x) # shape: (batch_size, embedding_dim, reduced_time_steps)

        return channel_embeddings
    
    def message(self, x, b, ch): 
        res = self.mpnn(x, b, ch) # Shape: (reduced_time_steps, batch_size*num_channels, embedding_dim)

        return res
    

    def classify(self, x):
        res = self.channelreduction(x).squeeze() # Shape: (batch_size, embedding_dim, reduced_time_steps)
        if self.view_strat == 'split':
            res = self.flatten(res) # Shape: (batch_size, embedding_dim*reduced_time_steps)
        else:
            res = self.proj(res).squeeze() # Shape: (batch_size, embedding_dim)
        res = self.classifier(res) # Shape: (batch_size, classes)
        return res
    
    def update_classifier(self, channels, classes):
        if self.view_strat == 'split':
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
