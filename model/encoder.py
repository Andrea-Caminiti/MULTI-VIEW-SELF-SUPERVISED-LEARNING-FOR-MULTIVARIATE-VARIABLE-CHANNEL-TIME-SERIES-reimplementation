import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipConnectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0, stride = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride) 
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        conv_out = self.conv(x)
        if conv_out.shape[-1] != x.shape[-1]:  # Downsample if shape mismatch
            x = F.interpolate(x, size=conv_out.shape[-1], mode='nearest')  
        if conv_out.shape[1] != x.shape[1]:
            x = self.proj(x)
        return x + conv_out  # Skip connection
    
class Encoder(nn.Module):
    def __init__(self, embedding_dim=128):
        
        super(Encoder, self).__init__()

        self.conv_blocks = nn.Sequential(
            SkipConnectionBlock(1, 256, kernel_size=3, stride=3, padding=1),  # T = 3000 -> 1000
            nn.Dropout(0.1),
            nn.GroupNorm(32, 256),
            nn.GELU(),

            SkipConnectionBlock(256, 256, kernel_size=2, stride=2), # T = 1000 -> 500
            nn.Dropout(0.1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            
            SkipConnectionBlock(256, 256, kernel_size=2, stride=2), # T = 500 -> 250
            nn.Dropout(0.1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            
            SkipConnectionBlock(256, 256, kernel_size=2, stride=2), # T = 250 -> 125
            nn.Dropout(0.1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            
            SkipConnectionBlock(256, 256, kernel_size=2, stride=2), # T = 125 -> 62
            nn.Dropout(0.1),
            nn.GroupNorm(32, 256),
            nn.GELU(),

            SkipConnectionBlock(256, embedding_dim, kernel_size=2, stride=2)  # T = 62 -> 31
        )

    def forward(self, x):
        
        return self.conv_blocks(x)

