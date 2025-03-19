import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, embedding_dim=64):
        """
        Args:
            embedding_dim: Output embedding size.
        """
        super(Encoder, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=3, stride=3, padding=1),  # 3000 -> 1000
            nn.Dropout(0.1),
            nn.GroupNorm(32, 256),
            nn.GELU(),

            nn.Conv1d(256, embedding_dim, kernel_size=2, stride=2)  # 1000 -> 500
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 1, time_steps)

        Returns:
            Tensor of shape (batch_size, embedding_dim, reduced_time_steps)
        """
        return self.conv_blocks(x)

