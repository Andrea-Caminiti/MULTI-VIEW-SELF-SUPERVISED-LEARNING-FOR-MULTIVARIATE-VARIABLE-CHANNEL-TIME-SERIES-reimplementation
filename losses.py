import torch
import torch.nn.functional as F

class TS2VecLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(TS2VecLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_t1, z_t2):
        """
        Args:
            z_t1, z_t2: (batch_size, time_steps, embedding_dim) - two views of the time series.
        
        Returns:
            TS2Vec loss scalar.
        """
        B, C, D, T = z_t1.shape
        # Normalize embeddings
        z_t1 = F.normalize(z_t1, dim=2)  # (B, C, D, T)
        z_t2 = F.normalize(z_t2, dim=2)  # (B, C, D, T)

        ### 1. Temporal Consistency Loss
        # Compute dot product similarity for each channel independently
        sim_temporal = torch.einsum('bcdt,bcdt->bct', z_t1, z_t2)  # Shape: (B, C, T)

        # Temporal contrastive loss
        loss_temporal = -torch.log(
            torch.exp(sim_temporal / self.temperature) /
            torch.exp(sim_temporal / self.temperature).sum(dim=-1, keepdim=True)
        ).mean()

        ### 2. Instance Consistency Loss
        # Flatten across channels for batch-wise similarity
        z_t1_flat = z_t1.permute(0, 3, 1, 2).reshape(B, T, C * D)  # Shape: (B, T, C*D)
        z_t2_flat = z_t2.permute(0, 3, 1, 2).reshape(B, T, C * D)  # Shape: (B, T, C*D)

        # Compute similarity between all instances (cross-batch, same time step)
        sim_instance = torch.einsum('btd,jtd->btj', z_t1_flat, z_t2_flat)  # Shape: (B, T, B)

        # Mask to avoid self-similarity (i == j)
        mask = ~torch.eye(B, dtype=torch.bool, device=z_t1.device).unsqueeze(1)  # Shape: (B, 1, B)
        sim_instance = sim_instance.masked_select(mask).view(B, T, B - 1)  # Shape: (B, T, B-1)

        # Instance contrastive loss
        loss_instance = -torch.log(
            torch.exp(sim_instance / self.temperature) /
            torch.exp(sim_instance / self.temperature).sum(dim=-1, keepdim=True)
        ).mean()

        # Combine both losses
        loss = (loss_temporal + loss_instance) / 2

        return loss