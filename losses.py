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
        
        # Normalize embeddings
        z_t1 = F.normalize(z_t1, dim=-1)
        z_t2 = F.normalize(z_t2, dim=-1)

        # Compute similarity for temporal and instance losses
        sim_temporal = torch.einsum('btd,btd->bt', z_t1, z_t2)  # Per time step
        sim_instance = torch.einsum('btd,bjd->btj', z_t1, z_t2)  # Across batch

        # Compute loss
        loss_temporal = -torch.log(torch.exp(sim_temporal / self.temperature) / torch.sum(torch.exp(sim_temporal / self.temperature), dim=-1, keepdim=True))
        loss_instance = -torch.log(torch.exp(sim_instance / self.temperature) / torch.sum(torch.exp(sim_instance / self.temperature), dim=-1, keepdim=True))

        return (loss_temporal.mean() + loss_instance.mean()) / 2