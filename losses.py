import torch
import torch.nn.functional as F
    
class TS2VecLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(TS2VecLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_t1, z_t2):
        B, D, T = z_t1.shape
        z_t1 = F.normalize(z_t1, dim=1)  
        z_t2 = F.normalize(z_t2, dim=1)  
        
        # Temporal contrastive loss
        sim_temporal = torch.einsum('bdt,bdt->bt', z_t1, z_t2)
        loss_temporal = -torch.log_softmax(sim_temporal / self.temperature, dim=-1).mean() 

        # Instance contrastive loss        
        sim_instance = torch.einsum('bdt,jdt->btj', z_t1, z_t2)  

        # Mask to avoid self-similarity (i == j)
        mask = torch.ones(B, B, device=z_t1.device).bool()
        mask.fill_diagonal_(False)
        mask = mask.unsqueeze(1)
        sim_instance = sim_instance.masked_select(mask).view(B, T, B - 1) 
        loss_instance = -torch.log_softmax(sim_instance / self.temperature, dim=-1).mean()

        # Combine both losses
        loss = (loss_temporal + loss_instance) / 2
        return loss