import kan
import torch
import torch.nn as nn 

class CombinedKan(nn.Module):
    def __init__(self, kan_hyps, grid, k, seed, grid_range, device, noise=True):
        super(CombinedKan, self).__init__()
        self.device = device
        self.actor_c1 = kan.KAN(width=kan_hyps, grid=grid, k=k, seed=seed, grid_range=grid_range, device=device)
        self.actor_c2 = kan.KAN(width=kan_hyps, grid=grid, k=k, seed=seed, grid_range=grid_range, device=device)
        self.noise = noise
    

    def forward(self, x, noise_data=False):
        u_1 = self.actor_c1(x)
        
        y_2 = 0
        if self.noise_data:
            y_2 = u_1 + self.noise_data
        else:
            y_2 = u_1 + self.noise*torch.normal(0, 1, x.size(), device=self.device)
        
        u_2 = self.actor_c2(y_2)
        return u_1, y_2, u_2


