import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, latent_size, num_ch):
        super(AdaIN, self).__init__()
        self.mu_fc = nn.Linear(latent_size, num_ch)
        self.sigma_fc = nn.Linear(latent_size, num_ch)

    def forward(self, x, a):
        x = torch.div(torch.sub(x, torch.mean(x, dim=(2, 3), keepdim=True)), torch.std(x, dim=(2, 3), keepdim=True))
        mu = self.mu_fc(a)
        sigma = self.sigma_fc(a).exp()
        return (x * sigma) + mu