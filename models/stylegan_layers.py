import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, latent_size, num_ch):
        super(AdaIN, self).__init__()
        self.mu_fc = nn.Linear(latent_size, num_ch)
        self.sigma_fc = nn.Linear(latent_size, num_ch)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, a, noise=None):
        if noise is not None:
            x = x + noise
        x = self.leaky(x)
        mean = torch.mean(x, dim=(2, 3), keepdim=True).detach()
        std = (torch.var(x.view(x.shape[0], x.shape[1], -1), dim=-1, keepdim=True) + 1e-8).sqrt().unsqueeze(-1).detach()
        x = (x - mean) / (std + 1e-8)
        mu = self.mu_fc(a).unsqueeze(2).unsqueeze(3)
        sigma = self.sigma_fc(a).unsqueeze(2).unsqueeze(3)
        return x * (1 + sigma) + mu


class StylizationNoiseNetwork(nn.Module):
    def __init__(self, num_channels, device):
        super(StylizationNoiseNetwork, self).__init__()
        self.B1 = nn.Parameter(data=torch.zeros(1, num_channels[0], 1, 1, device=device).normal_(0, 1), requires_grad=True)
        self.B2 = nn.Parameter(data=torch.empty(1, num_channels[1], 1, 1, device=device).normal_(0, 1), requires_grad=True)
        self.B3 = nn.Parameter(data=torch.empty(1, num_channels[2], 1, 1, device=device).normal_(0, 1), requires_grad=True)
        self.B4 = nn.Parameter(data=torch.empty(1, num_channels[3], 1, 1, device=device).normal_(0, 1), requires_grad=True)
        self.B5 = nn.Parameter(data=torch.empty(1, num_channels[4], 1, 1, device=device).normal_(0, 1), requires_grad=True)
        self.B6 = nn.Parameter(data=torch.empty(1, num_channels[5], 1, 1, device=device).normal_(0, 1), requires_grad=True)


    def forward(self, x):
        scaled_noise_1 = torch.mul(x[0], self.B1)
        scaled_noise_2 = torch.mul(x[1], self.B2)
        scaled_noise_3 = torch.mul(x[2], self.B3)
        scaled_noise_4 = torch.mul(x[3], self.B4)
        scaled_noise_5 = torch.mul(x[4], self.B5)
        scaled_noise_6 = torch.mul(x[5], self.B6)
        return scaled_noise_1, scaled_noise_2, scaled_noise_3, scaled_noise_4, scaled_noise_5, scaled_noise_6
