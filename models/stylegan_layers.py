import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, latent_size, num_ch):
        super(AdaIN, self).__init__()
        self.mu_fc = nn.Linear(latent_size, num_ch)
        self.sigma_fc = nn.Linear(latent_size, num_ch)

    def forward(self, x, a):
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        std = (torch.var(x.view(x.shape[0], x.shape[1], -1), dim=-1, keepdim=True) + 1e-6).sqrt().unsqueeze(-1)
        x = (x - mean) / (std + 1e-6)
        mu = self.mu_fc(a).unsqueeze(2).unsqueeze(3)
        sigma = self.sigma_fc(a).unsqueeze(2).unsqueeze(3)
        return x * (1 + sigma) + mu


class StylizationNoiseNetwork(nn.Module):
    def __init__(self, num_channels, device):
        super(StylizationNoiseNetwork, self).__init__()

        self.B1 = torch.empty(1, num_channels[0], 1, 1, device=device).normal_(0, 0.1)
        self.B1.requires_grad_(True)

        self.B2 = torch.empty(1, num_channels[1], 1, 1, device=device).normal_(0, 0.1)
        self.B2.requires_grad_(True)

        self.B3 = torch.empty(1, num_channels[2], 1, 1, device=device).normal_(0, 0.1)
        self.B3.requires_grad_(True)

        self.B4 = torch.empty(1, num_channels[3], 1, 1, device=device).normal_(0, 0.1)
        self.B4.requires_grad_(True)

        self.B5 = torch.empty(1, num_channels[4], 1, 1, device=device).normal_(0, 0.1)
        self.B5.requires_grad_(True)

        self.B6 = torch.empty(1, num_channels[5], 1, 1, device=device).normal_(0, 0.1)
        self.B6.requires_grad_(True)


    def forward(self, x):
        scaled_noise_1 = torch.mul(x[0], self.B1)
        scaled_noise_2 = torch.mul(x[1], self.B2)
        scaled_noise_3 = torch.mul(x[2], self.B3)
        scaled_noise_4 = torch.mul(x[3], self.B4)
        scaled_noise_5 = torch.mul(x[4], self.B5)
        scaled_noise_6 = torch.mul(x[5], self.B6)
        return scaled_noise_1, scaled_noise_2, scaled_noise_3, scaled_noise_4, scaled_noise_5, scaled_noise_6
