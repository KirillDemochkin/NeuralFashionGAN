import torch
import torch.nn as nn
from .stylegan_layers import AdaIN


class SPADE(nn.Module):
    def __init__(self, scale_factor, n_filters, mask_channels):
        super(SPADE, self).__init__()
        self.scale_factor = scale_factor
        self.bn = nn.InstanceNorm2d(n_filters)
        self.shared_conv = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(mask_channels, 128, kernel_size=3, padding=1)),
                                         nn.ReLU(inplace=True))
        self.mu_conv = nn.utils.spectral_norm(nn.Conv2d(128, n_filters, kernel_size=3, padding=1))
        self.sigma_conv = nn.utils.spectral_norm(nn.Conv2d(128, n_filters, kernel_size=3, padding=1))

    def forward(self, x, mask):
        mask = nn.functional.interpolate(mask, scale_factor=self.scale_factor)
        conditional_features = self.shared_conv(mask)
        mu = self.mu_conv(conditional_features)
        sigma = self.sigma_conv(conditional_features)
        return (self.bn(x) * sigma) + mu


class SPADE_ResBlock(nn.Module):
    def __init__(self, scale_factor, in_filters, n_filters , mask_channels):
        super(SPADE_ResBlock, self).__init__()
        self.spade_1 = SPADE(scale_factor, in_filters, mask_channels)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_1 = nn.utils.spectral_norm(nn.Conv2d(in_filters, n_filters, kernel_size=3, padding=1), eps=1e-6)
        self.spade_2 = SPADE(scale_factor, n_filters, mask_channels)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.spade_skip = SPADE(scale_factor, in_filters, mask_channels)
        self.relu_skip = nn.ReLU(inplace=True)
        self.conv_skip = nn.utils.spectral_norm(nn.Conv2d(in_filters, n_filters, kernel_size=3, padding=1), eps=1e-6)

    def forward(self, x, mask):
        out = self.conv_1(self.relu_1(self.spade_1(x, mask)))
        out = self.conv_2(self.relu_2(self.spade_2(out, mask)))
        x = self.conv_skip(self.relu_skip(self.spade_skip(x, mask)))
        return out + x


class Style_SPADE(nn.Module):
    def __init__(self, scale_factor, n_filters, mask_channels, latent_dim):
        super(Style_SPADE, self).__init__()
        self.scale_factor = scale_factor
        self.bn = AdaIN(latent_dim, n_filters)
        self.shared_conv = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(mask_channels, 128, kernel_size=3, padding=1), eps=1e-6),
                                         nn.ReLU(inplace=True))
        self.mu_conv = nn.utils.spectral_norm(nn.Conv2d(128, n_filters, kernel_size=3, padding=1), eps=1e-6)
        self.sigma_conv = nn.utils.spectral_norm(nn.Conv2d(128, n_filters, kernel_size=3, padding=1), eps=1e-6)

    def forward(self, x, mask, style_code):
        mask = nn.functional.interpolate(mask, scale_factor=self.scale_factor)
        conditional_features = self.shared_conv(mask)
        mu = self.mu_conv(conditional_features)
        sigma = self.sigma_conv(conditional_features)
        adapted_input = self.bn(x, style_code)
        return (adapted_input * sigma) + mu


class Style_SPADE_ResBlock(nn.Module):
    def __init__(self, scale_factor, in_filters, n_filters , mask_channels, latent_dim):
        super(Style_SPADE_ResBlock, self).__init__()
        self.spade_1 = Style_SPADE(scale_factor, in_filters, mask_channels, latent_dim)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_1 = nn.utils.spectral_norm(nn.Conv2d(in_filters, n_filters, kernel_size=3, padding=1), eps=1e-6)
        self.spade_2 = Style_SPADE(scale_factor, n_filters, mask_channels, latent_dim)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_2 = nn.utils.spectral_norm(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1), eps=1e-6)
        self.spade_skip = Style_SPADE(scale_factor, in_filters, mask_channels, latent_dim)
        self.relu_skip = nn.ReLU(inplace=True)
        self.conv_skip = nn.utils.spectral_norm(nn.Conv2d(in_filters, n_filters, kernel_size=3, padding=1), eps=1e-6)

    def forward(self, x, mask, style_code):
        out = self.conv_1(self.relu_1(self.spade_1(x, mask, style_code)))
        out = self.conv_2(self.relu_2(self.spade_2(out, mask, style_code)))
        x = self.conv_skip(self.relu_skip(self.spade_skip(x, mask, style_code)))
        return out + x


class Discriminator_block(nn.Module):
    def __init__(self, in_filters, out_filters, normalization=True):
        super(Discriminator_block, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), eps=1e-6)
        self.norm = nn.InstanceNorm2d(out_filters)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.normalization = normalization

    def forward(self, x):
        x = self.conv(x)
        if self.normalization:
            x = self.norm(x)
        x = self.leaky(x)
        return x