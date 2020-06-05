import torch
import torch.nn as nn

from .gaugan_layers import SPADE_ResBlock, Style_SPADE_ResBlock
from .encoders import MappingNetwork, StyleEncoder
from .stylegan_layers import StylizationNoiseNetwork

class GauGANGenerator(nn.Module):
    def __init__(self, mask_channels, latent_dim, initial_image_size):
        super(GauGANGenerator, self).__init__()
        self.linear = nn.Linear(latent_dim, initial_image_size*initial_image_size*latent_dim*4)
        self.initial_image_size = initial_image_size
        self.latent_dim = latent_dim

        self.spd_blck_1 = SPADE_ResBlock(1/2 ** 4, latent_dim*4, latent_dim*4, mask_channels)
        self.upsample_1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_2 = SPADE_ResBlock(1 / 2 ** 3, latent_dim * 4, latent_dim * 4, mask_channels)
        self.upsample_2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_3 = SPADE_ResBlock(1 / 2 ** 2, latent_dim * 4, latent_dim * 4, mask_channels)
        self.upsample_3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_4 = SPADE_ResBlock(1 / 2 ** 1, latent_dim * 4, latent_dim * 2, mask_channels)
        self.upsample_4 = nn.UpsamplingNearest2d(scale_factor=2)

        #self.spd_blck_5 = SPADE_ResBlock(1 / 2 ** 2, latent_dim * 2, latent_dim, mask_channels)
        #self.upsample_5 = nn.UpsamplingNearest2d(scale_factor=2)

        #self.spd_blck_6 = SPADE_ResBlock(1 / 2, latent_dim, latent_dim // 2, mask_channels)
        #self.upsample_6 = nn.UpsamplingNearest2d(scale_factor=2)

        # self.spd_blck_7 = SPADE_ResBlock(1, latent_dim // 2, latent_dim // 4, mask_channels)
        # self.upsample_7 = nn.UpsamplingNearest2d(scale_factor=2)

        #self.out_conv = nn.Conv2d(latent_dim // 2, 3, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(latent_dim *2, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        x = self.linear(x)
        x = torch.reshape(x, (-1, self.latent_dim*4, self.initial_image_size, self.initial_image_size))
        x = self.upsample_1(self.spd_blck_1(x, mask))
        x = self.upsample_2(self.spd_blck_2(x, mask))
        x = self.upsample_3(self.spd_blck_3(x, mask))
        x = self.upsample_4(self.spd_blck_4(x, mask))
        #x = self.upsample_5(self.spd_blck_5(x, mask))
        #x = self.upsample_6(self.spd_blck_6(x, mask))
        # x = self.upsample_7(self.spd_blck_7(x, mask))
        x = self.tanh(self.out_conv(x))
        return x

class GauGANUnetGenerator(nn.Module):
    def __init__(self, mask_channels, latent_dim, initial_image_size, skip_dim):
        super(GauGANUnetGenerator, self).__init__()
        self.linear = nn.Linear(latent_dim, initial_image_size*initial_image_size*latent_dim*4)
        self.initial_image_size = initial_image_size
        self.latent_dim = latent_dim

        self.spd_blck_1 = SPADE_ResBlock(1 / 2 ** 4, latent_dim * 4 + skip_dim, latent_dim * 4, mask_channels)
        self.upsample_1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_2 = SPADE_ResBlock(1 / 2 ** 3, latent_dim * 4 + skip_dim, latent_dim * 4, mask_channels)
        self.upsample_2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_3 = SPADE_ResBlock(1 / 2 ** 2, latent_dim * 4 + skip_dim, latent_dim * 4, mask_channels)
        self.upsample_3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_4 = SPADE_ResBlock(1 / 2 ** 1, latent_dim * 4 + skip_dim, latent_dim * 2, mask_channels)
        self.upsample_4 = nn.UpsamplingNearest2d(scale_factor=2)

        #self.spd_blck_5 = SPADE_ResBlock(1 / 2 ** 2, latent_dim * 2, latent_dim, mask_channels)
        #self.upsample_5 = nn.UpsamplingNearest2d(scale_factor=2)

        #self.spd_blck_6 = SPADE_ResBlock(1 / 2, latent_dim, latent_dim // 2, mask_channels)
        #self.upsample_6 = nn.UpsamplingNearest2d(scale_factor=2)

        # self.spd_blck_7 = SPADE_ResBlock(1, latent_dim // 2, latent_dim // 4, mask_channels)
        # self.upsample_7 = nn.UpsamplingNearest2d(scale_factor=2)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        #self.out_conv = nn.Conv2d(latent_dim // 2, 3, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(latent_dim *2, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, mask, skips):
        x = self.linear(x)
        x = torch.reshape(x, (-1, self.latent_dim*4, self.initial_image_size, self.initial_image_size))
        x = self.upsample_1(self.spd_blck_1(torch.cat((x, skips[3]), dim=1), mask))
        x = self.upsample_2(self.spd_blck_2(torch.cat((x, skips[2]), dim=1), mask))
        x = self.upsample_3(self.spd_blck_3(torch.cat((x, skips[1]), dim=1), mask))
        x = self.upsample_4(self.spd_blck_4(torch.cat((x, skips[0]), dim=1), mask))
        #x = self.upsample_5(self.spd_blck_5(x, mask))
        #x = self.upsample_6(self.spd_blck_6(x, mask))
        # x = self.upsample_7(self.spd_blck_7(x, mask))
        x = self.tanh(self.out_conv(self.leaky(x)))
        return x


class GauGANUnetStylizationGenerator(nn.Module):
    def __init__(self, mask_channels, latent_dim, initial_image_size, skip_dim, device):
        super(GauGANUnetStylizationGenerator, self).__init__()
        self.noise_net = StylizationNoiseNetwork([latent_dim * 4 + skip_dim,
                                                  latent_dim * 4 + skip_dim,
                                                  latent_dim * 4 + skip_dim,
                                                  latent_dim * 4 + skip_dim,
                                                  latent_dim * 2 + skip_dim,
                                                  latent_dim + skip_dim
                                                  ], device)
        self.linear = nn.Linear(latent_dim, initial_image_size*initial_image_size*latent_dim*4)
        self.initial_image_size = initial_image_size
        self.latent_dim = latent_dim
        self.skip_dim = skip_dim
        self.device = device
        self.starting_noise = nn.Parameter(data=torch.zeros(1, latent_dim, device=device).normal_(0, 0.02), requires_grad=True)

        self.spd_blck_1 = Style_SPADE_ResBlock(1 / 2 ** 6, latent_dim * 4 + skip_dim, latent_dim * 4, mask_channels, latent_dim)
        self.upsample_1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_2 = Style_SPADE_ResBlock(1 / 2 ** 5, latent_dim * 4 + skip_dim, latent_dim * 4, mask_channels, latent_dim)
        self.upsample_2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_3 = Style_SPADE_ResBlock(1 / 2 ** 4, latent_dim * 4 + skip_dim, latent_dim * 4, mask_channels, latent_dim)
        self.upsample_3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_4 = Style_SPADE_ResBlock(1 / 2 ** 3, latent_dim * 4 + skip_dim, latent_dim * 2, mask_channels, latent_dim)
        self.upsample_4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_5 = Style_SPADE_ResBlock(1 / 2 ** 2, latent_dim * 2 + skip_dim, latent_dim, mask_channels, latent_dim)
        self.upsample_5 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_6 = Style_SPADE_ResBlock(1 / 2, latent_dim + skip_dim, latent_dim // 2, mask_channels, latent_dim)
        self.upsample_6 = nn.UpsamplingNearest2d(scale_factor=2)

        # self.spd_blck_7 = SPADE_ResBlock(1, latent_dim // 2, latent_dim // 4, mask_channels)
        # self.upsample_7 = nn.UpsamplingNearest2d(scale_factor=2)
        self.leaky_pre = nn.LeakyReLU(0.2, inplace=True)
        self.out_conv = nn.Conv2d(latent_dim // 2, latent_dim//2, kernel_size=3, padding=1)
        self.leaky_post = nn.LeakyReLU(0.2, inplace=True)
        self.instance_pre = nn.InstanceNorm2d(latent_dim//2)
        self.to_rgb = nn.Conv2d(latent_dim//2, 3, kernel_size=1)
        # self.out_conv = nn.Conv2d(latent_dim *2, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, style_code, mask, skips):
        style_noise = self.noise_net([torch.randn(style_code.shape[0], 1, self.initial_image_size, self.initial_image_size, device=self.device),
                                      torch.randn(style_code.shape[0], 1, self.initial_image_size*2, self.initial_image_size*2, device=self.device),
                                      torch.randn(style_code.shape[0], 1, self.initial_image_size*4, self.initial_image_size*4, device=self.device),
                                      torch.randn(style_code.shape[0], 1, self.initial_image_size*8, self.initial_image_size*8, device=self.device),
                                      torch.randn(style_code.shape[0], 1, self.initial_image_size*16, self.initial_image_size*16, device=self.device),
                                      torch.randn(style_code.shape[0], 1, self.initial_image_size*32, self.initial_image_size*32, device=self.device)])

        x = self.starting_noise.repeat(style_code.shape[0], 1)
        if self.training:
            x = self.linear(x+torch.zeros_like(x).uniform_(-0.05, 0.05))
        else:
            x = self.linear(x)
        #x = self.linear(x)
        x = torch.reshape(x, (-1, self.latent_dim*4, self.initial_image_size, self.initial_image_size))
        x = self.upsample_1(self.spd_blck_1(torch.cat((x, skips[5]), dim=1), mask, style_code, style_noise[0]))
        x = self.upsample_2(self.spd_blck_2(torch.cat((x, skips[4]), dim=1), mask, style_code, style_noise[1]))
        x = self.upsample_3(self.spd_blck_3(torch.cat((x, skips[3]), dim=1), mask, style_code, style_noise[2]))
        x = self.upsample_4(self.spd_blck_4(torch.cat((x, skips[2]), dim=1), mask, style_code, style_noise[3]))
        x = self.upsample_5(self.spd_blck_5(torch.cat((x, skips[1]), dim=1), mask, style_code, style_noise[4]))
        x = self.upsample_6(self.spd_blck_6(torch.cat((x, skips[0]), dim=1), mask, style_code, style_noise[5]))
        # x = self.upsample_7(self.spd_blck_7(x, mask))
        x = self.tanh(self.to_rgb(self.instance_pre(self.leaky_post(self.out_conv(self.leaky_pre(x))))))
        return x

