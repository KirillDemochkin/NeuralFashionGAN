import torch
import torch.nn as nn

from .gaugan_layers import SPADE_ResBlock


class GauGANGenerator(nn.Module):
    def __init__(self, mask_channels, latent_dim, initial_image_size):
        super(GauGANGenerator, self).__init__()
        self.linear = nn.Linear(latent_dim, initial_image_size*initial_image_size*latent_dim*4)
        self.initial_image_size = initial_image_size
        self.latent_dim = latent_dim

        self.spd_blck_1 = SPADE_ResBlock(1/2 ** 6, latent_dim*4, latent_dim*4, mask_channels)
        self.upsample_1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_2 = SPADE_ResBlock(1 / 2 ** 5, latent_dim * 4, latent_dim * 4, mask_channels)
        self.upsample_2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_3 = SPADE_ResBlock(1 / 2 ** 4, latent_dim * 4, latent_dim * 4, mask_channels)
        self.upsample_3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_4 = SPADE_ResBlock(1 / 2 ** 3, latent_dim * 4, latent_dim * 2, mask_channels)
        self.upsample_4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_5 = SPADE_ResBlock(1 / 2 ** 2, latent_dim * 2, latent_dim, mask_channels)
        self.upsample_5 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_6 = SPADE_ResBlock(1 / 2, latent_dim, latent_dim // 2, mask_channels)
        self.upsample_6 = nn.UpsamplingNearest2d(scale_factor=2)

        # self.spd_blck_7 = SPADE_ResBlock(1, latent_dim // 2, latent_dim // 4, mask_channels)
        # self.upsample_7 = nn.UpsamplingNearest2d(scale_factor=2)

        #self.out_conv = nn.Conv2d(latent_dim // 2, 3, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(latent_dim//2, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        x = self.linear(x)
        x = torch.reshape(x, (-1, self.latent_dim*4, self.initial_image_size, self.initial_image_size))
        x = self.upsample_1(self.spd_blck_1(x, mask))
        x = self.upsample_2(self.spd_blck_2(x, mask))
        x = self.upsample_3(self.spd_blck_3(x, mask))
        x = self.upsample_4(self.spd_blck_4(x, mask))
        x = self.upsample_5(self.spd_blck_5(x, mask))
        x = self.upsample_6(self.spd_blck_6(x, mask))
        # x = self.upsample_7(self.spd_blck_7(x, mask))
        x = self.tanh(self.out_conv(x))
        return x

class GauGANUnetGenerator(nn.Module):
    def __init__(self, mask_channels, latent_dim, initial_image_size, skip_dim):
        super(GauGANUnetGenerator, self).__init__()
        self.linear = nn.Linear(latent_dim, initial_image_size*initial_image_size*latent_dim*4)
        self.initial_image_size = initial_image_size
        self.latent_dim = latent_dim

        self.spd_blck_1 = SPADE_ResBlock(1 / 2 ** 6, latent_dim * 4 + skip_dim, latent_dim * 4, mask_channels)
        self.upsample_1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_2 = SPADE_ResBlock(1 / 2 ** 5, latent_dim * 4 + skip_dim, latent_dim * 4, mask_channels)
        self.upsample_2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_3 = SPADE_ResBlock(1 / 2 ** 4, latent_dim * 4 + skip_dim, latent_dim * 4, mask_channels)
        self.upsample_3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_4 = SPADE_ResBlock(1 / 2 ** 3, latent_dim * 4 + skip_dim, latent_dim * 2, mask_channels)
        self.upsample_4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_5 = SPADE_ResBlock(1 / 2 ** 2, latent_dim * 2 + skip_dim, latent_dim, mask_channels)
        self.upsample_5 = nn.UpsamplingNearest2d(scale_factor=2)

        self.spd_blck_6 = SPADE_ResBlock(1 / 2, latent_dim + skip_dim, latent_dim // 2, mask_channels)
        self.upsample_6 = nn.UpsamplingNearest2d(scale_factor=2)

        # self.spd_blck_7 = SPADE_ResBlock(1, latent_dim // 2, latent_dim // 4, mask_channels)
        # self.upsample_7 = nn.UpsamplingNearest2d(scale_factor=2)

        self.out_conv = nn.Conv2d(latent_dim // 2, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, mask, skips):
        x = self.linear(x)
        x = torch.reshape(x, (-1, self.latent_dim*4, self.initial_image_size, self.initial_image_size))
        x = self.upsample_1(self.spd_blck_1(torch.cat((x, skips[5]), dim=1), mask))
        x = self.upsample_2(self.spd_blck_2(torch.cat((x, skips[4]), dim=1), mask))
        x = self.upsample_3(self.spd_blck_3(torch.cat((x, skips[3]), dim=1), mask))
        x = self.upsample_4(self.spd_blck_4(torch.cat((x, skips[2]), dim=1), mask))
        x = self.upsample_5(self.spd_blck_5(torch.cat((x, skips[1]), dim=1), mask))
        x = self.upsample_6(self.spd_blck_6(torch.cat((x, skips[0]), dim=1), mask))
        # x = self.upsample_7(self.spd_blck_7(x, mask))
        x = self.tanh(self.out_conv(x))
        return x
