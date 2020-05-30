import torch
import torch.nn as nn
from .gaugan_layers import Discriminator_block


class MultiscaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.opt = opt
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)
        subnetD1 = GauGANDiscriminator(nn.Module)
        subnetD2 = GauGANDiscriminator(nn.Module)

    def forward(self, input):
        out = []
        out1 = subnetD1(input)
        input = self.avg_pool(input)
        out.append(out1)
        out2 = subnetD2(input)
        out.append(out2)
        return out


class GauGANDiscriminator(nn.Module):
    def __init__(self,in_channels):
        super(GauGANDiscriminator, self).__init__()
        self.block_1 = Discriminator_block(in_channels * 2, 64,  normalization=False)
        self.block_2 = Discriminator_block(64, 128)
        self.block_3 = Discriminator_block(128, 256)
        self.block_4 = Discriminator_block(256, 512)
        self.out_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x, mask):
        input = torch.cat((x, mask), 1)
        x = self.out_conv(self.block_4(self.block_3(self.block_2(self.block_1(input)))))
        return x
