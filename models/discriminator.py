import torch
import torch.nn as nn
from .gaugan_layers import Discriminator_block

class GauGANDiscriminator(nn.Module):
    def __init__(self,in_channels):
        super(GauGANDiscriminator, self).__init__()
        self.block_1 = Discriminator_block(in_channels * 2, 64,  normalization=False)
        self.block_2 = Discriminator_block(64, 128)
        self.block_3 = Discriminator_block(128, 256)
        self.block_4 = Discriminator_block(256, 512)
        self.out_conv = nn.Conv2d(512, 1, kernel_size=4, padding=1)

    def forward(self, x, mask):
        input = torch.cat((x, mask), 1)
        x = self.out_conv(self.block_4(self.block_3(self.block_2(self.block_1(input)))))
        return x
