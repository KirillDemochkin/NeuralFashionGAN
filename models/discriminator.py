import torch
import torch.nn as nn
from .gaugan_layers import Discriminator_block


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)
        self.subnetD1 = GauGANDiscriminator(in_channels)
        self.subnetD2 = GauGANDiscriminator(in_channels)
        self.subnetD3 = GauGANDiscriminator(in_channels)

    def forward(self, x, mask):
        outs = []
        preds = []
        out, pred = self.subnetD1(x, mask)
        x = self.avg_pool(x)
        mask = self.avg_pool(mask)
        outs.append(out)
        preds.append(pred)
        out, pred = self.subnetD2(x, mask)
        x = self.avg_pool(x)
        mask = self.avg_pool(mask)
        outs.append(out)
        preds.append(pred)
        out, pred = self.subnetD3(x, mask)
        outs.append(out)
        preds.append(pred)
        return outs, preds


class GauGANDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(GauGANDiscriminator, self).__init__()
        self.block_1 = Discriminator_block(in_channels, 64,  normalization=False)
        self.block_2 = Discriminator_block(64, 128)
        self.block_3 = Discriminator_block(128, 256)
        self.block_4 = Discriminator_block(256, 512)
        self.out_conv = nn.Conv2d(512, 1, kernel_size=4, padding=1)

    def forward(self, x, mask):
        input = torch.cat((x, mask), 1)
        x1 = self.block_1(input)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        x4 = self.block_4(x3)
        preds = self.out_conv(x4)
        feats = (x1, x2, x3, x4)
        return preds, torch.cat([r.view(r.size(0), -1) for r in feats], dim=1)
