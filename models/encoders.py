import torch
import torch.nn as nn
import torchvision

class BasicEncoder(nn.Module):
    def __init__(self,in_channels):
        super(BasicEncoder, self).__init__()
        self.block_1 = Discriminator_block(in_channels * 2, 64,  normalization=False)
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


class Vgg19Full(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19Full, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19_bn(pretrained=True).features
        print(vgg_pretrained_features)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.slice8 = torch.nn.Sequential()
        self.slice9 = torch.nn.Sequential()
        self.slice10 = torch.nn.Sequential()
        self.slice11 = torch.nn.Sequential()
        self.slice12 = torch.nn.Sequential()
        self.slice13 = torch.nn.Sequential()
        self.slice14 = torch.nn.Sequential()
        self.slice15 = torch.nn.Sequential()
        self.slice16 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 6):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 10):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 13):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(13, 17):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(17, 20):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 23):
            self.slice7.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 26):
            self.slice8.add_module(str(x), vgg_pretrained_features[x])
        for x in range(26, 30):
            self.slice9.add_module(str(x), vgg_pretrained_features[x])
        for x in range(30, 33):
            self.slice10.add_module(str(x), vgg_pretrained_features[x])
        for x in range(33, 36):
            self.slice11.add_module(str(x), vgg_pretrained_features[x])
        for x in range(36, 39):
            self.slice12.add_module(str(x), vgg_pretrained_features[x])
        for x in range(39, 43):
            self.slice13.add_module(str(x), vgg_pretrained_features[x])
        for x in range(43, 46):
            self.slice14.add_module(str(x), vgg_pretrained_features[x])
        for x in range(46, 49):
            self.slice15.add_module(str(x), vgg_pretrained_features[x])
        for x in range(49, 52):
            self.slice16.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        h_relu6 = self.slice6(h_relu5)
        h_relu7 = self.slice7(h_relu6)
        h_relu8 = self.slice8(h_relu7)
        h_relu9 = self.slice9(h_relu8)
        h_relu10 = self.slice10(h_relu9)
        h_relu11 = self.slice11(h_relu10)
        h_relu12 = self.slice12(h_relu11)
        h_relu13 = self.slice13(h_relu12)
        h_relu14 = self.slice14(h_relu13)
        h_relu15 = self.slice15(h_relu14)
        h_relu16 = self.slice16(h_relu15)
        res = (h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7, h_relu8, h_relu9, h_relu10, h_relu11, h_relu12, h_relu13, h_relu14, h_relu15, h_relu16)
        return torch.cat([r.view(r.size(0), -1) for r in res], dim=1)