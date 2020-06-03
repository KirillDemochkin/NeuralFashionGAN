import torch
import torch.nn as nn
import torchvision


class BasicDownsamplingConBlock(nn.Module):
    def __init__(self, inc, nc):
        super(BasicDownsamplingConBlock, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(inc, nc, kernel_size=4, stride=2, padding=1), eps=1e-6)
        self.norm = nn.InstanceNorm2d(nc)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.leaky(self.norm(self.conv(x)))


class BasicEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(BasicEncoder, self).__init__()
        self.conv_1 = BasicDownsamplingConBlock(3, 64)
        self.conv_2 = BasicDownsamplingConBlock(64, 128)
        self.conv_3 = BasicDownsamplingConBlock(128, 256)
        self.conv_4 = BasicDownsamplingConBlock(256, 512)
        self.conv_5 = BasicDownsamplingConBlock(512, 512)
        self.conv_6 = BasicDownsamplingConBlock(512, 512)
        self.mu_fc = nn.Linear(8192, latent_dim)
        self.sigma_fc = nn.Linear(8192, latent_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = torch.reshape(x, (-1, 8192))
        mu = self.mu_fc(x)
        sigma = self.sigma_fc(x)
        std = torch.exp(0.5 * sigma)
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu), mu, sigma


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim):
        super(MappingNetwork, self).__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(latent_dim, latent_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(latent_dim, latent_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(latent_dim, latent_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(latent_dim, latent_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(latent_dim, latent_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(latent_dim, latent_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(latent_dim, latent_dim),
                                 nn.ReLU(inplace=True)
                                 )

    def forward(self, x):
        x = x / (torch.norm(x + 1e-10, p=2, dim=1, keepdim=True) + 1e-10)
        return self.net(x)


class UnetEncoder(nn.Module):
    def __init__(self, latent_dim, skip_dim, resample=True):
        super(UnetEncoder, self).__init__()
        self.resample = resample
        self.conv_1 = BasicDownsamplingConBlock(3, 64)
        self.skip_1 = nn.Sequential(nn.Conv2d(64, skip_dim, kernel_size=1), nn.ReLU(inplace=True))
        self.conv_2 = BasicDownsamplingConBlock(64, 128)
        self.skip_2 = nn.Sequential(nn.Conv2d(128, skip_dim, kernel_size=1), nn.ReLU(inplace=True))
        self.conv_3 = BasicDownsamplingConBlock(128, 256)
        self.skip_3 = nn.Sequential(nn.Conv2d(256, skip_dim, kernel_size=1), nn.ReLU(inplace=True))
        self.conv_4 = BasicDownsamplingConBlock(256, 512)
        self.skip_4 = nn.Sequential(nn.Conv2d(512, skip_dim, kernel_size=1), nn.ReLU(inplace=True))
        #self.conv_5 = BasicDownsamplingConBlock(512, 512)
        #self.conv_6 = BasicDownsamplingConBlock(512, 512)
        self.mu_fc = nn.Linear(8192, latent_dim)
        if self.resample:
            self.sigma_fc = nn.Linear(8192, latent_dim)

    def forward(self, x):
        skips = []
        x = self.conv_1(x)
        skips.append(self.skip_1(x))
        x = self.conv_2(x)
        skips.append(self.skip_2(x))
        x = self.conv_3(x)
        skips.append(self.skip_3(x))
        x = self.conv_4(x)
        skips.append(self.skip_4(x))
        #x = self.conv_5(x)
        #x = self.conv_6(x)
        x = torch.reshape(x, (-1, 8192))
        mu = self.mu_fc(x)

        if not self.resample:
            return mu, skips
        else:
            sigma = self.sigma_fc(x)
            std = torch.exp(0.5 * sigma)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu), mu, sigma, skips


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


class StyleEncoder(nn.Module):
    def __init__(self, latent_dim, skip_dim, rs):
        super(StyleEncoder, self).__init__()
        self.rs = rs
        self.conv_1 = BasicDownsamplingConBlock(3, 64)
        self.fc_1 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(64, skip_dim, kernel_size=1), eps=1e-6), nn.LeakyReLU(0.2, inplace=True))
        self.conv_2 = BasicDownsamplingConBlock(64, 128)
        self.fc_2 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(128, skip_dim, kernel_size=1), eps=1e-6), nn.LeakyReLU(0.2, inplace=True))
        self.conv_3 = BasicDownsamplingConBlock(128, 256)
        self.fc_3 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(256, skip_dim, kernel_size=1), eps=1e-6), nn.LeakyReLU(0.2, inplace=True))
        self.conv_4 = BasicDownsamplingConBlock(256, 512)
        self.fc_4 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(512, skip_dim, kernel_size=1), eps=1e-6), nn.LeakyReLU(0.2, inplace=True))
        self.conv_5 = BasicDownsamplingConBlock(512, 512)
        self.fc_5 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(512, skip_dim, kernel_size=1), eps=1e-6), nn.LeakyReLU(0.2, inplace=True))
        self.conv_6 = BasicDownsamplingConBlock(512, 512)
        self.fc_6 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(512, skip_dim, kernel_size=1), eps=1e-6), nn.LeakyReLU(0.2, inplace=True))
        self.fc = nn.Linear(8192//(rs**2), latent_dim)

    def forward(self, x, need_skips=True):
        skips = []
        x = self.conv_1(x)
        if need_skips:
            skips.append(self.fc_1(x))
        x = self.conv_2(x)
        if need_skips:
            skips.append(self.fc_2(x))
        x = self.conv_3(x)
        if need_skips:
            skips.append(self.fc_3(x))
        x = self.conv_4(x)
        if need_skips:
            skips.append(self.fc_4(x))
        x = self.conv_5(x)
        if need_skips:
            skips.append(self.fc_5(x))
        x = self.conv_6(x)
        if need_skips:
            skips.append(self.fc_6(x))
        x = torch.reshape(x, (-1, 8192//(self.rs**2)))
        x = self.fc(x)
        return x, skips

