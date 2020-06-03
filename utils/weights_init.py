import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.orthogonal_(m.weight.data, gain=0.0001)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, gain=0.0001)
        #nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.orthogonal_(m.weight.data, gain = 0.0001)
        nn.init.constant_(m.bias.data, 0.0)