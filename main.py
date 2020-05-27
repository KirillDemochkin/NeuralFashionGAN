from models.gaugan_generators import GauGANGenerator
from models.discriminator import GauGANDiscriminator
import torch

def test_generator():
    test = torch.empty(3, 256).uniform_(-1, 1)
    mask = torch.ones(3, 5, 256, 256)

    gg = GauGANGenerator(5, 256, 4)

    out = gg(test, mask)

def test_discriminator():
    test = torch.empty(3, 5, 256, 256).uniform_(-1, 1)
    mask = torch.ones(3, 5, 256, 256)

    gd = GauGANDiscriminator(5)

    out = gd(test, mask)
    print(out.shape)
