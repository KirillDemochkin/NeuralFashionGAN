from models.gaugan_generators import GauGANGenerator
import torch

def test_generator():
    test = torch.empty(3, 256).uniform_(-1, 1)
    mask = torch.ones(3, 5, 256, 256)

    gg = GauGANGenerator(5, 256, 4)

    out = gg(test, mask)

