import torch


def KL_divergence(mu, logsigma):
    return -0.5 * torch.sum(1 + logsigma - mu ** 2 - torch.exp(logsigma))


def hinge_loss_discriminator(fake_preds, real_preds):
    rpl = -torch.ones_like(real_preds) + real_preds
    rpl = torch.where(rpl > 0, torch.zeros_like(rpl), rpl)
    fpl = -torch.ones_like(fake_preds) - fake_preds
    fpl = torch.where(fpl > 0, torch.zeros_like(fpl), fpl)
    return -torch.mean(
        torch.add(rpl, fpl))


def hinge_loss_generator(fake_preds):
    return -torch.mean(fake_preds)


def perceptual_loss(fake_feats, real_feats, lmbda):
    return lmbda * torch.nn.functional.l1_loss(fake_feats, real_feats, reduction='mean')

def masked_l1(predict, target, mask):
    return torch.sum(torch.abs((predict-target)*mask)) / torch.sum(mask)