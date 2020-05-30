import torch


def KL_divergence(mu, logsigma):
    return -0.5 * torch.sum(1 + logsigma - mu ** 2 - torch.exp(logsigma))


def hinge_loss_discriminator(fake_preds, real_preds):
    return -torch.mean(
        torch.add(min(0, -torch.ones_like(real_preds) + real_preds), min(0, -torch.ones_like(fake_preds) - fake_preds)))


def hinge_loss_generator(fake_preds):
    return -torch.mean(fake_preds)


def perceptual_loss(fake_feats, real_feats, lmbda):
    return lmbda * torch.nn.functional.l1_loss(fake_feats, real_feats, reduction='mean')
