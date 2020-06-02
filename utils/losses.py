import torch


def KL_divergence(mu, logsigma):
    return -0.5 * torch.sum(1 + logsigma - mu ** 2 - torch.exp(logsigma))


def hinge_loss_discriminator(fake_preds, real_preds):
    rpl = torch.min(real_preds - 1, torch.zeros_like(real_preds))
    fpl = torch.min(-fake_preds - 1,  torch.zeros_like(fake_preds))
    return -torch.mean(
        torch.add(rpl, fpl))


def hinge_loss_generator(fake_preds):
    return -torch.mean(fake_preds)


def ls_loss_discriminator(fake_preds, real_preds):
    return 0.5 * torch.mean(real_preds**2) + 0.5 * torch.mean((fake_preds - 0.9)**2)


def ls_loss_generator(fake_preds):
    return 0.5 * torch.mean(fake_preds**2)


def perceptual_loss(fake_feats, real_feats, lmbda):
    return lmbda * torch.nn.functional.l1_loss(fake_feats, real_feats, reduction='mean')


def masked_l1(predict, target, mask):
    return torch.sum(torch.abs((predict-target)*mask)) / torch.sum(mask)
