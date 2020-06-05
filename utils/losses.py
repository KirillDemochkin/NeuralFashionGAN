import torch


def KL_divergence(mu, logsigma):
    return -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())


def hinge_loss_discriminator(fake_preds, real_preds):
    rpl = torch.min(real_preds - 1, torch.zeros_like(real_preds))
    fpl = torch.min(-fake_preds - 1, torch.zeros_like(fake_preds))
    return -torch.mean(
        torch.add(rpl, fpl))


def hinge_loss_generator(fake_preds):
    return -torch.mean(fake_preds)


def perceptual_loss(fake_feats, real_feats, lmbda):
    return lmbda * torch.nn.functional.l1_loss(fake_feats, real_feats, reduction='mean')


def masked_l1(predict, target, mask):
    return torch.sum(torch.abs((predict-target)*mask)) / torch.sum(mask)

def ortho(model, strength=1e-4, blacklist=[]):
        with torch.no_grad():
            for param in model.parameters():
                # Only apply this to parameters with at least 2 axes, and not in the blacklist
                if len(param.shape) < 2 or any([param is item for item in blacklist]):
                    continue
                w = param.view(param.shape[0], -1)
                grad = (2 * torch.mm(torch.mm(w, w.t())
                                     * (1. - torch.eye(w.shape[0], device=w.device)), w))
                param.grad.data += strength * grad.view(param.shape)