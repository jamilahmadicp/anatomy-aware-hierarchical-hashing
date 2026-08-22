from __future__ import annotations
import torch
import torch.nn.functional as F


def supervised_contrastive_loss(z, labels, temperature=0.07):
    z = F.normalize(z, dim=1)
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    B = z.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    exp_logits = torch.exp(logits) * (~eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_count = positives.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return z.sum() * 0.0
    mean_log_prob_pos = (positives * log_prob).sum(dim=1) / pos_count.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def quantization_loss(q):
    target = q.sign().detach()
    target[target == 0] = 1
    return (q - target).pow(2).mean()


def bit_balance_loss(q):
    return q.mean(dim=0).pow(2).mean()
