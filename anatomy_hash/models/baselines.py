from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
import timm


class FlatHashModel(nn.Module):
    def __init__(self, num_classes, bits=128, backbone="swin_tiny_patch4_window7_224.ms_in1k", pretrained=True, embedding_dim=256):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        d = int(self.encoder.num_features)
        self.proj = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, embedding_dim))
        self.cls = nn.Linear(embedding_dim, num_classes)
        self.hash = nn.Linear(embedding_dim, bits)
    def forward(self, x):
        f = self.encoder(x)
        z = F.normalize(self.proj(f), dim=-1)
        q = torch.tanh(self.hash(z))
        return {"features": f, "embedding": z, "continuous_code": q, "logits": self.cls(z)}


def pairwise_similarity(labels):
    return (labels[:, None] == labels[None, :]).float()


def dsh_loss(q, labels, margin=2.0):
    sim = pairwise_similarity(labels)
    dist = torch.cdist(q, q, p=2).pow(2)
    pos = sim * dist
    neg = (1 - sim) * torch.relu(margin - torch.sqrt(dist + 1e-8)).pow(2)
    mask = 1 - torch.eye(len(labels), device=q.device)
    return ((pos + neg) * mask).sum() / mask.sum().clamp_min(1)


def hashnet_loss(q, labels, alpha=1.0):
    sim = pairwise_similarity(labels)
    theta = alpha * (q @ q.T) / max(1, q.shape[1])
    loss = F.softplus(theta) - sim * theta
    mask = 1 - torch.eye(len(labels), device=q.device)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def dch_loss(q, labels, gamma=10.0, quant_weight=0.1):
    sim = pairwise_similarity(labels)
    d2 = torch.cdist(q, q, p=2).pow(2)
    p = gamma / (gamma + d2)
    pair = -(sim * torch.log(p + 1e-8) + (1 - sim) * torch.log(1 - p + 1e-8))
    mask = 1 - torch.eye(len(labels), device=q.device)
    pair = (pair * mask).sum() / mask.sum().clamp_min(1)
    quant = (q.abs() - 1).abs().mean()
    return pair + quant_weight * quant
