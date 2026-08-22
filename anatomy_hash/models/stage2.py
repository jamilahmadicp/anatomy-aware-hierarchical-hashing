from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
import timm


class MLPProjection(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, out_dim))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class HierarchicalHashNet(nn.Module):
    def __init__(self, anatomy_fine_maps: dict, hash_bits=128, embedding_dim=256,
                 backbone="swin_tiny_patch4_window7_224.ms_in1k", pretrained=True,
                 use_anatomy_heads=True):
        super().__init__()
        self.backbone_name = backbone
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.feature_dim = int(self.encoder.num_features)
        self.hash_bits = int(hash_bits)
        self.embedding_dim = int(embedding_dim)
        self.use_anatomy_heads = bool(use_anatomy_heads)
        self.anatomy_fine_maps = {int(k): v for k, v in anatomy_fine_maps.items()}
        if self.use_anatomy_heads:
            self.proj = nn.ModuleDict()
            self.cls = nn.ModuleDict()
            self.hash = nn.ModuleDict()
            for a, m in self.anatomy_fine_maps.items():
                key = str(a)
                self.proj[key] = MLPProjection(self.feature_dim, 512, embedding_dim)
                self.cls[key] = nn.Linear(embedding_dim, len(m["global_ids"]))
                self.hash[key] = nn.Linear(embedding_dim, hash_bits)
        else:
            global_fine = sorted({f for m in self.anatomy_fine_maps.values() for f in m["global_ids"]})
            self.global_fine = global_fine
            self.global_to_local = {int(f): i for i, f in enumerate(global_fine)}
            self.local_to_global = {i: int(f) for i, f in enumerate(global_fine)}
            self.proj_shared = MLPProjection(self.feature_dim, 512, embedding_dim)
            self.cls_shared = nn.Linear(embedding_dim, len(global_fine))
            self.hash_shared = nn.Linear(embedding_dim, hash_bits)

    def encode(self, x):
        return self.encoder(x)

    def route_from_features(self, features, route_ids):
        """Apply the route-specific heads to a batch of encoder features.

        The routed outputs are accumulated in FP32 intentionally. Under CUDA AMP,
        Swin encoder features can be FP16 while normalization/tanh/head outputs may
        be promoted to FP32 by autocast. Allocating the destination tensors with
        ``features.new_zeros`` therefore causes a Half/Float indexed-assignment
        error on some PyTorch/CUDA combinations. Keeping the routed embedding/hash
        outputs in FP32 is both AMP-safe and numerically preferable for contrastive
        and quantization losses. Gradients still propagate through the casts and
        indexed CopySlices operations.
        """
        B = features.shape[0]
        z_out = torch.zeros((B, self.embedding_dim), device=features.device, dtype=torch.float32)
        q_out = torch.zeros((B, self.hash_bits), device=features.device, dtype=torch.float32)
        groups = []
        if not self.use_anatomy_heads:
            z = self.proj_shared(features).float()
            q = torch.tanh(self.hash_shared(z)).float()
            logits = self.cls_shared(z).float()
            return {"embedding": z, "continuous_code": q, "groups": [(torch.arange(B, device=features.device), -1, logits)]}
        for a in torch.unique(route_ids).tolist():
            idx = torch.where(route_ids == int(a))[0]
            key = str(int(a))
            if key not in self.proj:
                raise KeyError(f"No anatomy head for route {a}")
            z = self.proj[key](features[idx]).float()
            q = torch.tanh(self.hash[key](z)).float()
            logits = self.cls[key](z).float()
            z_out[idx] = z
            q_out[idx] = q
            groups.append((idx, int(a), logits))
        return {"embedding": z_out, "continuous_code": q_out, "groups": groups}

    def forward(self, x, route_ids):
        f = self.encode(x)
        out = self.route_from_features(f, route_ids)
        out["features"] = f
        return out

    def local_targets(self, anatomy_id, global_fine_targets):
        if not self.use_anatomy_heads:
            return torch.tensor([self.global_to_local[int(v)] for v in global_fine_targets.tolist()], device=global_fine_targets.device)
        mp = self.anatomy_fine_maps[int(anatomy_id)]["global_to_local"]
        return torch.tensor([mp[int(v)] for v in global_fine_targets.tolist()], device=global_fine_targets.device)

    def local_to_global_ids(self, anatomy_id):
        if not self.use_anatomy_heads:
            return [self.local_to_global[i] for i in range(len(self.local_to_global))]
        m = self.anatomy_fine_maps[int(anatomy_id)]["local_to_global"]
        return [m[i] for i in range(len(m))]
