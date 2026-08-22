from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def calibrated_stage1_prob(model, x, temperature=1.0):
    return F.softmax(model(x) / float(temperature), dim=1)

@torch.no_grad()
def hierarchical_global_prob(stage1, stage2, x, temperature=1.0, top_r=None):
    p_anat = calibrated_stage1_prob(stage1, x, temperature)
    feat = stage2.encode(x)
    n_global = max(f for m in stage2.anatomy_fine_maps.values() for f in m["global_ids"]) + 1
    out = x.new_zeros((x.shape[0], n_global))
    for a, mp in stage2.anatomy_fine_maps.items():
        route = torch.full((x.shape[0],), int(a), device=x.device, dtype=torch.long)
        routed = stage2.route_from_features(feat, route)
        logits = routed["groups"][0][2]
        cond = F.softmax(logits, dim=1)
        gids = stage2.local_to_global_ids(a)
        weight = p_anat[:, int(a)].unsqueeze(1)
        out[:, gids] = cond * weight
    if top_r is not None:
        top = torch.topk(p_anat, k=min(int(top_r), p_anat.shape[1]), dim=1).indices
        mask_anat = torch.zeros_like(p_anat).scatter_(1, top, 1.0)
        fine_mask = torch.zeros_like(out)
        for a, mp in stage2.anatomy_fine_maps.items():
            gids = stage2.local_to_global_ids(a)
            fine_mask[:, gids] = mask_anat[:, int(a)].unsqueeze(1)
        out = out * fine_mask
        out = out / out.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return p_anat, out, feat
