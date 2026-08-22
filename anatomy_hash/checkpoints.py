from __future__ import annotations
import torch
from .models.stage1 import Stage1AnatomyClassifier
from .models.stage2 import HierarchicalHashNet
from .models.baselines import FlatHashModel


def load_stage1(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Stage1AnatomyClassifier(ck["num_classes"], ck["model_name"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, float(ck.get("temperature", 1.0)), ck


def load_stage2(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = HierarchicalHashNet(ck["anatomy_fine_maps"], hash_bits=ck["hash_bits"], embedding_dim=ck["embedding_dim"],
                                backbone=ck["backbone"], pretrained=False, use_anatomy_heads=ck.get("use_anatomy_heads", True))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


def load_flat_hash(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = FlatHashModel(ck["num_classes"], bits=ck["hash_bits"], backbone=ck["backbone"], pretrained=False,
                          embedding_dim=ck["embedding_dim"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck
