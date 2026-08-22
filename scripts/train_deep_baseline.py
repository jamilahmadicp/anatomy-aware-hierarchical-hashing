#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from anatomy_hash.data.manifest import load_manifest, split_df
from anatomy_hash.data.dataset import RadiographDataset, PKBatchSampler
from anatomy_hash.data.transforms import build_transform
from anatomy_hash.models.baselines import FlatHashModel, dsh_loss, hashnet_loss, dch_loss
from anatomy_hash.losses import supervised_contrastive_loss, quantization_loss, bit_balance_loss
from anatomy_hash.training import make_optimizer, make_scheduler, device_from_arg, class_weights
from anatomy_hash.utils.seed import seed_everything
from anatomy_hash.utils.io import environment_snapshot, save_json


def pair_loss(method, q, y, z):
    if method == "dsh":
        return dsh_loss(q, y)
    if method == "hashnet":
        return hashnet_loss(q, y)
    if method == "dch":
        return dch_loss(q, y)
    if method == "flat_hash":
        return (
            supervised_contrastive_loss(z, y, 0.07)
            + 0.1 * quantization_loss(q)
            + 0.01 * bit_balance_loss(q)
        )
    raise ValueError(method)


def binary_map(codes: np.ndarray, labels: np.ndarray) -> float:
    """Mean AP for exact-label retrieval within a validation set."""
    codes = np.asarray(codes, dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int64)
    if len(codes) < 2:
        return float("nan")
    dot = codes.astype(np.int16) @ codes.astype(np.int16).T
    dist = (codes.shape[1] - dot.astype(np.float32)) / (2.0 * codes.shape[1])
    np.fill_diagonal(dist, np.inf)
    aps = []
    for i in range(len(codes)):
        relevant = labels == labels[i]
        relevant[i] = False
        total_rel = int(relevant.sum())
        if total_rel == 0:
            continue
        order = np.argsort(dist[i], kind="stable")
        order = order[np.isfinite(dist[i, order])]
        rel = relevant[order].astype(np.int32)
        csum = np.cumsum(rel)
        precision = csum / np.arange(1, len(rel) + 1)
        aps.append(float((precision * rel).sum() / total_rel))
    return float(np.mean(aps)) if aps else float("nan")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = 0
    correct = 0
    codes = []
    labels = []
    for b in loader:
        x = b["image"].to(device)
        y = b["fine"].to(device)
        o = model(x)
        correct += int((o["logits"].argmax(1) == y).sum())
        n += len(y)
        q = o["continuous_code"].float()
        bcode = torch.where(q >= 0, torch.ones_like(q), -torch.ones_like(q)).to(torch.int8)
        codes.append(bcode.cpu().numpy())
        labels.append(y.cpu().numpy())
    c = np.concatenate(codes, axis=0)
    y = np.concatenate(labels, axis=0)
    return {
        "val_accuracy": correct / max(1, n),
        "val_hash_mAP": binary_map(c, y),
        "val_unique_codes": int(len(np.unique(c, axis=0))),
        "val_plus_one_fraction": float((c == 1).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["dsh", "hashnet", "dch", "flat_hash"], required=True)
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--backbone", default="swin_tiny_patch4_window7_224.ms_in1k")
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--classes-per-batch", type=int, default=16)
    ap.add_argument("--samples-per-class", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--patience", type=int, default=8)
    a = ap.parse_args()

    seed_everything(a.seed)
    device = device_from_arg(a.device)
    df, _ = load_manifest(a.manifest)
    tr = split_df(df, "train")
    va = split_df(df, "val")
    nc = int(df.fine_id.max() + 1)

    train_ds = RadiographDataset(tr, build_transform(train=True))
    sampler = PKBatchSampler(
        tr.fine_id.values,
        a.classes_per_batch,
        a.samples_per_class,
        seed=a.seed,
    )
    dl = DataLoader(train_ds, batch_sampler=sampler, num_workers=a.workers)
    vdl = DataLoader(
        RadiographDataset(va, build_transform(train=False)),
        batch_size=a.batch_size,
        num_workers=a.workers,
        shuffle=False,
    )

    model = FlatHashModel(nc, a.bits, a.backbone, True, a.embedding_dim).to(device)
    weights = class_weights(tr.fine_id.values, nc).to(device)
    opt = make_optimizer(model, a.lr, a.weight_decay)
    sch = make_scheduler(opt, a.epochs)

    best_map = -np.inf
    best_acc = -np.inf
    bad = 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hist = []

    for ep in range(a.epochs):
        model.train()
        losses = []
        for b in tqdm(dl, desc=f"{a.method} {ep+1}/{a.epochs}"):
            x = b["image"].to(device)
            y = b["fine"].to(device)
            opt.zero_grad(set_to_none=True)
            o = model(x)
            ce = F.cross_entropy(o["logits"], y, weight=weights)
            pl = pair_loss(a.method, o["continuous_code"], y, o["embedding"])
            loss = ce + pl
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        sch.step()

        val = evaluate(model, vdl, device)
        row = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), **val}
        hist.append(row)
        pd.DataFrame(hist).to_csv(out.with_suffix(".history.csv"), index=False)
        print(
            f"epoch {ep+1}: val_acc={val['val_accuracy']:.4f}, "
            f"val_hash_mAP={val['val_hash_mAP']:.4f}, unique_codes={val['val_unique_codes']}"
        )

        score = val["val_hash_mAP"] if np.isfinite(val["val_hash_mAP"]) else -np.inf
        improved = (score > best_map + 1e-8) or (
            abs(score - best_map) <= 1e-8 and val["val_accuracy"] > best_acc
        )
        if improved:
            best_map = score
            best_acc = val["val_accuracy"]
            bad = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_classes": nc,
                    "hash_bits": a.bits,
                    "backbone": a.backbone,
                    "embedding_dim": a.embedding_dim,
                    "method": a.method,
                    "seed": a.seed,
                    "train_args": vars(a),
                    "selection_metric": "validation_hash_mAP",
                    "best_val_hash_mAP": float(best_map),
                    "best_val_accuracy": float(best_acc),
                },
                out,
            )
        else:
            bad += 1
        if bad >= a.patience:
            print(f"Early stopping after {ep+1} epochs; best validation hash mAP={best_map:.4f}")
            break

    save_json(environment_snapshot(), out.with_suffix(".environment.json"))
    print(f"{a.method}: best validation hash mAP={best_map:.4f}; val accuracy={best_acc:.4f} -> {out}")


if __name__ == "__main__":
    main()
