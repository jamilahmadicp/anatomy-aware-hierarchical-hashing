#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from anatomy_hash.data.manifest import load_manifest, split_df, build_anatomy_fine_maps
from anatomy_hash.data.dataset import RadiographDataset, PKBatchSampler
from anatomy_hash.data.transforms import build_transform
from anatomy_hash.models.stage2 import HierarchicalHashNet
from anatomy_hash.losses import supervised_contrastive_loss, quantization_loss
from anatomy_hash.training import make_optimizer, make_scheduler, device_from_arg, amp_context, class_weights
from anatomy_hash.utils.seed import seed_everything
from anatomy_hash.utils.io import save_json, environment_snapshot
from anatomy_hash.ablation import ABLATION_PROFILES, apply_ablation_profile
from anatomy_hash.codebooks import build_route_codebooks

# Protocol identifier stored in checkpoints to make training/evaluation settings auditable.
HASH_PROTOCOL = "prototype_margin_hash_v3"


def grouped_ce(model, groups, global_fine, class_weights_by_anat=None):
    losses = []
    for idx, a, logits in groups:
        targets = model.local_targets(a, global_fine[idx])
        weight = None
        if class_weights_by_anat and int(a) in class_weights_by_anat:
            weight = class_weights_by_anat[int(a)].to(logits.device)
        losses.append(F.cross_entropy(logits.float(), targets, weight=weight))
    return torch.stack(losses).mean() if losses else global_fine.sum() * 0.0


def grouped_hash_supcon(groups, continuous_code, global_fine, temperature=0.07):
    losses = []
    for idx, _a, _logits in groups:
        labels = global_fine[idx]
        if idx.numel() < 2 or torch.unique(labels).numel() < 2:
            continue
        losses.append(supervised_contrastive_loss(continuous_code[idx].float(), labels, temperature))
    return torch.stack(losses).mean() if losses else continuous_code.sum() * 0.0


def grouped_pairwise_hash_loss(groups, continuous_code, global_fine, temperature=0.20):
    """Pairwise semantic hash loss within each anatomy route."""
    losses = []
    for idx, _a, _logits in groups:
        if idx.numel() < 2:
            continue
        q = continuous_code[idx].float()
        y = global_fine[idx]
        if torch.unique(y).numel() < 2:
            continue
        nbits = max(1, q.shape[1])
        sim = (q @ q.T) / (float(nbits) * float(temperature))
        eye = torch.eye(len(idx), dtype=torch.bool, device=q.device)
        same = y[:, None].eq(y[None, :])
        target = torch.where(same, torch.ones_like(sim), -torch.ones_like(sim))
        mask = ~eye
        losses.append(F.softplus(-target[mask] * sim[mask]).mean())
    return torch.stack(losses).mean() if losses else continuous_code.sum() * 0.0


def grouped_bit_balance_loss(groups, continuous_code):
    losses = []
    for idx, _a, _logits in groups:
        if idx.numel() == 0:
            continue
        losses.append(continuous_code[idx].float().mean(dim=0).pow(2).mean())
    return torch.stack(losses).mean() if losses else continuous_code.sum() * 0.0


def grouped_prototype_losses(model, groups, continuous_code, global_fine, codebooks, sign_margin=0.35):
    """Directly supervise the *sign* of every hash bit with route-specific class prototypes.

    Returns (prototype_regression_loss, sign_margin_loss). The sign-margin term
    is zero only when each continuous hash value has the target sign and reaches
    at least the requested absolute margin.
    """
    proto_losses, margin_losses = [], []
    for idx, a, _logits in groups:
        if idx.numel() == 0:
            continue
        local = model.local_targets(a, global_fine[idx])
        cb = codebooks[int(a)].to(continuous_code.device, dtype=torch.float32)
        target = cb[local]
        q = continuous_code[idx].float()
        proto_losses.append(F.smooth_l1_loss(q, target, beta=0.25))
        signed = q * target
        margin_losses.append(F.relu(float(sign_margin) - signed).mean())
    zero = continuous_code.sum() * 0.0
    return (
        torch.stack(proto_losses).mean() if proto_losses else zero,
        torch.stack(margin_losses).mean() if margin_losses else zero,
    )


def make_local_weights(df, amap):
    out = {}
    for a, m in amap.items():
        g = df[df.anatomy_id == a]
        local = np.array([m["global_to_local"][int(f)] for f in g.fine_id], int)
        cnt = np.bincount(local, minlength=len(m["global_ids"])).astype(float)
        cnt[cnt == 0] = 1
        out[a] = torch.tensor(len(local) / (len(cnt) * cnt), dtype=torch.float32)
    return out


def gt_route_binary_map(codes, routes, labels):
    codes = np.asarray(codes, dtype=np.int8)
    routes = np.asarray(routes, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    aps = []
    for r in np.unique(routes):
        m = np.flatnonzero(routes == r)
        if len(m) < 2:
            continue
        b = codes[m]
        y = labels[m]
        dot = b.astype(np.int16) @ b.astype(np.int16).T
        dist = (b.shape[1] - dot.astype(np.float32)) / (2.0 * b.shape[1])
        np.fill_diagonal(dist, np.inf)
        same = y[:, None] == y[None, :]
        np.fill_diagonal(same, False)
        for i in range(len(m)):
            total_rel = int(same[i].sum())
            if total_rel <= 0:
                continue
            order = np.argsort(dist[i], kind="stable")
            order = order[np.isfinite(dist[i, order])]
            rel = same[i, order].astype(np.int32)
            csum = np.cumsum(rel)
            ranks = np.arange(1, len(rel) + 1)
            prec = csum / ranks
            aps.append(float((prec * rel).sum() / total_rel))
    return float(np.mean(aps)) if aps else float("nan")


def binary_diversity(codes, routes, labels=None):
    codes = np.asarray(codes, dtype=np.int8)
    routes = np.asarray(routes, dtype=np.int64)
    labels = None if labels is None else np.asarray(labels, dtype=np.int64)
    per_route = []
    min_interclass_hamming = np.inf
    for r in np.unique(routes):
        m = routes == r
        c = codes[m]
        per_route.append(len(np.unique(c, axis=0)))
        if labels is not None:
            y = labels[m]
            classes = np.unique(y)
            # Median bit per class gives a robust binary centroid.
            centroids = []
            for cls in classes:
                cc = c[y == cls]
                s = cc.astype(np.float32).mean(axis=0)
                centroids.append(np.where(s >= 0, 1, -1).astype(np.int8))
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    d = float(np.mean(centroids[i] != centroids[j]))
                    min_interclass_hamming = min(min_interclass_hamming, d)
    return {
        "val_hash_unique_global": int(len(np.unique(codes, axis=0))) if len(codes) else 0,
        "val_hash_min_unique_route": int(min(per_route)) if per_route else 0,
        "val_hash_mean_unique_route": float(np.mean(per_route)) if per_route else 0.0,
        "val_hash_min_interclass_hamming": float(min_interclass_hamming) if np.isfinite(min_interclass_hamming) else float("nan"),
    }


@torch.no_grad()
def evaluate(model, loader, device, class_weights):
    model.eval()
    n = 0
    correct = 0
    losses = []
    all_codes, all_routes, all_fine = [], [], []
    for b in loader:
        x = b["image"].to(device)
        a = b["anatomy"].to(device)
        f = b["fine"].to(device)
        out = model(x, a)
        ce = grouped_ce(model, out["groups"], f, class_weights)
        losses.append(float(ce))
        for idx, anat, logits in out["groups"]:
            pred = logits.argmax(1)
            tgt = model.local_targets(anat, f[idx])
            correct += int((pred == tgt).sum())
            n += len(idx)
        q = out["continuous_code"].float()
        binary = torch.where(q >= 0, torch.ones_like(q), -torch.ones_like(q)).to(torch.int8)
        all_codes.append(binary.cpu().numpy())
        all_routes.append(a.cpu().numpy())
        all_fine.append(f.cpu().numpy())
    codes = np.concatenate(all_codes, axis=0) if all_codes else np.empty((0, model.hash_bits), np.int8)
    routes = np.concatenate(all_routes, axis=0) if all_routes else np.empty((0,), np.int64)
    fine = np.concatenate(all_fine, axis=0) if all_fine else np.empty((0,), np.int64)
    val_map = gt_route_binary_map(codes, routes, fine)
    div = binary_diversity(codes, routes, fine)
    return float(np.mean(losses)), correct / max(1, n), val_map, div


def _ramp(epoch_zero_based: int, warmup_epochs: int, ramp_epochs: int) -> float:
    if epoch_zero_based < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, (epoch_zero_based - warmup_epochs + 1) / ramp_epochs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="swin_tiny_patch4_window7_224.ms_in1k")
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--classes-per-batch", type=int, default=16)
    ap.add_argument("--samples-per-class", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--hash-pair-temperature", type=float, default=0.20)
    ap.add_argument("--lambda-sup", type=float, default=1.0)
    ap.add_argument("--lambda-hash-sup", type=float, default=0.25)
    ap.add_argument("--lambda-hash-pair", type=float, default=0.25)
    ap.add_argument("--lambda-prototype", type=float, default=3.0)
    ap.add_argument("--lambda-sign-margin", type=float, default=2.0)
    ap.add_argument("--sign-margin", type=float, default=0.35)
    ap.add_argument("--prototype-min-hamming", type=float, default=0.35)
    ap.add_argument("--lambda-quant", type=float, default=0.02)
    ap.add_argument("--lambda-balance", type=float, default=0.01)
    ap.add_argument("--semantic-warmup-epochs", type=int, default=2)
    ap.add_argument("--quant-ramp-epochs", type=int, default=5)
    ap.add_argument("--min-unique-per-route", type=int, default=2)
    ap.add_argument("--shared-head", action="store_true")
    ap.add_argument(
        "--ablation-profile",
        choices=sorted(ABLATION_PROFILES),
        default="full",
        help="Named Stage-2 ablation. The profile overrides only the components it disables.",
    )
    ap.add_argument(
        "--allow-collapsed-checkpoint",
        action="store_true",
        help="Allow checkpoint selection without the binary-diversity gate. Intended only for diagnostic ablations.",
    )
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()
    apply_ablation_profile(a)

    seed_everything(a.seed)
    device = device_from_arg(a.device)
    df, _ = load_manifest(a.manifest)
    amap = build_anatomy_fine_maps(df[df.split.isin(["train", "val"])])
    tr = split_df(df, "train")
    va = split_df(df, "val")
    train_ds = RadiographDataset(tr, build_transform(train=True))
    val_ds = RadiographDataset(va, build_transform(train=False))

    if a.shared_head:
        train = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                           pin_memory=device.type == "cuda")
    else:
        sampler = PKBatchSampler(tr.fine_id.values, a.classes_per_batch, a.samples_per_class, seed=a.seed)
        train = DataLoader(train_ds, batch_sampler=sampler, num_workers=a.workers,
                           pin_memory=device.type == "cuda")
    val = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                     pin_memory=device.type == "cuda")

    model = HierarchicalHashNet(amap, a.bits, a.embedding_dim, a.backbone, True,
                                use_anatomy_heads=not a.shared_head).to(device)
    if a.shared_head:
        local_weights = {-1: class_weights(tr.fine_id.values, int(df.fine_id.max() + 1))}
        # Shared-head ablation cannot use independent route codebooks in the same way.
        # Fall back to a single global codebook keyed by -1.
        global_ids = sorted(tr.fine_id.astype(int).unique().tolist())
        pseudo = {-1: {"global_ids": global_ids, "global_to_local": {g: i for i, g in enumerate(global_ids)}}}
        codebooks = build_route_codebooks(pseudo, a.bits, a.seed, a.prototype_min_hamming)
    else:
        local_weights = make_local_weights(tr, amap)
        codebooks = build_route_codebooks(amap, a.bits, a.seed, a.prototype_min_hamming)

    opt = make_optimizer(model, a.lr, a.weight_decay)
    sch = make_scheduler(opt, a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not a.no_amp)

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    best_map = -np.inf
    best_acc = -np.inf
    bad = 0
    healthy_seen = False

    for ep in range(a.epochs):
        model.train()
        run = []
        quant_scale = _ramp(ep, a.semantic_warmup_epochs, a.quant_ramp_epochs)
        for b in tqdm(train, desc=f"stage2 {ep+1}/{a.epochs}"):
            x = b["image"].to(device)
            anat = b["anatomy"].to(device)
            fine = b["fine"].to(device)
            opt.zero_grad(set_to_none=True)
            route = anat if not a.shared_head else torch.zeros_like(anat)
            with amp_context(device, not a.no_amp):
                o = model(x, route)
                ce = grouped_ce(model, o["groups"], fine, local_weights)
                sup = supervised_contrastive_loss(o["embedding"], fine, a.temperature) if a.lambda_sup else ce * 0
                hash_sup = grouped_hash_supcon(o["groups"], o["continuous_code"], fine, a.temperature) if a.lambda_hash_sup else ce * 0
                pair = grouped_pairwise_hash_loss(o["groups"], o["continuous_code"], fine, a.hash_pair_temperature) if a.lambda_hash_pair else ce * 0
                proto, sign_margin = grouped_prototype_losses(
                    model, o["groups"], o["continuous_code"], fine, codebooks, a.sign_margin
                )
                ql = quantization_loss(o["continuous_code"]) if a.lambda_quant else ce * 0
                bl = grouped_bit_balance_loss(o["groups"], o["continuous_code"]) if a.lambda_balance else ce * 0
                loss = (
                    ce
                    + a.lambda_sup * sup
                    + a.lambda_hash_sup * hash_sup
                    + a.lambda_hash_pair * pair
                    + a.lambda_prototype * proto
                    + a.lambda_sign_margin * sign_margin
                    + quant_scale * a.lambda_quant * ql
                    + quant_scale * a.lambda_balance * bl
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run.append([
                float(loss.detach()), float(ce.detach()), float(sup.detach()),
                float(hash_sup.detach()), float(pair.detach()), float(proto.detach()),
                float(sign_margin.detach()), float(ql.detach()), float(bl.detach())
            ])

        sch.step()
        vl, va_acc, val_map, div = evaluate(model, val, device, local_weights)
        vals = np.mean(run, axis=0)
        row = {
            "epoch": ep + 1,
            "loss": vals[0],
            "ce": vals[1],
            "supcon": vals[2],
            "hash_supcon": vals[3],
            "hash_pair": vals[4],
            "prototype": vals[5],
            "sign_margin": vals[6],
            "quant": vals[7],
            "balance_routewise": vals[8],
            "quant_scale": quant_scale,
            "val_ce": vl,
            "val_gt_route_accuracy": va_acc,
            "val_gt_route_hash_mAP": val_map,
            **div,
        }
        hist.append(row)
        pd.DataFrame(hist).to_csv(outp.with_suffix(".history.csv"), index=False)
        print(
            f"epoch {ep+1}: val_acc={va_acc:.4f}, val_hash_mAP={val_map:.4f}, "
            f"min_unique/route={div['val_hash_min_unique_route']}, global_unique={div['val_hash_unique_global']}, "
            f"min_class_H={div['val_hash_min_interclass_hamming']:.3f}, quant_scale={quant_scale:.2f}"
        )

        required = int(a.min_unique_per_route)
        healthy = div["val_hash_min_unique_route"] >= required
        eligible = healthy or bool(a.allow_collapsed_checkpoint)
        healthy_seen = healthy_seen or eligible
        score_map = val_map if np.isfinite(val_map) else -np.inf
        improved = eligible and (
            (score_map > best_map + 1e-8)
            or (abs(score_map - best_map) <= 1e-8 and va_acc > best_acc)
        )
        if improved:
            best_map = score_map
            best_acc = va_acc
            bad = 0
            serial_codebooks = {int(k): v.cpu() for k, v in codebooks.items()}
            torch.save({
                "state_dict": model.state_dict(),
                "anatomy_fine_maps": amap,
                "hash_bits": a.bits,
                "embedding_dim": a.embedding_dim,
                "backbone": a.backbone,
                "use_anatomy_heads": not a.shared_head,
                "train_args": vars(a),
                "seed": a.seed,
                "hash_protocol": HASH_PROTOCOL,
                "hash_codebooks": serial_codebooks,
                "selection_metric": (
                    "val_gt_route_hash_mAP" if a.allow_collapsed_checkpoint
                    else "val_gt_route_hash_mAP_with_diversity_gate"
                ),
                "ablation_profile": a.ablation_profile,
                "best_val_hash_mAP": float(best_map),
                "best_val_gt_route_accuracy": float(best_acc),
                "best_val_min_unique_route": int(div["val_hash_min_unique_route"]),
                "best_val_min_interclass_hamming": float(div["val_hash_min_interclass_hamming"]),
            }, outp)
        else:
            # Do not count unhealthy warm-up epochs against early stopping before
            # the first diversity-valid checkpoint has ever been seen.
            if healthy_seen:
                bad += 1
        if healthy_seen and bad >= a.patience:
            qualifier = "validation" if a.allow_collapsed_checkpoint else "diversity-valid validation"
            print(f"Early stopping after {ep+1} epochs; best {qualifier} hash mAP={best_map:.4f}")
            break

    save_json(environment_snapshot(), outp.with_suffix(".environment.json"))
    if not outp.exists():
        raise SystemExit(
            "No eligible Stage-2 checkpoint was produced. "
            f"Every route must have at least {a.min_unique_per_route} binary codes unless "
            "--allow-collapsed-checkpoint is explicitly used for an ablation. "
            "Inspect the history CSV before continuing."
        )
    qualifier = "val" if a.allow_collapsed_checkpoint else "diversity-valid val"
    print(f"best {qualifier} hash mAP={best_map:.4f}; corresponding GT-route val acc={best_acc:.4f} -> {outp}")


if __name__ == "__main__":
    main()
