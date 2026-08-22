#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from anatomy_hash.ablation import ABLATION_PROFILES


def run_command(args: list[str], expected: Path | None = None, force: bool = False) -> None:
    if expected is not None and expected.exists() and not force:
        print(f"[skip] {expected}")
        return
    print("[run]", shlex.join([str(x) for x in args]))
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(args, check=True, env=env)


def py(script: str, *args: object) -> list[str]:
    return [sys.executable, script, *[str(x) for x in args]]


def train_hierarchical(
    manifest: str,
    out: Path,
    profile: str,
    bits: int,
    epochs: int,
    seed: int,
    device: str,
    workers: int,
    force: bool,
) -> None:
    cmd = py(
        "scripts/train_stage2.py",
        "--manifest", manifest,
        "--out", out,
        "--bits", bits,
        "--epochs", epochs,
        "--seed", seed,
        "--device", device,
        "--workers", workers,
        "--ablation-profile", profile,
    )
    run_command(cmd, expected=out, force=force)


def build_hierarchical_index(
    manifest: str,
    stage1: str,
    stage2: str | Path,
    out: Path,
    policy: str,
    top_r: int,
    threshold: float,
    device: str,
    workers: int,
    force: bool,
) -> None:
    cmd = py(
        "scripts/build_hierarchical_index.py",
        "--manifest", manifest,
        "--stage1", stage1,
        "--stage2", stage2,
        "--out", out,
        "--policy", policy,
        "--top-r", top_r,
        "--threshold", threshold,
        "--device", device,
        "--workers", workers,
    )
    run_command(cmd, expected=out, force=force)


def evaluate_hierarchical(
    manifest: str,
    stage1: str,
    stage2: str | Path,
    index: Path,
    out_dir: Path,
    query_policy: str,
    top_r: int,
    threshold: float,
    alpha: float,
    relevance_col: str,
    query_split: str,
    device: str,
    workers: int,
    force: bool,
) -> None:
    summary = out_dir / "retrieval_summary.csv"
    cmd = py(
        "scripts/evaluate_retrieval.py",
        "--manifest", manifest,
        "--stage1", stage1,
        "--stage2", stage2,
        "--index", index,
        "--out-dir", out_dir,
        "--query-policy", query_policy,
        "--top-r", top_r,
        "--threshold", threshold,
        "--alpha", alpha,
        "--relevance-col", relevance_col,
        "--query-split", query_split,
        "--device", device,
        "--workers", workers,
    )
    run_command(cmd, expected=summary, force=force)


def audit_index(index: Path, out: Path, force: bool) -> None:
    cmd = py(
        "scripts/audit_hash_codes.py",
        "--index", index,
        "--out", out,
        "--min-unique-per-route", 2,
    )
    run_command(cmd, expected=out, force=force)


def component_ablations(a, root: Path, full_stage2: Path) -> None:
    comp = root / "component"
    comp.mkdir(parents=True, exist_ok=True)

    # Flat model: same Swin family and fine-grained labels, but no anatomy routing.
    flat = comp / "flat_no_routing"
    flat.mkdir(parents=True, exist_ok=True)
    flat_ck = flat / "model.pt"
    run_command(
        py(
            "scripts/train_deep_baseline.py",
            "--manifest", a.manifest,
            "--out", flat_ck,
            "--method", "flat_hash",
            "--bits", a.bits,
            "--epochs", a.epochs,
            "--seed", a.seed,
            "--device", a.device,
            "--workers", a.workers,
        ),
        expected=flat_ck,
        force=a.force,
    )
    flat_idx = flat / "index.npz"
    run_command(
        py(
            "scripts/build_flat_index.py",
            "--manifest", a.manifest,
            "--checkpoint", flat_ck,
            "--out", flat_idx,
            "--device", a.device,
            "--workers", a.workers,
        ),
        expected=flat_idx,
        force=a.force,
    )
    run_command(
        py(
            "scripts/evaluate_flat_retrieval.py",
            "--manifest", a.manifest,
            "--checkpoint", flat_ck,
            "--index", flat_idx,
            "--out-dir", flat / "retrieval",
            "--query-split", a.query_split,
            "--relevance-col", a.relevance_col,
            "--device", a.device,
            "--workers", a.workers,
        ),
        expected=flat / "retrieval" / "retrieval_summary.csv",
        force=a.force,
    )

    profiles = [
        "full",
        "shared_head",
        "no_embedding_supcon",
        "no_prototype_sign_margin",
        "no_semantic_hash",
        "no_quant_balance",
    ]
    for profile in profiles:
        d = comp / profile
        d.mkdir(parents=True, exist_ok=True)
        if profile == "full" and full_stage2.exists():
            ck = full_stage2
            print(f"[reuse] full Stage-2 checkpoint: {ck}")
        else:
            ck = d / "stage2.pt"
            train_hierarchical(
                a.manifest, ck, profile, a.bits, a.epochs, a.seed,
                a.device, a.workers, a.force,
            )
        idx = d / "index_adaptive.npz"
        build_hierarchical_index(
            a.manifest, a.stage1, ck, idx, "adaptive", a.top_r,
            a.threshold, a.device, a.workers, a.force,
        )
        audit_index(idx, d / "hash_audit.json", a.force)
        evaluate_hierarchical(
            a.manifest, a.stage1, ck, idx, d / "retrieval", "topk",
            a.top_r, a.threshold, a.alpha, a.relevance_col, a.query_split,
            a.device, a.workers, a.force,
        )


def routing_ablations(a, root: Path, full_stage2: Path) -> tuple[Path, Path]:
    routing = root / "routing"
    routing.mkdir(parents=True, exist_ok=True)
    if not full_stage2.exists():
        full_stage2 = routing / "full_stage2.pt"
        train_hierarchical(
            a.manifest, full_stage2, "full", a.bits, a.epochs, a.seed,
            a.device, a.workers, a.force,
        )

    idx_top1 = routing / "index_top1.npz"
    idx_adaptive = routing / "index_adaptive.npz"
    idx_oracle = routing / "index_oracle.npz"
    build_hierarchical_index(a.manifest, a.stage1, full_stage2, idx_top1, "top1", 1, a.threshold, a.device, a.workers, a.force)
    build_hierarchical_index(a.manifest, a.stage1, full_stage2, idx_adaptive, "adaptive", a.top_r, a.threshold, a.device, a.workers, a.force)
    build_hierarchical_index(a.manifest, a.stage1, full_stage2, idx_oracle, "oracle", 1, a.threshold, a.device, a.workers, a.force)

    specs = [
        ("top1_db_top1_query", idx_top1, "top1", 1),
        ("top1_db_top2_query", idx_top1, "topk", 2),
        ("adaptive_db_top2_query", idx_adaptive, "topk", a.top_r),
        ("oracle_db_oracle_query", idx_oracle, "oracle", 1),
    ]
    for label, idx, policy, top_r in specs:
        evaluate_hierarchical(
            a.manifest, a.stage1, full_stage2, idx, routing / label,
            policy, top_r, a.threshold, a.alpha, a.relevance_col,
            a.query_split, a.device, a.workers, a.force,
        )
    return idx_top1, idx_adaptive


def routing_robustness(a, root: Path, full_stage2: Path, idx_top1: Path, idx_adaptive: Path) -> None:
    out = root / "routing_error_sensitivity"
    expected = out / "routing_robustness.csv"
    cmd = py(
        "scripts/run_routing_robustness.py",
        "--manifest", a.manifest,
        "--stage1", a.stage1,
        "--stage2", full_stage2,
        "--index-top1", idx_top1,
        "--index-adaptive", idx_adaptive,
        "--out-dir", out,
        "--query-split", a.query_split,
        "--relevance-col", a.relevance_col,
        "--alpha", a.alpha,
        "--rates", 0, 0.05, 0.10, 0.15, 0.20, 0.30,
        "--seed", a.seed,
        "--device", a.device,
        "--workers", a.workers,
    )
    run_command(cmd, expected=expected, force=a.force)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the component, routing-policy, and routing-error ablations used in the manuscript."
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stage1", required=True, help="Calibrated Stage-1 checkpoint.")
    ap.add_argument("--full-stage2", default="", help="Optional existing full Stage-2 checkpoint to reuse.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--query-split", default="test")
    ap.add_argument("--relevance-col", choices=["fine_id", "joint_id", "anatomy_id"], default="fine_id")
    ap.add_argument("--top-r", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--skip-components", action="store_true")
    ap.add_argument("--skip-routing", action="store_true")
    ap.add_argument("--skip-routing-error", action="store_true")
    ap.add_argument("--force", action="store_true", help="Recompute outputs even if target files already exist.")
    a = ap.parse_args()

    root = Path(a.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    full_stage2 = Path(a.full_stage2) if a.full_stage2 else root / "component" / "full" / "stage2.pt"

    if not a.skip_components:
        component_ablations(a, root, full_stage2)

    if not full_stage2.exists():
        full_stage2.parent.mkdir(parents=True, exist_ok=True)
        train_hierarchical(
            a.manifest, full_stage2, "full", a.bits, a.epochs, a.seed,
            a.device, a.workers, a.force,
        )

    idx_top1 = root / "routing" / "index_top1.npz"
    idx_adaptive = root / "routing" / "index_adaptive.npz"
    if not a.skip_routing:
        idx_top1, idx_adaptive = routing_ablations(a, root, full_stage2)

    if not a.skip_routing_error:
        if not idx_top1.exists() or not idx_adaptive.exists():
            idx_top1, idx_adaptive = routing_ablations(a, root, full_stage2)
        routing_robustness(a, root, full_stage2, idx_top1, idx_adaptive)

    run_command(
        py("scripts/aggregate_ablation_results.py", "--root", root, "--out-dir", root / "tables"),
        expected=root / "tables" / "component_ablation.csv",
        force=True,
    )
    print(f"Ablation suite complete: {root}")


if __name__ == "__main__":
    main()
