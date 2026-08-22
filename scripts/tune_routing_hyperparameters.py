#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


def run(cmd):
    print("RUN:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description="Select routing parameters on a validation split only."
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--routes", nargs="+", type=int, default=[1, 2, 3], help="Query-time top-r values.")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.025, 0.05, 0.1])
    ap.add_argument("--db-top-r", type=int, default=2, help="Maximum database routes for uncertain images.")
    ap.add_argument("--db-splits", nargs="+", default=["train"])
    ap.add_argument("--query-split", default="val")
    ap.add_argument("--relevance-col", choices=["fine_id", "joint_id", "anatomy_id"], default="fine_id")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    wd = Path(a.work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    sd = Path(__file__).resolve().parent
    rows = []

    for threshold in a.thresholds:
        idx = wd / f"index_threshold_{threshold:.2f}_dbtop{a.db_top_r}.npz"
        run([
            sys.executable, str(sd / "build_hierarchical_index.py"),
            "--manifest", a.manifest,
            "--stage1", a.stage1,
            "--stage2", a.stage2,
            "--out", str(idx),
            "--db-splits", *a.db_splits,
            "--policy", "adaptive",
            "--top-r", str(a.db_top_r),
            "--threshold", str(threshold),
            "--device", a.device,
            "--workers", str(a.workers),
        ])
        for r in a.routes:
            for alpha in a.alphas:
                od = wd / f"th{threshold:.2f}_r{r}_a{alpha:.3f}"
                run([
                    sys.executable, str(sd / "evaluate_retrieval.py"),
                    "--manifest", a.manifest,
                    "--stage1", a.stage1,
                    "--stage2", a.stage2,
                    "--index", str(idx),
                    "--out-dir", str(od),
                    "--query-split", a.query_split,
                    "--query-policy", "topk",
                    "--top-r", str(r),
                    "--threshold", str(threshold),
                    "--alpha", str(alpha),
                    "--relevance-col", a.relevance_col,
                    "--device", a.device,
                    "--workers", str(a.workers),
                ])
                result = pd.read_csv(od / "retrieval_summary.csv").iloc[0].to_dict()
                rows.append({
                    "threshold": threshold,
                    "db_top_r": a.db_top_r,
                    "query_r": r,
                    "alpha": alpha,
                    **result,
                })

    table = pd.DataFrame(rows).sort_values(["mAP", "p@20"], ascending=False)
    table.to_csv(wd / "routing_grid.csv", index=False)
    best = table.iloc[0]
    pd.DataFrame([best]).to_csv(wd / "selected_routing_hyperparameters.csv", index=False)
    print("\nBEST VALIDATION SETTING\n", best.to_string())


if __name__ == "__main__":
    main()
