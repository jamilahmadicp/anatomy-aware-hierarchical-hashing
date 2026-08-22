#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return pd.read_csv(path).iloc[0].to_dict()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    component_rows = []
    comp = root / "component"
    if comp.exists():
        for d in sorted(p for p in comp.iterdir() if p.is_dir()):
            summary = read_summary(d / "retrieval" / "retrieval_summary.csv")
            if not summary:
                continue
            audit = read_json(d / "hash_audit.json")
            index_meta = read_json(d / "index_adaptive.json")
            component_rows.append({
                "configuration": d.name,
                **summary,
                "unique_codes": audit.get("unique_codes"),
                "collapsed_routes": ",".join(map(str, audit.get("collapsed_routes", []))) if audit else "",
                "hash_audit_pass": audit.get("pass") if audit else None,
                "index_overhead": index_meta.get("index_overhead"),
            })
    if component_rows:
        pd.DataFrame(component_rows).to_csv(out / "component_ablation.csv", index=False)

    routing_rows = []
    routing = root / "routing"
    if routing.exists():
        for label in [
            "top1_db_top1_query",
            "top1_db_top2_query",
            "adaptive_db_top2_query",
            "oracle_db_oracle_query",
        ]:
            summary = read_summary(routing / label / "retrieval_summary.csv")
            if not summary:
                continue
            if label.startswith("adaptive"):
                meta = read_json(routing / "index_adaptive.json")
            elif label.startswith("oracle"):
                meta = read_json(routing / "index_oracle.json")
            else:
                meta = read_json(routing / "index_top1.json")
            routing_rows.append({"configuration": label, **summary, "index_overhead": meta.get("index_overhead")})
    if routing_rows:
        pd.DataFrame(routing_rows).to_csv(out / "routing_policy_ablation.csv", index=False)

    src = root / "routing_error_sensitivity" / "routing_robustness.csv"
    if src.exists():
        pd.read_csv(src).to_csv(out / "routing_error_sensitivity.csv", index=False)

    print(f"Ablation tables written to {out}")


if __name__ == "__main__":
    main()
