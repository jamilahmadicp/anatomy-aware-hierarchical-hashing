#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-unique-per-route", type=int, default=2)
    ap.add_argument("--fail-on-collapse", action="store_true")
    a = ap.parse_args()

    z = np.load(a.index, allow_pickle=True)
    codes = z["codes"].astype(np.int8)
    routes = z["route"].astype(int)
    true_anat = z["anatomy_id"].astype(int) if "anatomy_id" in z.files else routes.copy()
    fine = z["fine_id"].astype(int) if "fine_id" in z.files else np.full(len(codes), -1, dtype=int)

    unique_global = len(np.unique(codes, axis=0))
    report = {
        "index": str(a.index),
        "n_entries": int(len(codes)),
        "n_bits": int(z["n_bits"]),
        "unique_codes": int(unique_global),
        "unique_ratio": float(unique_global / max(1, len(codes))),
        "mean_abs_bit_mean": float(np.abs(codes.mean(axis=0)).mean()),
        "plus_one_fraction": float((codes == 1).mean()),
        "routes": {},
    }

    collapsed = []
    for r in sorted(np.unique(routes)):
        all_mask = routes == r
        native_mask = all_mask & (true_anat == r)
        c_all = codes[all_mask]
        c_native = codes[native_mask]
        labels_native = fine[native_mask]

        u_all = len(np.unique(c_all, axis=0)) if len(c_all) else 0
        u_native = len(np.unique(c_native, axis=0)) if len(c_native) else 0
        n_classes_native = len(np.unique(labels_native)) if len(labels_native) else 0

        per_class = {}
        for cls in sorted(np.unique(labels_native)) if len(labels_native) else []:
            cc = c_native[labels_native == cls]
            per_class[str(int(cls))] = {
                "entries": int(len(cc)),
                "unique_codes": int(len(np.unique(cc, axis=0))),
            }

        report["routes"][str(int(r))] = {
            "entries_all": int(len(c_all)),
            "unique_codes_all": int(u_all),
            "entries_native": int(len(c_native)),
            "native_fine_classes": int(n_classes_native),
            "unique_codes_native": int(u_native),
            "native_unique_ratio": float(u_native / max(1, len(c_native))),
            "per_native_fine_class": per_class,
        }

        # The safety decision is based on correctly routed/native images only.
        # Adaptive indexing may add foreign images to a route; those must not be
        # allowed to hide collapse of the route's own fine-grained classes.
        required = min(a.min_unique_per_route, max(1, n_classes_native))
        if n_classes_native >= 2 and u_native < required:
            collapsed.append(int(r))

    report["collapsed_routes"] = collapsed
    report["pass"] = len(collapsed) == 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if a.fail_on_collapse and collapsed:
        raise SystemExit(
            f"Hash collapse detected in native route(s): {collapsed}. "
            f"Each route with >=2 fine classes must have at least {a.min_unique_per_route} native binary codes."
        )


if __name__ == "__main__":
    main()
