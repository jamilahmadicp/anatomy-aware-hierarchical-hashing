#!/usr/bin/env python
from __future__ import annotations
import argparse, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--protocol", default="prototype_margin_hash_v3")
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    got = ck.get("hash_protocol", "")
    if got != a.protocol:
        print(f"INCOMPATIBLE checkpoint protocol: {got or '<legacy>'}; required: {a.protocol}")
        raise SystemExit(2)
    if "hash_codebooks" not in ck:
        print("INCOMPATIBLE checkpoint: route-specific hash codebooks are missing")
        raise SystemExit(3)
    print(
        f"Checkpoint protocol OK: {got}; selection={ck.get('selection_metric','<unknown>')}; "
        f"best_val_hash_mAP={ck.get('best_val_hash_mAP','<unknown>')}"
    )


if __name__ == "__main__":
    main()
