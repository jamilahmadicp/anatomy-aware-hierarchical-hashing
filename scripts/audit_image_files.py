#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError


def is_appledouble(path: str) -> bool:
    p = Path(path)
    return p.name.startswith("._") or any(part.startswith("._") for part in p.parts)


def main():
    ap = argparse.ArgumentParser(description="Validate all image paths in a manifest before training.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean-manifest", default=None,
                    help="Optional CSV written after removing macOS AppleDouble sidecar rows only.")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if "path" not in df.columns:
        raise SystemExit("Manifest has no 'path' column.")

    apple_mask = df["path"].astype(str).map(is_appledouble)
    apple_rows = df.loc[apple_mask, [c for c in ["path","sample_id","split"] if c in df.columns]].copy()
    clean = df.loc[~apple_mask].reset_index(drop=True)

    missing = []
    unreadable = []
    for path in clean["path"].astype(str):
        p = Path(path)
        if not p.exists():
            missing.append(path)
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except (UnidentifiedImageError, OSError, ValueError) as e:
            unreadable.append({"path": path, "error": f"{type(e).__name__}: {e}"})

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "rows_original": int(len(df)),
        "appledouble_rows_removed": int(apple_mask.sum()),
        "rows_after_appledouble_filter": int(len(clean)),
        "missing_files": int(len(missing)),
        "unreadable_real_images": int(len(unreadable)),
        "appledouble_examples": apple_rows.head(20).to_dict(orient="records"),
        "missing_examples": missing[:20],
        "unreadable_examples": unreadable[:20],
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.clean_manifest:
        cp = Path(args.clean_manifest); cp.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(cp, index=False)
        print(f"Cleaned manifest written to {cp} (removed only {int(apple_mask.sum())} AppleDouble row(s)).")

    print(json.dumps(report, indent=2))
    if missing or unreadable:
        raise SystemExit(
            "Image audit failed: the cleaned manifest still contains missing or unreadable real image files. "
            "See the JSON audit report for paths."
        )
    print("Image audit PASSED: all non-AppleDouble manifest images are present and PIL-readable.")


if __name__ == "__main__":
    main()
