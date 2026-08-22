from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED = ["path", "split", "anatomy_label", "fine_label"]


def _encode_sorted(df: pd.DataFrame, source_col: str, out_col: str):
    if source_col not in df.columns:
        return df, {}
    vals = sorted(df[source_col].astype(str).dropna().unique().tolist())
    mapping = {v: i for i, v in enumerate(vals)}
    df[out_col] = df[source_col].astype(str).map(mapping).astype(int)
    return df, mapping


def load_manifest(path: str | Path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    df = df.copy()
    df["path"] = df["path"].astype(str)
    maps = {}
    if "anatomy_id" not in df.columns:
        df, maps["anatomy"] = _encode_sorted(df, "anatomy_label", "anatomy_id")
    else:
        maps["anatomy"] = dict(df[["anatomy_label", "anatomy_id"]].drop_duplicates().astype({"anatomy_label": str}).values)
    if "fine_id" not in df.columns:
        df, maps["fine"] = _encode_sorted(df, "fine_label", "fine_id")
    else:
        maps["fine"] = dict(df[["fine_label", "fine_id"]].drop_duplicates().astype({"fine_label": str}).values)
    if "joint_label" in df.columns and "joint_id" not in df.columns:
        df, maps["joint"] = _encode_sorted(df, "joint_label", "joint_id")
    elif "joint_id" in df.columns:
        maps["joint"] = dict(df[["joint_label", "joint_id"]].drop_duplicates().astype({"joint_label": str}).values) if "joint_label" in df.columns else {}
    if "sample_id" not in df.columns:
        df["sample_id"] = np.arange(len(df)).astype(str)
    else:
        df["sample_id"] = df["sample_id"].astype(str)
    for col in ["patient_id", "study_id", "irma_code"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if "abnormal_label" not in df.columns:
        df["abnormal_label"] = -1
    return df, maps


def build_anatomy_fine_maps(df: pd.DataFrame):
    out = {}
    for a, g in df.groupby("anatomy_id"):
        fine_ids = sorted(g["fine_id"].astype(int).unique().tolist())
        out[int(a)] = {
            "global_ids": fine_ids,
            "global_to_local": {int(fid): i for i, fid in enumerate(fine_ids)},
            "local_to_global": {i: int(fid) for i, fid in enumerate(fine_ids)},
        }
    return out


def split_df(df, split):
    splits = {split} if isinstance(split, str) else set(split)
    return df[df["split"].isin(splits)].reset_index(drop=True)
