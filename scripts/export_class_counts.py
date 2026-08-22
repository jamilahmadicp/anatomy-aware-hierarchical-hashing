#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from anatomy_hash.data.manifest import load_manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--out-dir",required=True); a=ap.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); df,_=load_manifest(a.manifest)
    anat=df.groupby(["anatomy_label","split"]).size().unstack(fill_value=0); anat["total"]=anat.sum(axis=1); anat.to_csv(out/"anatomy_class_counts.csv")
    fine=df.groupby(["fine_label","anatomy_label","split"]).size().unstack(fill_value=0); fine["total"]=fine.sum(axis=1); fine.to_csv(out/"fine_class_counts.csv")
    if "joint_label" in df.columns:
        joint=df.groupby(["joint_label","split"]).size().unstack(fill_value=0); joint["total"]=joint.sum(axis=1); joint.to_csv(out/"joint_class_counts.csv")
    print(anat)
if __name__=="__main__": main()
