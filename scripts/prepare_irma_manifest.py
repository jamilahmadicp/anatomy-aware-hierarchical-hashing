#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

EXT={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def scan(root, split):
    rows=[]; root=Path(root)
    for c in sorted([d for d in root.iterdir() if d.is_dir()]):
        for p in c.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXT:
                rows.append(dict(path=str(p.resolve()),split=split,fine_label=c.name,sample_id=f"{split}/{c.name}/{p.name}"))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-dir",required=True); ap.add_argument("--test-dir",required=True); ap.add_argument("--mapping-csv",required=True)
    ap.add_argument("--val-fraction",type=float,default=0.1); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--out",required=True)
    a=ap.parse_args(); tr=scan(a.train_dir,"source_train"); te=scan(a.test_dir,"test")
    mp=pd.read_csv(a.mapping_csv,dtype=str)
    if not {"fine_label","anatomy_label"}.issubset(mp.columns): raise SystemExit("mapping CSV needs fine_label, anatomy_label")
    if mp.anatomy_label.isna().any() or (mp.anatomy_label.str.strip()=="").any(): raise SystemExit("Fill all anatomy_label cells in mapping CSV")
    tr=tr.merge(mp,on="fine_label",how="left"); te=te.merge(mp,on="fine_label",how="left")
    if tr.anatomy_label.isna().any() or te.anatomy_label.isna().any(): raise SystemExit("Some image fine classes are missing from mapping CSV")
    # stratify by fine class when feasible; otherwise by anatomy.
    strat=tr.fine_label if tr.groupby("fine_label").size().min()>=2 else tr.anatomy_label
    idx_train,idx_val=train_test_split(np.arange(len(tr)),test_size=a.val_fraction,random_state=a.seed,stratify=strat)
    tr.loc[idx_train,"split"]="train"; tr.loc[idx_val,"split"]="val"
    df=pd.concat([tr,te],ignore_index=True)
    for col,out in [("anatomy_label","anatomy_id"),("fine_label","fine_id")]:
        enc={v:i for i,v in enumerate(sorted(df[col].unique()))}; df[out]=df[col].map(enc)
    df["patient_id"]=""; df["study_id"]=""; df["abnormal_label"]=-1
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
    print(df.groupby(["split","anatomy_label"]).size().to_string()); print(f"Wrote {len(df)} rows to {a.out}")
if __name__=="__main__": main()
