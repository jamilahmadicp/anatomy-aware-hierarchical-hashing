#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

ANATS=["SHOULDER","HUMERUS","ELBOW","FOREARM","WRIST","HAND","FINGER"]

def scan(root):
    rows=[]; root=Path(root); skipped_appledouble=0
    for split_dir in ["train","valid","val","test"]:
        d=root/split_dir
        if not d.exists(): continue
        source_split="official_valid" if split_dir in {"valid","val","test"} else "official_train"
        for p in d.rglob("*.png"):
            # macOS archive/resource-fork sidecars such as ._image1.png are not images.
            if p.name.startswith("._") or any(part.startswith("._") for part in p.parts):
                skipped_appledouble += 1
                continue
            parts=p.parts
            anat=next((a for a in ANATS if f"XR_{a}" in parts),None)
            if anat is None: continue
            study=next((x for x in parts if x.startswith("study")),"")
            patient=next((x for x in parts if x.startswith("patient")),"")
            abnormal=1 if "positive" in study else 0
            al=f"XR_{anat}"; fine=f"{al}_{'positive' if abnormal else 'negative'}"
            rows.append(dict(path=str(p.resolve()),source_split=source_split,anatomy_label=al,fine_label=fine,joint_label=fine,
                             abnormal_label=abnormal,patient_id=patient,study_id=f"{patient}/{study}",sample_id=str(p.relative_to(root))))
    df = pd.DataFrame(rows)
    df.attrs["skipped_appledouble"] = skipped_appledouble
    return df

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--out",required=True); ap.add_argument("--val-fraction",type=float,default=0.1); ap.add_argument("--seed",type=int,default=42); a=ap.parse_args()
    df=scan(a.root)
    skipped_appledouble = int(df.attrs.get("skipped_appledouble", 0))
    if df.empty: raise SystemExit("No MURA PNG files found. Expected MURA-v1.1/{train,valid}/XR_*/patient*/study*/*.png")
    tr=df[df.source_split=="official_train"].copy(); te=df[df.source_split=="official_valid"].copy(); te["split"]="test"
    # Patient-disjoint internal validation carved only from official training data.
    n_splits=max(2,round(1/a.val_fraction))
    try:
        sg=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=a.seed)
        ti,vi=next(sg.split(tr, y=tr.joint_label, groups=tr.patient_id))
    except Exception:
        gs=GroupShuffleSplit(n_splits=1,test_size=a.val_fraction,random_state=a.seed); ti,vi=next(gs.split(tr,groups=tr.patient_id))
    tr.iloc[ti,tr.columns.get_loc("source_split")]=tr.iloc[ti].source_split
    tr["split"]="train"; tr.iloc[vi,tr.columns.get_loc("split")]= "val"
    df=pd.concat([tr,te],ignore_index=True)
    for col,out in [("anatomy_label","anatomy_id"),("fine_label","fine_id"),("joint_label","joint_id")]:
        mp={v:i for i,v in enumerate(sorted(df[col].unique()))}; df[out]=df[col].map(mp)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
    print(df.groupby(["split","anatomy_label","abnormal_label"]).size().to_string())
    print(f"Skipped {skipped_appledouble} macOS AppleDouble sidecar file(s) (._*.png).")
    print(f"Wrote {len(df)} rows to {a.out}")
    for idcol in ["patient_id","study_id"]:
        ov=df.groupby(idcol).split.nunique(); print(f"Cross-split {idcol} overlaps: {(ov>1).sum()}")
if __name__=="__main__": main()
