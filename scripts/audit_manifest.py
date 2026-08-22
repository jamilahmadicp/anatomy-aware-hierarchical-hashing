#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from anatomy_hash.data.manifest import load_manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    df,_=load_manifest(a.manifest); missing=[p for p in df.path if not Path(p).exists()]
    overlap={}
    for idcol in ["sample_id","patient_id","study_id"]:
        if idcol in df and (df[idcol].astype(str)!="").any():
            per=df[df[idcol].astype(str)!=""].groupby(idcol).split.nunique(); overlap[idcol]=int((per>1).sum())
    report={"n_total":len(df),"split_counts":df.split.value_counts().to_dict(),"anatomy_counts":{f"{k[0]}::{k[1]}":int(v) for k,v in df.groupby(["split","anatomy_label"]).size().to_dict().items()},
            "fine_class_count":int(df.fine_id.nunique()),"missing_files":len(missing),"cross_split_id_overlaps":overlap}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out,"w") as f: json.dump(report,f,indent=2,default=lambda x:list(x) if isinstance(x,tuple) else str(x))
    print(report); print("WARNING missing files" if missing else "All files exist")
if __name__=="__main__": main()
