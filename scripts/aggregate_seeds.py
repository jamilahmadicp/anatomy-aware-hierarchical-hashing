#!/usr/bin/env python
from __future__ import annotations
import argparse, glob
import pandas as pd
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--glob",required=True,help='e.g. "outputs/seeds/*/retrieval_summary.csv"'); ap.add_argument("--out",required=True); a=ap.parse_args(); frames=[]
    for p in glob.glob(a.glob):
        d=pd.read_csv(p); d["source"]=p; frames.append(d)
    if not frames: raise SystemExit("No matching files")
    x=pd.concat(frames,ignore_index=True); num=x.select_dtypes("number").columns; rows=[]
    for c in num: rows.append({"metric":c,"mean":x[c].mean(),"std":x[c].std(ddof=1),"n_runs":x[c].notna().sum()})
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.out,index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
