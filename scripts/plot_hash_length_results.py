#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv",required=True); ap.add_argument("--out",required=True); a=ap.parse_args(); df=pd.read_csv(a.csv).sort_values("bits")
    fig,ax=plt.subplots(figsize=(7,5))
    for c in [x for x in ["mAP","p@20","r@20","ndcg@20"] if x in df.columns]: ax.plot(df.bits,df[c],marker="o",label=c)
    ax.set_xlabel("Hash code length (bits)"); ax.set_ylabel("Retrieval metric"); ax.set_xticks(df.bits); ax.legend(); fig.tight_layout(); Path(a.out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.out,dpi=300); plt.close(fig)
if __name__=="__main__": main()
