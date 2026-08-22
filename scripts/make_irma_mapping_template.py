#!/usr/bin/env python
from pathlib import Path
import argparse, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train-dir",required=True); ap.add_argument("--out",default="templates/irma_fine_to_anatomy.csv"); a=ap.parse_args()
    root=Path(a.train_dir); classes=sorted([d.name for d in root.iterdir() if d.is_dir()])
    pd.DataFrame({"fine_label":classes,"anatomy_label":[""]*len(classes)}).to_csv(a.out,index=False)
    print(f"Wrote {a.out}. Fill anatomy_label for every fine class, then run prepare_irma_manifest.py")
if __name__=="__main__": main()
