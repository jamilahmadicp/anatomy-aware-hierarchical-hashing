#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from anatomy_hash.data.manifest import load_manifest, split_df
from anatomy_hash.data.dataset import RadiographDataset
from anatomy_hash.data.transforms import build_transform
from anatomy_hash.checkpoints import load_stage1
from anatomy_hash.models.stage1 import fit_temperature
from anatomy_hash.metrics import expected_calibration_error
from anatomy_hash.training import device_from_arg
from anatomy_hash.plotting import save_confusion

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--batch-size",type=int,default=128); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--device",default="auto"); a=ap.parse_args()
    device=device_from_arg(a.device); df,_=load_manifest(a.manifest); va=split_df(df,"val"); ds=RadiographDataset(va,build_transform(train=False)); dl=DataLoader(ds,batch_size=a.batch_size,num_workers=a.workers)
    model,T0,ck=load_stage1(a.checkpoint,device); logits=[]; ys=[]
    with torch.no_grad():
        for b in dl:
            lg=model(b["image"].to(device)); logits.append(lg); ys.append(b["anatomy"].to(device))
    logits=torch.cat(logits); ys=torch.cat(ys); T=fit_temperature(logits,ys); before=F.softmax(logits,1).cpu().numpy(); after=F.softmax(logits/T,1).cpu().numpy()
    e0,_=expected_calibration_error(ys.cpu().numpy(),before); e1,details=expected_calibration_error(ys.cpu().numpy(),after)
    ck["temperature"]=T; torch.save(ck,a.checkpoint)
    pd.DataFrame(details,columns=["bin_lo","bin_hi","accuracy","confidence","n"]).to_csv(Path(a.checkpoint).with_suffix(".calibration.csv"),index=False)
    print(f"Temperature={T:.6f}; ECE before={e0:.5f}, after={e1:.5f}")
if __name__=="__main__": main()
