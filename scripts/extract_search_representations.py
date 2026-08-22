#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from anatomy_hash.data.manifest import load_manifest, split_df
from anatomy_hash.data.dataset import RadiographDataset
from anatomy_hash.data.transforms import build_transform
from anatomy_hash.checkpoints import load_stage1, load_stage2
from anatomy_hash.evaluation import calibrated_stage1_prob
from anatomy_hash.training import device_from_arg

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--stage1",required=True); ap.add_argument("--stage2",required=True); ap.add_argument("--splits",nargs="+",default=["train","val"]); ap.add_argument("--out",required=True); ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--device",default="auto"); a=ap.parse_args()
    df,_=load_manifest(a.manifest); d=split_df(df,a.splits); device=device_from_arg(a.device); s1,T,_=load_stage1(a.stage1,device); s2,ck=load_stage2(a.stage2,device); dl=DataLoader(RadiographDataset(d,build_transform(train=False)),batch_size=a.batch_size,num_workers=a.workers)
    feats=[]; probs=[]; codes=[]
    with torch.no_grad():
        for b in dl:
            x=b["image"].to(device); p=calibrated_stage1_prob(s1,x,T); f=s2.encode(x); route=p.argmax(1); o=s2.route_from_features(f,route); q=o["continuous_code"].sign(); q[q==0]=1
            feats.append(f.cpu().numpy().astype(np.float32)); probs.append(p.cpu().numpy().astype(np.float32)); codes.append(q.cpu().numpy().astype(np.int8))
    np.savez_compressed(a.out,sample_id=d.sample_id.astype(str).values,path=d.path.astype(str).values,anatomy_id=d.anatomy_id.values.astype(np.int16),fine_id=d.fine_id.values.astype(np.int32),features=np.concatenate(feats),anatomy_prob=np.concatenate(probs),codes=np.concatenate(codes))
    print(f"Saved {len(d)} representations to {a.out}")
if __name__=="__main__": main()
