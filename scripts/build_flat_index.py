#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from anatomy_hash.data.manifest import load_manifest, split_df
from anatomy_hash.data.dataset import RadiographDataset
from anatomy_hash.data.transforms import build_transform
from anatomy_hash.checkpoints import load_flat_hash
from anatomy_hash.indexing import pack_codes
from anatomy_hash.training import device_from_arg

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--out",required=True); ap.add_argument("--db-splits",nargs="+",default=["train","val"]); ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--device",default="auto"); a=ap.parse_args()
    df,_=load_manifest(a.manifest); db=split_df(df,a.db_splits); device=device_from_arg(a.device); model,ck=load_flat_hash(a.checkpoint,device); dl=DataLoader(RadiographDataset(db,build_transform(train=False)),batch_size=a.batch_size,num_workers=a.workers)
    codes=[]
    with torch.no_grad():
        for b in dl:
            q=model(b["image"].to(device))["continuous_code"].sign().cpu().numpy().astype(np.int8); q[q==0]=1; codes.append(q)
    codes=np.concatenate(codes); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,sample_id=db.sample_id.astype(str).values,path=db.path.astype(str).values,route=np.full(len(db),-1,dtype=np.int16),route_prob=np.ones(len(db),dtype=np.float32),anatomy_id=db.anatomy_id.values.astype(np.int16),fine_id=db.fine_id.values.astype(np.int32),joint_id=(db.joint_id.values.astype(np.int32) if "joint_id" in db else np.full(len(db),-1,dtype=np.int32)),abnormal_label=db.abnormal_label.values.astype(np.int8),codes=codes,packed=pack_codes(codes),n_bits=np.array(ck["hash_bits"],dtype=np.int32))
    print(f"Wrote {len(db)} flat index entries to {out}")
if __name__=="__main__": main()
