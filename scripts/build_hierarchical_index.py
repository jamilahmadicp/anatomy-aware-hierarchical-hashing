#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from anatomy_hash.data.manifest import load_manifest, split_df
from anatomy_hash.data.dataset import RadiographDataset
from anatomy_hash.data.transforms import build_transform
from anatomy_hash.checkpoints import load_stage1, load_stage2
from anatomy_hash.evaluation import calibrated_stage1_prob
from anatomy_hash.indexing import choose_routes, pack_codes
from anatomy_hash.training import device_from_arg


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--stage1",required=True); ap.add_argument("--stage2",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--db-splits",nargs="+",default=["train","val"]); ap.add_argument("--policy",choices=["top1","topk","adaptive","oracle"],default="top1"); ap.add_argument("--top-r",type=int,default=2); ap.add_argument("--threshold",type=float,default=0.8); ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--device",default="auto"); a=ap.parse_args()
    device=device_from_arg(a.device); df,_=load_manifest(a.manifest); db=split_df(df,a.db_splits); ds=RadiographDataset(db,build_transform(train=False)); dl=DataLoader(ds,batch_size=a.batch_size,num_workers=a.workers,shuffle=False)
    s1,T,_=load_stage1(a.stage1,device); s2,ck2=load_stage2(a.stage2,device)
    rows=[]
    with torch.no_grad():
        offset=0
        for b in dl:
            x=b["image"].to(device); pa=calibrated_stage1_prob(s1,x,T).cpu().numpy(); feat=s2.encode(x)
            for i in range(len(x)):
                true_a=int(b["anatomy"][i]); routes=choose_routes(pa[i],a.policy,a.top_r,a.threshold,true_a)
                fi=feat[i:i+1]
                for c in routes:
                    route=torch.tensor([c],device=device); o=s2.route_from_features(fi,route); code=o["continuous_code"].sign().cpu().numpy().astype(np.int8)[0]; code[code==0]=1
                    r=db.iloc[offset+i]
                    rows.append((str(r.sample_id),str(r.path),int(c),float(pa[i,c]),int(r.anatomy_id),int(r.fine_id),int(r.joint_id) if "joint_id" in db.columns and not np.isnan(r.joint_id) else -1,int(r.abnormal_label),code))
            offset += len(x)
    codes=np.stack([r[-1] for r in rows]); packed=pack_codes(codes)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,sample_id=np.array([r[0] for r in rows],dtype=object),path=np.array([r[1] for r in rows],dtype=object),route=np.array([r[2] for r in rows],dtype=np.int16),route_prob=np.array([r[3] for r in rows],dtype=np.float32),anatomy_id=np.array([r[4] for r in rows],dtype=np.int16),fine_id=np.array([r[5] for r in rows],dtype=np.int32),joint_id=np.array([r[6] for r in rows],dtype=np.int32),abnormal_label=np.array([r[7] for r in rows],dtype=np.int8),codes=codes,packed=packed,n_bits=np.array(ck2["hash_bits"],dtype=np.int32))
    unique=len(set(r[0] for r in rows)); overhead=len(rows)/max(1,unique); mis=np.mean([r[2]!=r[4] for r in rows if a.policy=="top1"]) if a.policy=="top1" else np.nan
    meta={"db_splits":a.db_splits,"policy":a.policy,"top_r":a.top_r,"threshold":a.threshold,"unique_db_images":unique,"index_entries":len(rows),"index_overhead":overhead,"top1_misindex_fraction":float(mis) if np.isfinite(mis) else None,"bits":int(ck2["hash_bits"])}
    with open(out.with_suffix(".json"),"w") as f: json.dump(meta,f,indent=2)
    print(meta)
if __name__=="__main__": main()
