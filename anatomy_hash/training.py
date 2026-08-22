from __future__ import annotations
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def class_weights(labels, n_classes=None):
    y=np.asarray(labels,dtype=int); n_classes=int(n_classes or (y.max()+1))
    counts=np.bincount(y,minlength=n_classes).astype(np.float64); counts[counts==0]=1
    w=len(y)/(n_classes*counts)
    return torch.tensor(w,dtype=torch.float32)


def make_optimizer(model, lr=3e-4, weight_decay=0.05):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, epochs, warmup_epochs=2):
    def f(ep):
        if ep < warmup_epochs: return float(ep+1)/max(1,warmup_epochs)
        t=(ep-warmup_epochs)/max(1,epochs-warmup_epochs)
        return 0.5*(1+math.cos(math.pi*t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,f)


def device_from_arg(name="auto"):
    if name=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def amp_context(device, enabled=True):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(enabled and device.type=="cuda"))
