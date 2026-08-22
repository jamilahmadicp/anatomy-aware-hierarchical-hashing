from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel


def lsh_fit_transform(x_train, x_all, bits=128, seed=42):
    rng=np.random.default_rng(seed); W=rng.normal(size=(x_train.shape[1], bits)).astype(np.float32)
    return np.where(x_all @ W >= 0, 1, -1).astype(np.int8)


def spectral_hashing_pca(x_train, x_all, bits=128, seed=42):
    ncomp=min(bits, x_train.shape[1], max(1, x_train.shape[0]-1))
    pca=PCA(n_components=ncomp, random_state=seed).fit(x_train)
    tr=pca.transform(x_train); z=pca.transform(x_all); med=np.median(tr, axis=0)
    b=np.where(z >= med,1,-1).astype(np.int8)
    if ncomp<bits: b=np.pad(b,((0,0),(0,bits-ncomp)),constant_values=1)
    return b


def klsh_anchor_rbf(x_train, x_all, bits=128, n_anchors=300, gamma=None, seed=42):
    rng=np.random.default_rng(seed); idx=rng.choice(len(x_train), min(n_anchors,len(x_train)), replace=False); anchors=x_train[idx]
    if gamma is None: gamma=1.0/max(1,x_train.shape[1])
    Ktr=rbf_kernel(x_train,anchors,gamma=gamma); Kall=rbf_kernel(x_all,anchors,gamma=gamma)
    return spectral_hashing_pca(Ktr,Kall,bits=bits,seed=seed)
