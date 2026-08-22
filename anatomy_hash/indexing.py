from __future__ import annotations
import numpy as np

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def sign_code(q):
    b = np.where(np.asarray(q) >= 0, 1, -1).astype(np.int8)
    return b


def pack_codes(codes):
    bits = (np.asarray(codes) > 0).astype(np.uint8)
    return np.packbits(bits, axis=1)


def hamming_distance_packed(query_packed, db_packed, n_bits):
    xor = np.bitwise_xor(db_packed, query_packed[None, :])
    return _POPCOUNT[xor].sum(axis=1).astype(np.float32) / float(n_bits)


def choose_routes(prob, policy="top1", top_r=2, threshold=0.8, true_anatomy=None):
    prob = np.asarray(prob)
    order = np.argsort(prob)[::-1]
    if policy == "oracle":
        if true_anatomy is None: raise ValueError("oracle policy requires true_anatomy")
        return [int(true_anatomy)]
    if policy == "top1": return [int(order[0])]
    if policy == "topk": return [int(v) for v in order[:top_r]]
    if policy == "adaptive":
        return [int(order[0])] if float(prob[order[0]]) >= threshold else [int(v) for v in order[:top_r]]
    raise ValueError(f"Unknown policy: {policy}")


def corrupt_probabilities(prob, error_rate, rng):
    p = np.asarray(prob, dtype=np.float64).copy()
    if rng.random() >= error_rate:
        return p
    top = int(np.argmax(p))
    candidates = np.array([i for i in range(len(p)) if i != top])
    wrong = int(rng.choice(candidates))
    p[top], p[wrong] = p[wrong], p[top]
    return p
