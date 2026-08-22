from __future__ import annotations
import numpy as np
from .indexing import hamming_distance_packed, choose_routes
from .metrics import retrieval_metrics_for_query


def search_index(index, query_codes_by_route, query_prob, top_r=2, query_policy="topk", alpha=0.05,
                 k=0, eps=1e-8, true_anatomy=None, threshold=0.8):
    routes = choose_routes(query_prob, query_policy, top_r=top_r, threshold=threshold, true_anatomy=true_anatomy)
    best = {}
    for c in routes:
        m = index["route"] == int(c)
        if not np.any(m): continue
        qp = query_codes_by_route[int(c)]
        dist = hamming_distance_packed(qp, index["packed"][m], int(index["n_bits"]))
        score = dist - float(alpha) * np.log(float(query_prob[c]) + eps)
        rows = np.flatnonzero(m)
        for row, sc, d in zip(rows, score, dist):
            sid = str(index["sample_id"][row])
            item = (float(sc), float(d), int(row))
            if sid not in best or item[0] < best[sid][0]:
                best[sid] = item
    ranked_all = sorted(best.items(), key=lambda kv: kv[1][0])
    ranked = ranked_all[:k] if k and k > 0 else ranked_all
    return [(sid, sc, d, row) for sid, (sc, d, row) in ranked]


def evaluate_ranked_query(ranked, index, query_row, relevance_col="fine_id", ks=(10,20,50,100,200)):
    qval = int(query_row[relevance_col])
    unique_db = {}
    for i, sid in enumerate(index["sample_id"]):
        sid = str(sid)
        if sid not in unique_db:
            unique_db[sid] = int(index[relevance_col][i])
    qsid = str(query_row["sample_id"])
    total_rel = sum(1 for sid, v in unique_db.items() if sid != qsid and v == qval)
    relevance = []
    filtered = []
    for item in ranked:
        sid, sc, d, row = item
        if sid == qsid: continue
        relevance.append(int(index[relevance_col][row]) == qval)
        filtered.append(item)
    return retrieval_metrics_for_query(relevance, total_rel, ks=ks), filtered
