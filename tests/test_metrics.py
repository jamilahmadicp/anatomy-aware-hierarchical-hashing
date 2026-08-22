import numpy as np
from anatomy_hash.metrics import average_precision_from_ranked, retrieval_metrics_for_query, ndcg_at_k

def test_ap():
    rel=[1,0,1,0]
    ap=average_precision_from_ranked(rel)
    assert abs(ap-((1/1)+(2/3))/2)<1e-8

def test_retrieval():
    m=retrieval_metrics_for_query([1,0,1,0],2,ks=(2,4))
    assert abs(m['p@2']-0.5)<1e-8
    assert abs(m['r@4']-1.0)<1e-8
    assert 0 <= m['ndcg@4'] <= 1
