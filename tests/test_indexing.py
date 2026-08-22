import numpy as np
from anatomy_hash.indexing import pack_codes, hamming_distance_packed, choose_routes

def test_hamming():
    c=np.array([[1,1,-1,-1],[1,-1,-1,-1]],dtype=np.int8)
    p=pack_codes(c)
    d=hamming_distance_packed(p[0],p,4)
    assert np.allclose(d,[0,.25])

def test_routes():
    p=np.array([.1,.7,.2])
    assert choose_routes(p,'top1')==[1]
    assert choose_routes(p,'topk',2)==[1,2]
    assert choose_routes(p,'adaptive',2,.8)==[1,2]
    assert choose_routes(p,'adaptive',2,.6)==[1]
