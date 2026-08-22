import numpy as np
from anatomy_hash.codebooks import route_codebook


def test_two_class_codebook_antipodal():
    rng = np.random.default_rng(42)
    cb = route_codebook(2, 32, rng)
    assert cb.shape == (2, 32)
    assert np.all(cb[1] == -cb[0])
    assert np.mean(cb[0] != cb[1]) == 1.0


def test_multiclass_codebook_unique():
    rng = np.random.default_rng(42)
    cb = route_codebook(8, 32, rng)
    assert len(np.unique(cb, axis=0)) == 8
    for i in range(len(cb)):
        for j in range(i + 1, len(cb)):
            assert np.mean(cb[i] != cb[j]) >= 0.25
