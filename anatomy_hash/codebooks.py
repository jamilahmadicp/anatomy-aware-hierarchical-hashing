from __future__ import annotations

import numpy as np
import torch


def balanced_code(bits: int, rng: np.random.Generator) -> np.ndarray:
    """Return an approximately balanced binary vector in {-1,+1}."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    n_neg = bits // 2
    code = np.ones(bits, dtype=np.float32)
    code[:n_neg] = -1.0
    rng.shuffle(code)
    return code


def route_codebook(
    n_classes: int,
    bits: int,
    rng: np.random.Generator,
    min_hamming_fraction: float = 0.35,
    candidates: int = 2048,
) -> np.ndarray:
    """Construct deterministic, well-separated class prototypes for one route."""
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    first = balanced_code(bits, rng)
    if n_classes == 1:
        return first[None, :]
    if n_classes == 2:
        return np.stack([first, -first], axis=0).astype(np.float32)

    chosen = [first]
    for _ in range(1, n_classes):
        best = None
        best_score = -1.0
        for _candidate in range(candidates):
            cand = balanced_code(bits, rng)
            score = min(float(np.mean(cand != prev)) for prev in chosen)
            if score > best_score:
                best = cand.copy()
                best_score = score
            if best_score >= min_hamming_fraction:
                break
        if best is None:
            raise RuntimeError("Failed to construct hash prototype")
        chosen.append(best)
    return np.stack(chosen, axis=0).astype(np.float32)


def build_route_codebooks(
    anatomy_fine_maps: dict,
    bits: int,
    seed: int,
    min_hamming_fraction: float = 0.35,
) -> dict[int, torch.Tensor]:
    """Build one deterministic prototype codebook per anatomy route."""
    out: dict[int, torch.Tensor] = {}
    for anatomy_id in sorted(anatomy_fine_maps):
        n_classes = len(anatomy_fine_maps[anatomy_id]["global_ids"])
        rng = np.random.default_rng(
            int(seed) * 100003 + int(anatomy_id) * 1009 + int(bits) * 17
        )
        out[int(anatomy_id)] = torch.from_numpy(
            route_codebook(
                n_classes,
                bits,
                rng,
                min_hamming_fraction=min_hamming_fraction,
            )
        )
    return out
