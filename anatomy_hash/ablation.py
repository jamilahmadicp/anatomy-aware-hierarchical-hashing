from __future__ import annotations

"""Named ablation profiles for Stage-2 training.

The profiles are intentionally explicit so that manuscript ablations can be
reproduced without relying on undocumented command-line combinations.
"""

ABLATION_PROFILES = {
    "full": {},
    "no_embedding_supcon": {
        "lambda_sup": 0.0,
    },
    "no_prototype_sign_margin": {
        "lambda_prototype": 0.0,
        "lambda_sign_margin": 0.0,
    },
    "no_semantic_hash": {
        "lambda_hash_sup": 0.0,
        "lambda_hash_pair": 0.0,
        "lambda_prototype": 0.0,
        "lambda_sign_margin": 0.0,
        "allow_collapsed_checkpoint": True,
    },
    "no_quant_balance": {
        "lambda_quant": 0.0,
        "lambda_balance": 0.0,
    },
    "shared_head": {
        "shared_head": True,
    },
}


def apply_ablation_profile(args):
    """Apply a named profile to an argparse namespace in place."""
    name = getattr(args, "ablation_profile", "full")
    if name not in ABLATION_PROFILES:
        raise ValueError(f"Unknown ablation profile: {name}")
    for key, value in ABLATION_PROFILES[name].items():
        setattr(args, key, value)
    return args


def describe_profiles() -> dict[str, dict]:
    return {k: dict(v) for k, v in ABLATION_PROFILES.items()}
