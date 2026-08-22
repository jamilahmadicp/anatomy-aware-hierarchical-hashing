from types import SimpleNamespace
from anatomy_hash.ablation import apply_ablation_profile


def base_args(profile):
    return SimpleNamespace(
        ablation_profile=profile,
        lambda_sup=1.0,
        lambda_hash_sup=0.25,
        lambda_hash_pair=0.25,
        lambda_prototype=3.0,
        lambda_sign_margin=2.0,
        lambda_quant=0.02,
        lambda_balance=0.01,
        shared_head=False,
        allow_collapsed_checkpoint=False,
    )


def test_no_semantic_hash_profile():
    a = apply_ablation_profile(base_args("no_semantic_hash"))
    assert a.lambda_hash_sup == 0.0
    assert a.lambda_hash_pair == 0.0
    assert a.lambda_prototype == 0.0
    assert a.lambda_sign_margin == 0.0
    assert a.allow_collapsed_checkpoint is True


def test_shared_head_profile():
    a = apply_ablation_profile(base_args("shared_head"))
    assert a.shared_head is True
