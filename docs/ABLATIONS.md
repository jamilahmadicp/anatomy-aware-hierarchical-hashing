# Ablation experiments

The ablation suite separates architectural, loss-function and routing effects while keeping the manifest, data preprocessing, backbone family, code length and retrieval definition fixed.

## Component ablations

### `flat_no_routing`
A flat Swin-based hashing model. It has no Stage-1 anatomy restriction during retrieval and therefore measures whether explicit anatomical hierarchy adds value beyond a strong shared representation.

### `shared_head`
Stage-1 routing is retained, but Stage 2 uses one shared projection/classification/hash head. This tests the value of anatomy-specific expert heads.

### `no_embedding_supcon`
Sets `lambda_sup=0`. Direct semantic supervision in the hash branch remains active. This isolates embedding-space supervised contrastive learning.

### `no_prototype_sign_margin`
Sets `lambda_prototype=0` and `lambda_sign_margin=0`. Hash-space supervised contrastive and pairwise losses remain active. This tests the contribution of direct binary-prototype/sign supervision.

### `no_semantic_hash`
Sets all direct semantic hash terms to zero:

```text
lambda_hash_sup = 0
lambda_hash_pair = 0
lambda_prototype = 0
lambda_sign_margin = 0
```

The checkpoint is exported even if its binary-diversity gate fails. This is intentional: the experiment is designed to quantify both retrieval degradation and code degeneration. Always report `hash_audit.json` for this ablation.

### `no_quant_balance`
Sets `lambda_quant=0` and `lambda_balance=0`. Semantic supervision remains active. This tests whether compact-code regularization materially affects binary retrieval.

### `full`
Complete anatomy-specific Stage-2 configuration.

## Routing-policy ablations

The same full Stage-2 checkpoint is evaluated under four policies:

1. Top-1 database / Top-1 query.
2. Top-1 database / Top-2 query.
3. Confidence-adaptive database / Top-2 query.
4. Oracle database / Oracle query.

The oracle experiment is an upper bound, not a deployable method.

## Routing-error sensitivity

The Stage-1 query probability vector is deliberately corrupted at rates:

```text
0.00, 0.05, 0.10, 0.15, 0.20, 0.30
```

For each rate, the code evaluates hard and uncertainty-aware routing and reports mAP, P@20 and R@20.

## Run all ablations

```bash
python scripts/run_ablation_suite.py \
  --manifest /data/IRMA/irma_manifest.csv \
  --stage1 outputs/irma/seed42/stage1.pt \
  --full-stage2 outputs/irma/seed42/stage2_128.pt \
  --out-dir outputs/irma/ablations \
  --bits 128 \
  --epochs 40 \
  --seed 42 \
  --device cuda \
  --workers 8 \
  --relevance-col fine_id \
  --threshold 0.8 \
  --top-r 2 \
  --alpha 0.05
```

Replace routing values with the validation-selected values for the run being reported.

## Final table outputs

```text
outputs/irma/ablations/tables/component_ablation.csv
outputs/irma/ablations/tables/routing_policy_ablation.csv
outputs/irma/ablations/tables/routing_error_sensitivity.csv
```
