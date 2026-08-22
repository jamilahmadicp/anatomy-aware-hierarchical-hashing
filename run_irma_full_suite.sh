#!/usr/bin/env bash
set -euo pipefail
: "${IRMA_MANIFEST:?Set IRMA_MANIFEST=/absolute/path/irma_manifest.csv}"
DEVICE=${DEVICE:-auto}; SEEDS=${SEEDS:-"42"}; EPOCHS1=${EPOCHS1:-30}; EPOCHS2=${EPOCHS2:-40}; WORKERS=${WORKERS:-8}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/irma
python scripts/audit_manifest.py --manifest "$IRMA_MANIFEST" --out outputs/irma/manifest_audit.json
python scripts/export_class_counts.py --manifest "$IRMA_MANIFEST" --out-dir outputs/irma/class_counts
for SEED in $SEEDS; do
  BASE="outputs/irma/seed${SEED}"; mkdir -p "$BASE"
  [[ -f "$BASE/stage1.pt" ]] || python scripts/train_stage1.py --manifest "$IRMA_MANIFEST" --out "$BASE/stage1.pt" --epochs "$EPOCHS1" --seed "$SEED" --device "$DEVICE" --workers "$WORKERS"
  [[ -f "$BASE/stage1.calibration.csv" ]] || python scripts/calibrate_stage1.py --manifest "$IRMA_MANIFEST" --checkpoint "$BASE/stage1.pt" --device "$DEVICE" --workers "$WORKERS"
  python scripts/evaluate_stage1.py --manifest "$IRMA_MANIFEST" --checkpoint "$BASE/stage1.pt" --split test --out-dir "$BASE/stage1_eval" --device "$DEVICE" --workers "$WORKERS"
  if ! python scripts/check_stage2_protocol.py --checkpoint "$BASE/stage2_128.pt" >/dev/null 2>&1; then
    rm -f "$BASE/stage2_128.pt"
    python scripts/train_stage2.py --manifest "$IRMA_MANIFEST" --out "$BASE/stage2_128.pt" --bits 128 --epochs "$EPOCHS2" --seed "$SEED" --device "$DEVICE" --workers "$WORKERS" --ablation-profile full
  fi
  TUNE="$BASE/routing_tuning"; SELECTED="$TUNE/selected_routing_hyperparameters.csv"
  [[ -f "$SELECTED" ]] || python scripts/tune_routing_hyperparameters.py --manifest "$IRMA_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$BASE/stage2_128.pt" --work-dir "$TUNE" --thresholds 0.6 0.7 0.8 0.9 --routes 1 2 3 --alphas 0 0.025 0.05 0.1 --db-top-r 2 --db-splits train --query-split val --relevance-col fine_id --device "$DEVICE" --workers "$WORKERS"
  read THRESHOLD DBTOP QUERYR ALPHA < <(python - "$SELECTED" <<'PY'
import pandas as pd, sys
r=pd.read_csv(sys.argv[1]).iloc[0]
print(r.threshold, int(r.db_top_r), int(r.query_r), r.alpha)
PY
)
  echo "Selected routing: threshold=$THRESHOLD db_top_r=$DBTOP query_r=$QUERYR alpha=$ALPHA"
  python scripts/evaluate_stage2_classification.py --manifest "$IRMA_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$BASE/stage2_128.pt" --split test --top-r "$QUERYR" --out-dir "$BASE/stage2_eval" --device "$DEVICE" --workers "$WORKERS"
  python scripts/build_hierarchical_index.py --manifest "$IRMA_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$BASE/stage2_128.pt" --out "$BASE/index_top1.npz" --db-splits train val --policy top1 --device "$DEVICE" --workers "$WORKERS"
  python scripts/build_hierarchical_index.py --manifest "$IRMA_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$BASE/stage2_128.pt" --out "$BASE/index_adaptive.npz" --db-splits train val --policy adaptive --top-r "$DBTOP" --threshold "$THRESHOLD" --device "$DEVICE" --workers "$WORKERS"
  python scripts/build_hierarchical_index.py --manifest "$IRMA_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$BASE/stage2_128.pt" --out "$BASE/index_oracle.npz" --db-splits train val --policy oracle --device "$DEVICE" --workers "$WORKERS"
  python scripts/evaluate_retrieval.py --manifest "$IRMA_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$BASE/stage2_128.pt" --index "$BASE/index_adaptive.npz" --out-dir "$BASE/retrieval_selected" --query-policy topk --top-r "$QUERYR" --threshold "$THRESHOLD" --alpha "$ALPHA" --relevance-col fine_id --device "$DEVICE" --workers "$WORKERS"
  python scripts/bootstrap_retrieval_ci.py --query-metrics "$BASE/retrieval_selected/query_metrics.csv" --out "$BASE/retrieval_selected/bootstrap_ci.csv"
  for BITS in 32 64 256; do
    CK="$BASE/stage2_${BITS}.pt"
    if ! python scripts/check_stage2_protocol.py --checkpoint "$CK" >/dev/null 2>&1; then
      rm -f "$CK"
      python scripts/train_stage2.py --manifest "$IRMA_MANIFEST" --out "$CK" --bits "$BITS" --epochs "$EPOCHS2" --seed "$SEED" --device "$DEVICE" --workers "$WORKERS" --ablation-profile full
    fi
    IDX="$BASE/index_${BITS}.npz"
    python scripts/build_hierarchical_index.py --manifest "$IRMA_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$CK" --out "$IDX" --db-splits train val --policy adaptive --top-r "$DBTOP" --threshold "$THRESHOLD" --device "$DEVICE" --workers "$WORKERS"
    python scripts/audit_hash_codes.py --index "$IDX" --out "$BASE/hash_audit_${BITS}.json" --min-unique-per-route 2 --fail-on-collapse
    python scripts/evaluate_retrieval.py --manifest "$IRMA_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$CK" --index "$IDX" --out-dir "$BASE/hash_${BITS}" --query-policy topk --top-r "$QUERYR" --threshold "$THRESHOLD" --alpha "$ALPHA" --relevance-col fine_id --device "$DEVICE" --workers "$WORKERS"
  done
  python scripts/audit_hash_codes.py --index "$BASE/index_adaptive.npz" --out "$BASE/hash_audit_128.json" --min-unique-per-route 2 --fail-on-collapse
  for METHOD in flat_hash dsh hashnet dch; do
    CK="$BASE/${METHOD}.pt"
    [[ -f "$CK" ]] || python scripts/train_deep_baseline.py --manifest "$IRMA_MANIFEST" --out "$CK" --method "$METHOD" --bits 128 --epochs "$EPOCHS2" --seed "$SEED" --device "$DEVICE" --workers "$WORKERS"
    python scripts/build_flat_index.py --manifest "$IRMA_MANIFEST" --checkpoint "$CK" --out "$BASE/${METHOD}_index.npz" --device "$DEVICE" --workers "$WORKERS"
    python scripts/evaluate_flat_retrieval.py --manifest "$IRMA_MANIFEST" --checkpoint "$CK" --index "$BASE/${METHOD}_index.npz" --out-dir "$BASE/baseline_${METHOD}" --relevance-col fine_id --device "$DEVICE" --workers "$WORKERS"
  done
  python scripts/run_routing_robustness.py --manifest "$IRMA_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$BASE/stage2_128.pt" --index-top1 "$BASE/index_top1.npz" --index-adaptive "$BASE/index_adaptive.npz" --out-dir "$BASE/routing_robustness" --alpha "$ALPHA" --relevance-col fine_id --device "$DEVICE" --workers "$WORKERS"
  python scripts/extract_search_representations.py --manifest "$IRMA_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$BASE/stage2_128.pt" --splits train val --out "$BASE/db_repr.npz" --device "$DEVICE" --workers "$WORKERS"
  python scripts/run_efficiency.py --representations "$BASE/db_repr.npz" --out-dir "$BASE/efficiency"
done
echo "IRMA full suite completed."
