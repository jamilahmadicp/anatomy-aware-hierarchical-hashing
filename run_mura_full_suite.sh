#!/usr/bin/env bash
set -euo pipefail
: "${MURA_MANIFEST:?Set MURA_MANIFEST=/absolute/path/mura_manifest.csv}"
DEVICE=${DEVICE:-auto}; SEED=${SEED:-42}; EPOCHS1=${EPOCHS1:-20}; EPOCHS2=${EPOCHS2:-30}; WORKERS=${WORKERS:-8}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
BASE="outputs/mura/seed${SEED}"; mkdir -p "$BASE"
RUN_MANIFEST="$BASE/mura_manifest_clean.csv"
python scripts/audit_image_files.py --manifest "$MURA_MANIFEST" --out "$BASE/image_file_audit.json" --clean-manifest "$RUN_MANIFEST"
python scripts/audit_manifest.py --manifest "$RUN_MANIFEST" --out "$BASE/manifest_audit.json"
python scripts/export_class_counts.py --manifest "$RUN_MANIFEST" --out-dir "$BASE/class_counts"
[[ -f "$BASE/stage1.pt" ]] || python scripts/train_stage1.py --manifest "$RUN_MANIFEST" --out "$BASE/stage1.pt" --epochs "$EPOCHS1" --seed "$SEED" --device "$DEVICE" --workers "$WORKERS"
[[ -f "$BASE/stage1.calibration.csv" ]] || python scripts/calibrate_stage1.py --manifest "$RUN_MANIFEST" --checkpoint "$BASE/stage1.pt" --device "$DEVICE" --workers "$WORKERS"
python scripts/evaluate_stage1.py --manifest "$RUN_MANIFEST" --checkpoint "$BASE/stage1.pt" --split test --out-dir "$BASE/stage1_eval" --device "$DEVICE" --workers "$WORKERS"
for BITS in 32 64 128 256; do
  CK="$BASE/stage2_${BITS}.pt"
  if ! python scripts/check_stage2_protocol.py --checkpoint "$CK" >/dev/null 2>&1; then
    rm -f "$CK" "$BASE/stage2_${BITS}.history.csv"
    python scripts/train_stage2.py --manifest "$RUN_MANIFEST" --out "$CK" --bits "$BITS" --epochs "$EPOCHS2" --seed "$SEED" --device "$DEVICE" --workers "$WORKERS" --ablation-profile full
  fi
done
TUNE="$BASE/routing_tuning"; SELECTED="$TUNE/selected_routing_hyperparameters.csv"
[[ -f "$SELECTED" ]] || python scripts/tune_routing_hyperparameters.py --manifest "$RUN_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$BASE/stage2_128.pt" --work-dir "$TUNE" --thresholds 0.6 0.7 0.8 0.9 --routes 1 2 3 --alphas 0 0.025 0.05 0.1 --db-top-r 2 --db-splits train --query-split val --relevance-col joint_id --device "$DEVICE" --workers "$WORKERS"
read THRESHOLD DBTOP QUERYR ALPHA < <(python - "$SELECTED" <<'PY'
import pandas as pd, sys
r=pd.read_csv(sys.argv[1]).iloc[0]
print(r.threshold, int(r.db_top_r), int(r.query_r), r.alpha)
PY
)
echo "Selected routing: threshold=$THRESHOLD db_top_r=$DBTOP query_r=$QUERYR alpha=$ALPHA"
for BITS in 32 64 128 256; do
  CK="$BASE/stage2_${BITS}.pt"; IDX="$BASE/index_${BITS}.npz"
  python scripts/build_hierarchical_index.py --manifest "$RUN_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$CK" --db-splits train val --out "$IDX" --policy adaptive --top-r "$DBTOP" --threshold "$THRESHOLD" --device "$DEVICE" --workers "$WORKERS"
  python scripts/audit_hash_codes.py --index "$IDX" --out "$BASE/hash_audit_${BITS}.json" --min-unique-per-route 2 --fail-on-collapse
  python scripts/evaluate_retrieval.py --manifest "$RUN_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$CK" --index "$IDX" --query-split test --out-dir "$BASE/retrieval_${BITS}" --query-policy topk --top-r "$QUERYR" --threshold "$THRESHOLD" --alpha "$ALPHA" --relevance-col joint_id --device "$DEVICE" --workers "$WORKERS"
done
python scripts/evaluate_stage2_classification.py --manifest "$RUN_MANIFEST" --stage1 "$BASE/stage1.pt" --stage2 "$BASE/stage2_128.pt" --split test --top-r "$QUERYR" --out-dir "$BASE/classification_128" --device "$DEVICE" --workers "$WORKERS"
python scripts/evaluate_mura_binary_study.py --manifest "$RUN_MANIFEST" --predictions "$BASE/classification_128/stage2_predictions.npz" --split test --out-dir "$BASE/binary_study"
echo "MURA full suite completed. See $BASE"
