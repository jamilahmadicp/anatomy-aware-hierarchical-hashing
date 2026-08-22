#!/usr/bin/env bash
set -euo pipefail
: "${IRMA_MANIFEST:?Set IRMA_MANIFEST}"
: "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
DEVICE=${DEVICE:-auto}; WORKERS=${WORKERS:-8}; EPOCHS2=${EPOCHS2:-40}; SEED=${SEED:-42}; OUT_DIR=${OUT_DIR:-outputs/irma/ablations}; THRESHOLD=${THRESHOLD:-0.8}; TOP_R=${TOP_R:-2}; ALPHA=${ALPHA:-0.05}
ARGS=(scripts/run_ablation_suite.py --manifest "$IRMA_MANIFEST" --stage1 "$STAGE1_CHECKPOINT" --out-dir "$OUT_DIR" --bits 128 --epochs "$EPOCHS2" --seed "$SEED" --device "$DEVICE" --workers "$WORKERS" --relevance-col fine_id --threshold "$THRESHOLD" --top-r "$TOP_R" --alpha "$ALPHA")
[[ -n "${STAGE2_CHECKPOINT:-}" ]] && ARGS+=(--full-stage2 "$STAGE2_CHECKPOINT")
python "${ARGS[@]}"
