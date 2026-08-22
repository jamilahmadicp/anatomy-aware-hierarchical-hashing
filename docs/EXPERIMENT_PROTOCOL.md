# Experiment protocol

## Data separation

- Model fitting uses `train` only.
- Temperature calibration, early stopping, code-length/model selection and routing-parameter selection use `val` only.
- Final reported performance uses `test` only.
- The database for final retrieval may use `train + val` after all hyperparameters have been frozen.

For MURA, the official validation set is treated as final test data; internal validation is carved only from the official training set using patient-disjoint splitting.

## Stage 1

Backbone: ConvNeXt-Tiny.

Outputs:

- anatomy logits;
- temperature-scaled anatomy probabilities;
- top-r route identifiers;
- route probabilities/weights.

Temperature is fitted on validation logits only.

## Stage 2

Backbone: shared Swin-Tiny, evaluated once per image.

Selected anatomy-specific heads:

- projection head;
- fine-grained classification head;
- hash head.

Continuous hash values use `tanh`. Hard binary codes use `sign` only for indexing/retrieval.

## Checkpoint selection

Stage 1: best validation anatomy accuracy in the current reference implementation.

Stage 2: best validation ground-truth-route binary retrieval mAP, subject to the minimum per-route binary-diversity gate.

Deep hashing baselines: best validation binary retrieval mAP.

## Routing parameter selection

Validation search space:

- database confidence threshold: 0.6, 0.7, 0.8, 0.9;
- query route count: 1, 2, 3;
- fusion alpha: 0, 0.025, 0.05, 0.1;
- database top-r is fixed at 2 unless a separate experiment explicitly changes it.

The selected values are written to `selected_routing_hyperparameters.csv` and frozen before test evaluation.
