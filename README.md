# Anatomy-Aware Hierarchical Contrastive Hashing for Radiograph Retrieval

Reproducible PyTorch implementation of a two-stage radiograph classification and retrieval framework:

1. **Stage 1** predicts calibrated anatomy probabilities with ConvNeXt-Tiny.
2. **Stage 2** evaluates a shared Swin-Tiny encoder once, then applies the selected anatomy-specific projection, classification, and hash heads.
3. **Retrieval** uses compact binary codes, Hamming distance, confidence-adaptive database routing, top-*r* query routing, and route-probability-weighted candidate fusion.

The repository includes the complete training and evaluation workflow for IRMA and MURA, code-length experiments, deep-hashing baselines, routing robustness, component ablations, bootstrap confidence intervals, and search-time scalability measurements.

No datasets or pretrained checkpoints are distributed in this repository. Dataset access remains subject to the original IRMA and MURA terms.

---

## 1. Repository structure

```text
anatomy-aware-hierarchical-hashing/
├── anatomy_hash/                  # Core models, losses, indexing and metrics
│   ├── data/                      # Manifest loading, datasets and transforms
│   ├── models/                    # Stage-1, Stage-2 and flat hashing models
│   ├── utils/                     # Seeding, I/O and configuration helpers
│   ├── ablation.py                # Named Stage-2 ablation profiles
│   ├── codebooks.py               # Route-specific binary class prototypes
│   ├── indexing.py                # Routing and packed binary-code utilities
│   ├── losses.py                  # Contrastive, quantization and balance losses
│   ├── metrics.py                 # Retrieval metrics
│   └── retrieval.py               # Hamming search and ranking
├── configs/
│   ├── default.yaml               # Reference full-model configuration
│   ├── ablation_profiles.yaml     # Exact component ablations
│   └── experiment_registry.yaml   # Experiment inventory
├── docs/
│   ├── ABLATIONS.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── OUTPUTS.md
│   ├── REPRODUCIBILITY.md
│   └── TROUBLESHOOTING.md
├── scripts/                       # Training, evaluation and analysis entry points
├── templates/                     # IRMA mapping template
├── tests/                         # Unit tests
├── run_irma_full_suite.ps1        # Windows IRMA pipeline
├── run_irma_full_suite.sh         # Linux/macOS IRMA pipeline
├── run_irma_ablations.ps1         # Windows component/routing ablations
├── run_irma_ablations.sh          # Linux/macOS component/routing ablations
├── run_mura_full_suite.ps1        # Windows MURA pipeline
└── run_mura_full_suite.sh         # Linux/macOS MURA pipeline
```

---

## 2. Recommended environment

### Windows

Python **3.11** is recommended for the most predictable PyTorch/Windows behavior.

```powershell
conda create -n anatomy-hash python=3.11 -y
conda activate anatomy-hash
python -m pip install --upgrade pip
```

Install **PyTorch and torchvision first**, using the PyTorch build appropriate for your GPU driver and CUDA runtime. Use the official PyTorch installation selector rather than copying a CUDA command from another computer.

After PyTorch imports correctly:

```powershell
python -m pip install -r requirements-base.txt
python -m pip install -e . --no-deps
python scripts\validate_environment.py
python -m pytest -q
```

If you intentionally want `pip` to resolve PyTorch as well, use:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate PyTorch/torchvision build first.
python -m pip install -r requirements-base.txt
python -m pip install -e . --no-deps
python scripts/validate_environment.py
python -m pytest -q
```

### Environment verification

The following must work before starting a long experiment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import anatomy_hash; print('anatomy_hash import OK')"
```

---

## 3. Data manifest

All scripts use a CSV manifest. The required columns are:

| Column | Description |
|---|---|
| `path` | Absolute path to the radiograph image |
| `split` | `train`, `val`, or `test` |
| `anatomy_label` | Coarse anatomical region |
| `fine_label` | Fine-grained class |
| `sample_id` | Unique image identifier |

Recommended additional columns:

| Column | Description |
|---|---|
| `patient_id` | Patient identifier for leakage auditing |
| `study_id` | Study identifier, required for study-level MURA evaluation |
| `abnormal_label` | Binary MURA label, 0 normal / 1 abnormal |
| `joint_label` | Anatomy + abnormality class |
| `irma_code` | Complete IRMA code when available |

The manifest loader creates deterministic integer IDs (`anatomy_id`, `fine_id`, and `joint_id`) from the labels.

See [MANIFEST_SCHEMA.md](MANIFEST_SCHEMA.md) for the complete schema.

---

## 4. Prepare MURA

Expected dataset layout:

```text
MURA-v1.1/
├── train/
│   ├── XR_SHOULDER/patient*/study*_positive/*.png
│   └── ...
└── valid/
    ├── XR_SHOULDER/patient*/study*_negative/*.png
    └── ...
```

Create a three-way manifest:

```powershell
python scripts\prepare_mura_manifest.py `
  --root "D:\Research\Dataset\MURA-v1.1" `
  --val-fraction 0.10 `
  --seed 42 `
  --out "D:\Research\Dataset\MURA-v1.1\mura_manifest.csv"
```

The protocol is:

- official MURA `train` → internal `train` + patient-disjoint `val`;
- official MURA `valid` → untouched final `test`;
- calibration, checkpoint selection and routing-parameter selection use only internal `val`;
- reported MURA results use final `test`.

The MURA runner audits image files before training and removes macOS AppleDouble sidecar rows such as `._image2.png` from a clean runtime manifest.

---

## 5. Prepare IRMA

The code does **not infer the manuscript's anatomy mapping from an IRMA code**. Supply the exact fine-class → anatomy mapping used in the study.

Create a mapping template:

```bash
python scripts/make_irma_mapping_template.py \
  --train-dir /data/IRMA/train \
  --out /data/IRMA/irma_fine_to_anatomy.csv
```

Fill every `anatomy_label` field, then create the manifest:

```bash
python scripts/prepare_irma_manifest.py \
  --train-dir /data/IRMA/train \
  --test-dir /data/IRMA/test \
  --mapping-csv /data/IRMA/irma_fine_to_anatomy.csv \
  --val-fraction 0.10 \
  --seed 42 \
  --out /data/IRMA/irma_manifest.csv
```

Before training:

```bash
python scripts/audit_manifest.py --manifest /data/IRMA/irma_manifest.csv --out outputs/irma/manifest_audit.json
python scripts/export_class_counts.py --manifest /data/IRMA/irma_manifest.csv --out-dir outputs/irma/class_counts
```

Use the generated audit and class-count files to report the exact local archive composition. Do not alter the official test partition during model development.

---

## 6. Run the full MURA experiment

### Windows PowerShell

```powershell
conda activate anatomy-hash
cd "C:\path\to\anatomy-aware-hierarchical-hashing"

$env:MURA_MANIFEST = "D:\Research\Dataset\MURA-v1.1\mura_manifest.csv"
$env:DEVICE = "cuda"
$env:SEED = "42"
$env:EPOCHS1 = "20"
$env:EPOCHS2 = "30"
$env:WORKERS = "0"

.\run_mura_full_suite.ps1
```

`WORKERS=0` is deliberately conservative on Windows. Increase it to 2 or 4 only after the pipeline is stable.

### Linux

```bash
export MURA_MANIFEST=/data/MURA-v1.1/mura_manifest.csv
export DEVICE=cuda
export SEED=42
export EPOCHS1=20
export EPOCHS2=30
export WORKERS=8
bash run_mura_full_suite.sh
```

The runner performs:

1. image-file, leakage and class-count audits;
2. Stage-1 training and temperature scaling;
3. Stage-1 final evaluation;
4. Stage-2 training at 32, 64, 128 and 256 bits;
5. validation-only selection of database confidence threshold, query route count and fusion weight using the 128-bit model;
6. final hash indexing and retrieval evaluation for all code lengths;
7. hash-diversity audits;
8. 14-class Stage-2 evaluation;
9. official study-level binary MURA evaluation.

---

## 7. Run the full IRMA experiment

### Windows PowerShell

```powershell
$env:IRMA_MANIFEST = "D:\Research\Dataset\IRMA\irma_manifest.csv"
$env:DEVICE = "cuda"
$env:SEEDS = "42 43 44"
$env:EPOCHS1 = "30"
$env:EPOCHS2 = "40"
$env:WORKERS = "0"

.\run_irma_full_suite.ps1
```

### Linux

```bash
export IRMA_MANIFEST=/data/IRMA/irma_manifest.csv
export DEVICE=cuda
export SEEDS="42 43 44"
export EPOCHS1=30
export EPOCHS2=40
export WORKERS=8
bash run_irma_full_suite.sh
```

The IRMA suite additionally runs DSH, HashNet, DCH and flat-hashing baselines, query-level bootstrap confidence intervals, routing-error sensitivity, and the search-time scalability experiment.

---

## 8. Validation-only routing parameter selection

Routing parameters are selected **before final test evaluation**.

```bash
python scripts/tune_routing_hyperparameters.py \
  --manifest /data/IRMA/irma_manifest.csv \
  --stage1 outputs/irma/seed42/stage1.pt \
  --stage2 outputs/irma/seed42/stage2_128.pt \
  --work-dir outputs/irma/seed42/routing_tuning \
  --thresholds 0.6 0.7 0.8 0.9 \
  --routes 1 2 3 \
  --alphas 0 0.025 0.05 0.1 \
  --db-top-r 2 \
  --db-splits train \
  --query-split val \
  --relevance-col fine_id \
  --device cuda
```

Outputs:

```text
routing_tuning/
├── routing_grid.csv
└── selected_routing_hyperparameters.csv
```

The selected threshold, database top-*r*, query top-*r* and fusion coefficient must be frozen before the test split is evaluated.

---

## 9. Stage-2 objective

For each selected anatomy route, Stage 2 optimizes:

- fine-grained weighted classification loss;
- supervised contrastive loss in the projected embedding;
- supervised contrastive loss in the continuous hash space;
- pairwise semantic hash loss;
- route-specific binary class-prototype regression;
- sign-margin supervision;
- quantization regularization;
- route-wise bit-balance regularization.

Quantization and balance terms are introduced gradually after the semantic warm-up period. The hard binary code is produced with `sign(q)` only for indexing and retrieval.

The primary checkpoint is selected using validation ground-truth-route binary mAP, subject to the binary-diversity gate (`min_unique_per_route >= 2`).

---

## 10. Component and routing ablations

The complete ablation suite is implemented in `scripts/run_ablation_suite.py` and documented in [docs/ABLATIONS.md](docs/ABLATIONS.md).

### Main component ablations

| Name | What changes |
|---|---|
| `flat_no_routing` | Shared Swin hashing model with no anatomy routing |
| `shared_head` | Anatomy routing retained, but projection/classification/hash heads are shared |
| `no_embedding_supcon` | Remove embedding-space supervised contrastive loss |
| `no_prototype_sign_margin` | Remove direct prototype and sign-margin supervision |
| `no_semantic_hash` | Remove all direct semantic supervision from the hash space |
| `no_quant_balance` | Remove quantization and route-wise bit balance |
| `full` | Complete proposed Stage-2 objective |

### Routing-policy ablations

- Top-1 database / Top-1 query
- Top-1 database / Top-2 query
- Confidence-adaptive database / Top-2 query
- Oracle database / Oracle query

### Routing-error sensitivity

Injected Stage-1 query-routing errors:

```text
0%, 5%, 10%, 15%, 20%, 30%
```

### Windows example

Use the validation-selected routing values from `selected_routing_hyperparameters.csv`:

```powershell
$env:IRMA_MANIFEST = "D:\Research\Dataset\IRMA\irma_manifest.csv"
$env:STAGE1_CHECKPOINT = "outputs\irma\seed42\stage1.pt"
$env:STAGE2_CHECKPOINT = "outputs\irma\seed42\stage2_128.pt"
$env:DEVICE = "cuda"
$env:WORKERS = "0"
$env:EPOCHS2 = "40"
$env:SEED = "42"

.\run_irma_ablations.ps1 -Threshold 0.80 -TopR 2 -Alpha 0.05
```

Replace `0.80`, `2`, and `0.05` with the values selected on your validation split.

The final tables are written to:

```text
outputs/irma/ablations/tables/
├── component_ablation.csv
├── routing_policy_ablation.csv
└── routing_error_sensitivity.csv
```

---

## 11. Individual Stage-2 ablations

A single named ablation can be run directly:

```bash
python scripts/train_stage2.py \
  --manifest /data/IRMA/irma_manifest.csv \
  --out outputs/ablation_no_embedding_supcon.pt \
  --bits 128 \
  --epochs 40 \
  --ablation-profile no_embedding_supcon \
  --device cuda
```

Available profile names:

```text
full
shared_head
no_embedding_supcon
no_prototype_sign_margin
no_semantic_hash
no_quant_balance
```

`no_semantic_hash` intentionally allows checkpoint export even if the binary-diversity audit fails. This permits the ablation to quantify whether removing direct semantic hash supervision causes degenerate codes. The hash audit must therefore be reported alongside its retrieval metrics.

---

## 12. Hash-code quality audit

Every main hierarchical index should be checked:

```bash
python scripts/audit_hash_codes.py \
  --index outputs/mura/seed42/index_128.npz \
  --out outputs/mura/seed42/hash_audit_128.json \
  --min-unique-per-route 2 \
  --fail-on-collapse
```

The report includes:

- total unique binary codes;
- unique-code ratio;
- bit balance;
- unique codes per anatomy route;
- native route/fine-class counts;
- routes failing the minimum diversity requirement.

Do not report a main retrieval result when the main-model audit fails.

---

## 13. Deep hashing baselines

Implemented baselines:

```text
flat_hash
DSH
dsh
HashNet
hashnet
DCH
dch
```

Example:

```bash
python scripts/train_deep_baseline.py \
  --manifest /data/IRMA/irma_manifest.csv \
  --method dch \
  --bits 128 \
  --epochs 40 \
  --out outputs/dch.pt \
  --device cuda
```

Baseline checkpoints are selected using **validation binary retrieval mAP**, with validation classification accuracy used only as a tie-breaker.

---

## 14. Retrieval metrics

Each retrieval run produces query-level and aggregate values for:

- mean average precision (mAP);
- Precision@10/20/50/100/200;
- Recall@10/20/50/100/200;
- nDCG@10/20/50/100/200.

Query-level files enable bootstrap confidence intervals and paired method comparisons.

```bash
python scripts/bootstrap_retrieval_ci.py \
  --query-metrics outputs/irma/seed42/retrieval_selected/query_metrics.csv \
  --out outputs/irma/seed42/retrieval_selected/bootstrap_ci.csv
```

---

## 15. Scalability experiment

The efficiency code measures a **representation-level search stress test**, not additional independent patients.

```bash
python scripts/extract_search_representations.py \
  --manifest /data/IRMA/irma_manifest.csv \
  --stage1 outputs/irma/seed42/stage1.pt \
  --stage2 outputs/irma/seed42/stage2_128.pt \
  --splits train val \
  --out outputs/irma/seed42/db_repr.npz \
  --device cuda

python scripts/run_efficiency.py \
  --representations outputs/irma/seed42/db_repr.npz \
  --out-dir outputs/irma/seed42/efficiency
```

The experiment explicitly reports milliseconds/query and documents how larger representation indexes are constructed.

---

## 16. Reproducibility controls

The code records or fixes:

- train/validation/test manifests;
- random seed;
- model architecture names;
- hash length;
- optimizer settings;
- Stage-2 loss weights;
- Stage-1 temperature calibration;
- validation-selected routing parameters;
- checkpoint selection metric;
- binary-diversity statistics;
- Python/package/hardware environment snapshots;
- query-level retrieval metrics.

For final manuscript results, use the same manifest and fixed protocol across all compared methods. If multiple seeds are reported, aggregate them with `scripts/aggregate_seeds.py` rather than selecting the best seed.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## 17. Output directories

See [docs/OUTPUTS.md](docs/OUTPUTS.md) for a detailed file-by-file description.

Important outputs include:

```text
stage1.history.csv
stage1.calibration.csv
stage1.environment.json
stage2_128.history.csv
stage2_128.environment.json
routing_tuning/selected_routing_hyperparameters.csv
hash_audit_128.json
retrieval_128/query_metrics.csv
retrieval_128/retrieval_summary.csv
classification_128/stage2_metrics.csv
binary_study/mura_binary_study_metrics.csv
```

---

## 18. Windows troubleshooting

Common Windows-specific issues and fixes are documented in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md), including:

- `c10.dll` / PyTorch DLL load failures;
- DataLoader multiprocessing and `WORKERS=0`;
- macOS `._*.png` AppleDouble files;
- deterministic CUDA/CuBLAS configuration;
- missing local package imports;
- safe resume behavior after an interrupted run.

---

## 19. Testing

Run:

```bash
python -m pytest -q
```

The tests cover packed Hamming distance, routing policy selection, retrieval metrics, route-specific prototype codebooks, and named ablation profiles.

Long GPU training is not part of the unit-test suite.

---

## 20. Citation

If this repository is used with the associated manuscript, cite the final published article. A `CITATION.cff` file is included so that GitHub can expose citation metadata.

---

## 21. License

MIT License. See [LICENSE](LICENSE).
