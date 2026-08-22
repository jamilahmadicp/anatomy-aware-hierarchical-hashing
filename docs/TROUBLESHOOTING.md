# Troubleshooting

## Windows: `c10.dll` or WinError 1114

This is a PyTorch/CUDA runtime problem, not a dataset problem.

Recommended recovery:

1. create a clean Python 3.11 conda environment;
2. install the Microsoft Visual C++ x64 redistributable;
3. install the PyTorch build matching the machine's supported CUDA setup;
4. verify `import torch` before installing the rest of this repository;
5. install `requirements-base.txt` and then `pip install -e . --no-deps`.

## Windows DataLoader errors

Start with:

```powershell
$env:WORKERS = "0"
```

The transforms used by this repository are pickle-safe, but `WORKERS=0` remains the most conservative Windows configuration.

## `PIL.UnidentifiedImageError` for `._image2.png`

Files beginning with `._` are macOS AppleDouble metadata sidecars, not radiographs. Re-run the MURA manifest preparation or image-file audit. The full MURA runner automatically creates a cleaned runtime manifest.

## `ModuleNotFoundError: anatomy_hash`

From the repository root:

```bash
python -m pip install -e . --no-deps
```

or temporarily add the repository root to `PYTHONPATH`.

## Deterministic CuBLAS warning

Set before starting Python:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

or on Linux:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Hash-diversity audit fails

Do not bypass the audit for a main result. Inspect:

- `stage2_<bits>.history.csv`;
- `hash_audit_<bits>.json`;
- `val_hash_min_unique_route`;
- `val_hash_min_interclass_hamming`.

The `no_semantic_hash` ablation is the only standard profile that intentionally permits a collapsed checkpoint so that the failure can be quantified.

## Interrupted run

The PowerShell full-suite scripts reuse completed checkpoints where possible. If a run was interrupted during Stage 2, confirm that the checkpoint protocol is valid:

```bash
python scripts/check_stage2_protocol.py --checkpoint path/to/stage2_128.pt
```
