# Reproducibility checklist

Before reporting a result, archive the following together:

1. exact manifest CSV;
2. manifest audit JSON;
3. class-count exports;
4. random seed;
5. Stage-1 checkpoint, history and calibration file;
6. Stage-2 checkpoint and history;
7. routing grid and selected routing CSV;
8. index metadata JSON;
9. hash audit JSON;
10. query-level retrieval metrics;
11. aggregate retrieval summary;
12. environment JSON;
13. Git commit hash used for the run.

## Determinism

`seed_everything()` configures Python, NumPy and PyTorch seeds and enables deterministic algorithms where supported. CUDA matrix multiplication may additionally require:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The supplied PowerShell and shell runners set this variable when absent.

Some GPU kernels may still vary across PyTorch/CUDA/hardware versions. For paper-level reporting, prefer repeated seeds and confidence intervals rather than relying on bit-identical results across different machines.

## Test-set isolation

Do not select:

- temperature scaling;
- checkpoint epoch;
- code length;
- database confidence threshold;
- query top-r;
- fusion alpha;
- loss weights;

using test-set metrics.

## Final GitHub release

For a final archival release, record the exact package versions used by the accepted experiment. The environment JSON files generated during training are the source of truth for those versions.
