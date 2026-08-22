[CmdletBinding()]
param(
    [string]$Manifest = $env:IRMA_MANIFEST,
    [string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { 'auto' }),
    [string]$Seeds = $(if ($env:SEEDS) { $env:SEEDS } else { '42' }),
    [int]$Epochs1 = $(if ($env:EPOCHS1) { [int]$env:EPOCHS1 } else { 30 }),
    [int]$Epochs2 = $(if ($env:EPOCHS2) { [int]$env:EPOCHS2 } else { 40 }),
    [int]$Workers = $(if ($env:WORKERS) { [int]$env:WORKERS } else { 0 })
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ([string]::IsNullOrWhiteSpace($env:CUBLAS_WORKSPACE_CONFIG)) { $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8' }

function Run-Python {
    $PythonArgs = $args
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        $ExitCode = $LASTEXITCODE
        throw "Python command failed with exit code ${ExitCode}: python $($PythonArgs -join ' ')"
    }
}

function Test-Stage2Protocol {
    param([string]$Checkpoint)
    if (-not (Test-Path $Checkpoint)) { return $false }
    & python scripts/check_stage2_protocol.py --checkpoint $Checkpoint *> $null
    return ($LASTEXITCODE -eq 0)
}

if ([string]::IsNullOrWhiteSpace($Manifest)) { throw "Set IRMA_MANIFEST or pass -Manifest." }
if (-not (Test-Path $Manifest)) { throw "IRMA manifest not found: $Manifest" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) { $env:PYTHONPATH = $ScriptDir } else { $env:PYTHONPATH = "$ScriptDir;$env:PYTHONPATH" }
Push-Location $ScriptDir
try {
    New-Item -ItemType Directory -Force -Path 'outputs/irma' | Out-Null
    Run-Python scripts/audit_manifest.py --manifest $Manifest --out 'outputs/irma/manifest_audit.json'
    Run-Python scripts/export_class_counts.py --manifest $Manifest --out-dir 'outputs/irma/class_counts'

    $SeedList = $Seeds -split '[,\s]+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    foreach ($Seed in $SeedList) {
        $Base = "outputs/irma/seed${Seed}"
        New-Item -ItemType Directory -Force -Path $Base | Out-Null

        if (-not (Test-Path "$Base/stage1.pt")) {
            Run-Python scripts/train_stage1.py --manifest $Manifest --out "$Base/stage1.pt" --epochs $Epochs1 --seed $Seed --device $Device --workers $Workers
        }
        if (-not (Test-Path "$Base/stage1.calibration.csv")) {
            Run-Python scripts/calibrate_stage1.py --manifest $Manifest --checkpoint "$Base/stage1.pt" --device $Device --workers $Workers
        }
        if (-not (Test-Path "$Base/stage1_eval/stage1_metrics.csv")) {
            Run-Python scripts/evaluate_stage1.py --manifest $Manifest --checkpoint "$Base/stage1.pt" --split test --out-dir "$Base/stage1_eval" --device $Device --workers $Workers
        }

        if (-not (Test-Stage2Protocol "$Base/stage2_128.pt")) {
            Remove-Item "$Base/stage2_128.pt" -Force -ErrorAction SilentlyContinue
            Run-Python scripts/train_stage2.py --manifest $Manifest --out "$Base/stage2_128.pt" --bits 128 --epochs $Epochs2 --seed $Seed --device $Device --workers $Workers --ablation-profile full
        }

        $Tune = "$Base/routing_tuning"
        $SelectedFile = "$Tune/selected_routing_hyperparameters.csv"
        if (-not (Test-Path $SelectedFile)) {
            Run-Python scripts/tune_routing_hyperparameters.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --work-dir $Tune --thresholds 0.6 0.7 0.8 0.9 --routes 1 2 3 --alphas 0 0.025 0.05 0.1 --db-top-r 2 --db-splits train --query-split val --relevance-col fine_id --device $Device --workers $Workers
        }
        $Selected = Import-Csv $SelectedFile | Select-Object -First 1
        $Threshold = [double]$Selected.threshold
        $DbTopR = [int]$Selected.db_top_r
        $QueryR = [int]$Selected.query_r
        $Alpha = [double]$Selected.alpha
        Write-Host "Selected routing: threshold=$Threshold db_top_r=$DbTopR query_r=$QueryR alpha=$Alpha" -ForegroundColor Cyan

        Run-Python scripts/evaluate_stage2_classification.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --split test --top-r $QueryR --out-dir "$Base/stage2_eval" --device $Device --workers $Workers

        if (-not (Test-Path "$Base/index_top1.npz")) { Run-Python scripts/build_hierarchical_index.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --out "$Base/index_top1.npz" --db-splits train val --policy top1 --device $Device --workers $Workers }
        if (-not (Test-Path "$Base/index_adaptive.npz")) { Run-Python scripts/build_hierarchical_index.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --out "$Base/index_adaptive.npz" --db-splits train val --policy adaptive --top-r $DbTopR --threshold $Threshold --device $Device --workers $Workers }
        if (-not (Test-Path "$Base/index_oracle.npz")) { Run-Python scripts/build_hierarchical_index.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --out "$Base/index_oracle.npz" --db-splits train val --policy oracle --device $Device --workers $Workers }

        Run-Python scripts/evaluate_retrieval.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --index "$Base/index_adaptive.npz" --out-dir "$Base/retrieval_selected" --query-policy topk --top-r $QueryR --threshold $Threshold --alpha $Alpha --relevance-col fine_id --device $Device --workers $Workers
        Run-Python scripts/evaluate_retrieval.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --index "$Base/index_top1.npz" --out-dir "$Base/retrieval_top1_top1" --query-policy top1 --top-r 1 --alpha $Alpha --relevance-col fine_id --device $Device --workers $Workers
        Run-Python scripts/evaluate_retrieval.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --index "$Base/index_top1.npz" --out-dir "$Base/retrieval_top1_top2" --query-policy topk --top-r 2 --alpha $Alpha --relevance-col fine_id --device $Device --workers $Workers
        Run-Python scripts/evaluate_retrieval.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --index "$Base/index_oracle.npz" --out-dir "$Base/retrieval_oracle" --query-policy oracle --top-r 1 --alpha 0 --relevance-col fine_id --device $Device --workers $Workers
        Run-Python scripts/bootstrap_retrieval_ci.py --query-metrics "$Base/retrieval_selected/query_metrics.csv" --out "$Base/retrieval_selected/bootstrap_ci.csv"

        foreach ($Bits in 32,64,256) {
            $Ck = "$Base/stage2_${Bits}.pt"
            if (-not (Test-Stage2Protocol $Ck)) {
                Remove-Item $Ck -Force -ErrorAction SilentlyContinue
                Run-Python scripts/train_stage2.py --manifest $Manifest --out $Ck --bits $Bits --epochs $Epochs2 --seed $Seed --device $Device --workers $Workers --ablation-profile full
            }
            $Idx = "$Base/index_${Bits}.npz"
            Run-Python scripts/build_hierarchical_index.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 $Ck --out $Idx --db-splits train val --policy adaptive --top-r $DbTopR --threshold $Threshold --device $Device --workers $Workers
            Run-Python scripts/audit_hash_codes.py --index $Idx --out "$Base/hash_audit_${Bits}.json" --min-unique-per-route 2 --fail-on-collapse
            Run-Python scripts/evaluate_retrieval.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 $Ck --index $Idx --out-dir "$Base/hash_${Bits}" --query-policy topk --top-r $QueryR --threshold $Threshold --alpha $Alpha --relevance-col fine_id --device $Device --workers $Workers
        }
        Run-Python scripts/audit_hash_codes.py --index "$Base/index_adaptive.npz" --out "$Base/hash_audit_128.json" --min-unique-per-route 2 --fail-on-collapse

        foreach ($Method in 'flat_hash','dsh','hashnet','dch') {
            $Ck = "$Base/${Method}.pt"
            if (-not (Test-Path $Ck)) { Run-Python scripts/train_deep_baseline.py --manifest $Manifest --out $Ck --method $Method --bits 128 --epochs $Epochs2 --seed $Seed --device $Device --workers $Workers }
            Run-Python scripts/build_flat_index.py --manifest $Manifest --checkpoint $Ck --out "$Base/${Method}_index.npz" --device $Device --workers $Workers
            Run-Python scripts/evaluate_flat_retrieval.py --manifest $Manifest --checkpoint $Ck --index "$Base/${Method}_index.npz" --out-dir "$Base/baseline_${Method}" --relevance-col fine_id --device $Device --workers $Workers
        }

        Run-Python scripts/run_routing_robustness.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --index-top1 "$Base/index_top1.npz" --index-adaptive "$Base/index_adaptive.npz" --out-dir "$Base/routing_robustness" --alpha $Alpha --relevance-col fine_id --device $Device --workers $Workers
        Run-Python scripts/extract_search_representations.py --manifest $Manifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --splits train val --out "$Base/db_repr.npz" --device $Device --workers $Workers
        Run-Python scripts/run_efficiency.py --representations "$Base/db_repr.npz" --out-dir "$Base/efficiency"
    }
    Write-Host 'IRMA full suite completed.' -ForegroundColor Green
}
finally { Pop-Location }
