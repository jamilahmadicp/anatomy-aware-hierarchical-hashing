[CmdletBinding()]
param(
    [string]$Manifest = $env:MURA_MANIFEST,
    [string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { 'auto' }),
    [int]$Seed = $(if ($env:SEED) { [int]$env:SEED } else { 42 }),
    [int]$Epochs1 = $(if ($env:EPOCHS1) { [int]$env:EPOCHS1 } else { 20 }),
    [int]$Epochs2 = $(if ($env:EPOCHS2) { [int]$env:EPOCHS2 } else { 30 }),
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

if ([string]::IsNullOrWhiteSpace($Manifest)) { throw "Set MURA_MANIFEST or pass -Manifest." }
if (-not (Test-Path $Manifest)) { throw "MURA manifest not found: $Manifest" }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) { $env:PYTHONPATH = $ScriptDir } else { $env:PYTHONPATH = "$ScriptDir;$env:PYTHONPATH" }
Push-Location $ScriptDir
try {
    $Base = "outputs/mura/seed${Seed}"
    New-Item -ItemType Directory -Force -Path $Base | Out-Null
    $RunManifest = "$Base/mura_manifest_clean.csv"
    Run-Python scripts/audit_image_files.py --manifest $Manifest --out "$Base/image_file_audit.json" --clean-manifest $RunManifest
    Run-Python scripts/audit_manifest.py --manifest $RunManifest --out "$Base/manifest_audit.json"
    Run-Python scripts/export_class_counts.py --manifest $RunManifest --out-dir "$Base/class_counts"

    if (-not (Test-Path "$Base/stage1.pt")) { Run-Python scripts/train_stage1.py --manifest $RunManifest --out "$Base/stage1.pt" --epochs $Epochs1 --seed $Seed --device $Device --workers $Workers }
    if (-not (Test-Path "$Base/stage1.calibration.csv")) { Run-Python scripts/calibrate_stage1.py --manifest $RunManifest --checkpoint "$Base/stage1.pt" --device $Device --workers $Workers }
    Run-Python scripts/evaluate_stage1.py --manifest $RunManifest --checkpoint "$Base/stage1.pt" --split test --out-dir "$Base/stage1_eval" --device $Device --workers $Workers

    foreach ($Bits in 32,64,128,256) {
        $Stage2 = "$Base/stage2_${Bits}.pt"
        if (-not (Test-Stage2Protocol $Stage2)) {
            Remove-Item $Stage2 -Force -ErrorAction SilentlyContinue
            Remove-Item "$Base/stage2_${Bits}.history.csv" -Force -ErrorAction SilentlyContinue
            Run-Python scripts/train_stage2.py --manifest $RunManifest --out $Stage2 --bits $Bits --epochs $Epochs2 --seed $Seed --device $Device --workers $Workers --ablation-profile full
        }
    }

    $Tune = "$Base/routing_tuning"
    $SelectedFile = "$Tune/selected_routing_hyperparameters.csv"
    if (-not (Test-Path $SelectedFile)) {
        Run-Python scripts/tune_routing_hyperparameters.py --manifest $RunManifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --work-dir $Tune --thresholds 0.6 0.7 0.8 0.9 --routes 1 2 3 --alphas 0 0.025 0.05 0.1 --db-top-r 2 --db-splits train --query-split val --relevance-col joint_id --device $Device --workers $Workers
    }
    $Selected = Import-Csv $SelectedFile | Select-Object -First 1
    $Threshold = [double]$Selected.threshold
    $DbTopR = [int]$Selected.db_top_r
    $QueryR = [int]$Selected.query_r
    $Alpha = [double]$Selected.alpha
    Write-Host "Selected routing: threshold=$Threshold db_top_r=$DbTopR query_r=$QueryR alpha=$Alpha" -ForegroundColor Cyan

    foreach ($Bits in 32,64,128,256) {
        $Stage2 = "$Base/stage2_${Bits}.pt"
        $Index = "$Base/index_${Bits}.npz"
        Run-Python scripts/build_hierarchical_index.py --manifest $RunManifest --stage1 "$Base/stage1.pt" --stage2 $Stage2 --db-splits train val --out $Index --policy adaptive --top-r $DbTopR --threshold $Threshold --device $Device --workers $Workers
        Run-Python scripts/audit_hash_codes.py --index $Index --out "$Base/hash_audit_${Bits}.json" --min-unique-per-route 2 --fail-on-collapse
        Run-Python scripts/evaluate_retrieval.py --manifest $RunManifest --stage1 "$Base/stage1.pt" --stage2 $Stage2 --index $Index --query-split test --out-dir "$Base/retrieval_${Bits}" --query-policy topk --top-r $QueryR --threshold $Threshold --alpha $Alpha --relevance-col joint_id --device $Device --workers $Workers
    }

    Run-Python scripts/evaluate_stage2_classification.py --manifest $RunManifest --stage1 "$Base/stage1.pt" --stage2 "$Base/stage2_128.pt" --split test --top-r $QueryR --out-dir "$Base/classification_128" --device $Device --workers $Workers
    Run-Python scripts/evaluate_mura_binary_study.py --manifest $RunManifest --predictions "$Base/classification_128/stage2_predictions.npz" --split test --out-dir "$Base/binary_study"
    Write-Host "MURA full suite completed. See $Base" -ForegroundColor Green
}
finally { Pop-Location }
