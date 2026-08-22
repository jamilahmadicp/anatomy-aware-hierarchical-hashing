[CmdletBinding()]
param(
    [string]$Manifest = $env:IRMA_MANIFEST,
    [string]$Stage1 = $env:STAGE1_CHECKPOINT,
    [string]$FullStage2 = $env:STAGE2_CHECKPOINT,
    [string]$OutDir = 'outputs/irma/ablations',
    [string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { 'auto' }),
    [int]$Workers = $(if ($env:WORKERS) { [int]$env:WORKERS } else { 0 }),
    [int]$Epochs = $(if ($env:EPOCHS2) { [int]$env:EPOCHS2 } else { 40 }),
    [int]$Seed = $(if ($env:SEED) { [int]$env:SEED } else { 42 }),
    [double]$Threshold = 0.8,
    [int]$TopR = 2,
    [double]$Alpha = 0.05
)
if ([string]::IsNullOrWhiteSpace($Manifest)) { throw 'Set IRMA_MANIFEST or pass -Manifest.' }
if ([string]::IsNullOrWhiteSpace($Stage1)) { throw 'Set STAGE1_CHECKPOINT or pass -Stage1.' }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) { $env:PYTHONPATH = $ScriptDir } else { $env:PYTHONPATH = "$ScriptDir;$env:PYTHONPATH" }
Push-Location $ScriptDir
try {
$ArgsList = @('scripts/run_ablation_suite.py','--manifest',$Manifest,'--stage1',$Stage1,'--out-dir',$OutDir,'--bits','128','--epochs',$Epochs,'--seed',$Seed,'--device',$Device,'--workers',$Workers,'--relevance-col','fine_id','--threshold',$Threshold,'--top-r',$TopR,'--alpha',$Alpha)
if (-not [string]::IsNullOrWhiteSpace($FullStage2)) { $ArgsList += @('--full-stage2',$FullStage2) }
& python @ArgsList
if ($LASTEXITCODE -ne 0) { throw "Ablation suite failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
