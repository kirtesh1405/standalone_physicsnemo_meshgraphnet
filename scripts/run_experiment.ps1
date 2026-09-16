param(
    [ValidateSet("check-data", "prepare", "train", "evaluate", "all")]
    [string]$Stage = "all",
    [ValidateSet("train", "validation", "test")]
    [string]$Split = "test",
    [int]$Epochs = 50
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found. Follow docs/INSTALLATION.md first."
}
$env:WARP_CACHE_PATH = Join-Path $root ".warp_cache"

function Invoke-Stage([string]$name, [string[]]$arguments) {
    & $python -m meshgraphnet_surface.pipeline --config configs/experiment.yaml --stage $name @arguments
    if ($LASTEXITCODE -ne 0) { throw "$name failed." }
}

if ($Stage -eq "all") {
    Invoke-Stage "check-data" @()
    Invoke-Stage "prepare" @()
    Invoke-Stage "train" @("--epochs", $Epochs)
    Invoke-Stage "evaluate" @("--split", "validation")
    Invoke-Stage "evaluate" @("--split", "test")
} elseif ($Stage -eq "train") {
    Invoke-Stage "train" @("--epochs", $Epochs)
} elseif ($Stage -eq "evaluate") {
    Invoke-Stage "evaluate" @("--split", $Split)
} else {
    Invoke-Stage $Stage @()
}
