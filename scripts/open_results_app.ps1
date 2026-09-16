$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found. Follow docs/INSTALLATION.md first."
}
& $python -m meshgraphnet_surface.app --config configs/experiment.yaml --host 127.0.0.1 --port 8054
