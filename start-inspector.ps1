param(
    [int]$Port = 8765,
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cpu',
    [switch]$Check,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$inspectorPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $inspectorPython -PathType Leaf)) {
    Write-Host 'The project Python environment is missing.'
    Write-Host 'From this repository run: uv sync --locked --extra cpu --group dev'
    Write-Host 'Then retry .\start-inspector.ps1. See docs/STARTUP.md.'
    exit 2
}
$inspectorArgs = @((Join-Path $PSScriptRoot 'scripts\run_inspector.py'), '--port', "$Port", '--device', $Device)
if ($Check) { $inspectorArgs += '--check' }
if (-not $NoBrowser -and -not $Check) { $inspectorArgs += '--open-browser' }
Write-Host 'Loading the local project environment. Model verification follows...'
& $inspectorPython @inspectorArgs
exit $LASTEXITCODE
