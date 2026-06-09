<#
run_local.ps1 - Run/test the FiPy side locally using PFC 5.0's bundled
Python 2.7 (no need to launch PFC).

Why:
  PFC 5.0 ships python27 + numpy/scipy/fipy, matching the production runtime.
  This repo's PFC write-back (particle_writer) skips gracefully when itasca is
  absent, so the FiPy solve can run under this interpreter without opening PFC.

Usage:
  powershell -File scripts\run_local.ps1 scripts\smoke_test_src.py
  powershell -File scripts\run_local.ps1 -m pytest tests

Optional: override the PFC Python path
  $env:PFC_PYTHON = "D:\path\to\python27\python.exe"
#>

$ErrorActionPreference = "Stop"

# 1) Locate PFC bundled Python: env var first, then known install paths.
$candidates = @()
if ($env:PFC_PYTHON) { $candidates += $env:PFC_PYTHON }
$candidates += "D:\Program Files\Itasca\PFC500\exe64\python27\python.exe"
$candidates += "C:\Program Files\Itasca\PFC500\exe64\python27\python.exe"

$pfcPython = $null
foreach ($c in $candidates) {
    if (Test-Path $c) { $pfcPython = $c; break }
}
if (-not $pfcPython) {
    Write-Error "PFC bundled Python not found. Set `$env:PFC_PYTHON to the full path of python27\python.exe."
    exit 1
}

# 2) Put the repo root on sys.path so 'import src.*' resolves.
#    This script lives in scripts\, so the repo root is its parent.
$repoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $repoRoot

# 3) Forward all arguments to the PFC Python.
Write-Host "[run_local] python     : $pfcPython"
Write-Host "[run_local] PYTHONPATH : $repoRoot"
Write-Host "[run_local] args       : $($args -join ' ')"
Write-Host ""
& $pfcPython @args
exit $LASTEXITCODE
