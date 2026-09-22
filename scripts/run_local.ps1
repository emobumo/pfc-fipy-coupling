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

# 2b) Linear solver backend. FiPy's default here is PySparse (PCG), whose
#     native library crashed 3 of 5 long runs at 576 cells (segfault /
#     "Fatal Python error", non-deterministic). scipy ran
#     clean and reproduces the line-source engineering results
#     exactly; only the short-source toe control shifts (see cases/README.md).
#     Within the scipy backend the solver CLASS is chosen by the
#     "linear_solver" parameter (default "pcg": 1.3x pysparse on the test
#     suite; scipy's LU default was 4.5x). See equations.get_linear_solver.
#     Default to scipy; set $env:FIPY_SOLVERS yourself to override
#     (e.g. "pysparse" to reproduce pre-2026-09-22 numbers).
if (-not $env:FIPY_SOLVERS) { $env:FIPY_SOLVERS = "scipy" }

# 3) Forward all arguments to the PFC Python.
Write-Host "[run_local] python     : $pfcPython"
Write-Host "[run_local] PYTHONPATH : $repoRoot"
Write-Host "[run_local] FIPY_SOLVERS: $env:FIPY_SOLVERS"
Write-Host "[run_local] args       : $($args -join ' ')"
Write-Host ""
& $pfcPython @args
exit $LASTEXITCODE
