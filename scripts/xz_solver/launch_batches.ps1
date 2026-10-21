# Launch solver batches of scripts/xz_solver/run_cases.py in parallel (main line 2026-10-20).
#
#   powershell -ExecutionPolicy Bypass -File scripts\xz_solver\launch_batches.ps1 -Config 30m -Batches "S1,S8;S2,S9;S3;S4" -CapHours 12
#
# One detached process per batch (cases in a batch run one after another in that process). Every process gets
# MKL_NUM_THREADS = OMP_NUM_THREADS = 1 and XZ_CONFIG; run_cases.py --cases refuses to start otherwise for any
# configuration but 40m, and requires subnormal flushing from the first step for every case. Each batch keeps its
# own ledger (wall_<tags>.json) and checkpoints (ckpt_<tag>), so re-running the same command resumes every batch.
# Logs: <output dir>\batch_<tags>.log / .err. Returns the process ids.
param(
    [Parameter(Mandatory = $true)][string]$Batches,
    [string]$Config = "40m",
    [double]$CapHours = 0
)
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$out = if ($Config -eq "40m") { Join-Path $repo "outputs\xz_solver" } else { Join-Path $repo ("outputs\xz_solver_" + $Config) }
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($b in ($Batches -split ";")) {
    $tags = ($b -split ",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    if (-not $tags) { continue }
    $name = ($tags -join "-")
    $cap = if ($CapHours -gt 0) { " --cap-hours $CapHours" } else { "" }
    $cmd = "`$env:MKL_NUM_THREADS='1'; `$env:OMP_NUM_THREADS='1'; `$env:XZ_CONFIG='$Config'; " +
           "& '$repo\scripts\run_local.ps1' scripts\xz_solver\run_cases.py --cases '$($tags -join ',')'$cap"
    $p = Start-Process -FilePath powershell.exe -ArgumentList '-ExecutionPolicy', 'Bypass', '-Command', $cmd `
        -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $out "batch_$name.log") -RedirectStandardError (Join-Path $out "batch_$name.err")
    "{0}  batch {1}" -f $p.Id, $name
}
