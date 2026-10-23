param(
    [ValidateRange(1, 10)][int]$MaxTries = 10
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
Push-Location -LiteralPath $repoRoot
try {
    & $python -m gap_libero.benchmarks.vab prepare
    if ($LASTEXITCODE -ne 0) { throw 'Published source preparation failed; no generation started' }
    & $python -m gap_libero.benchmarks.vab smoke
    if ($LASTEXITCODE -ne 0) {
        throw 'Published bindings are not ready. Inspect cache/vab-readiness before using generation allowance.'
    }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    foreach ($scenario in @('fulfillment', 'packing', 'popcorn')) {
        $session = Join-Path $repoRoot "results/harness/published-$scenario-$stamp"
        & $python -m gap_harness.cli iterate --scenario "published-$scenario" `
            --session $session --live --execute --max-attempts $MaxTries
        if ($LASTEXITCODE -ne 0) { throw "Published $scenario stopped. Inspect $session before resuming." }
        & $python visualizer/libero_export.py $session
        if ($LASTEXITCODE -ne 0) { throw "Published $scenario visualization failed. Graphs remain in $session" }
    }
} finally {
    Pop-Location
}
