param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateSet('grocery', 'fulfillment', 'packing', 'popcorn')][string]$Scenario,
    [ValidateRange(1, 10)][int]$MaxTries = 10,
    [string]$Session
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $repoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'The project Python environment is missing' }
$taskName = if ($Scenario -eq 'grocery') { 'libero-grocery-local-vlm' } else { "libero-$Scenario-local-qwen" }
$taskPath = Join-Path $repoRoot "harness/tasks/$taskName.json"
$task = Get-Content -LiteralPath $taskPath -Raw | ConvertFrom-Json
if ((($task.seeds | Sort-Object) -join ',') -ne '0,1,2,3,4,5,6,7,8,9') {
    throw 'This runner requires the full ten seed task catalog'
}
$maxSubgraphs = 8
$maxCalls = $MaxTries * ($maxSubgraphs + 2)
if ($Session) {
    $Session = (Resolve-Path -LiteralPath $Session).Path
    $manifest = Get-Content -LiteralPath (Join-Path $Session 'session.json') -Raw | ConvertFrom-Json
    $maxSubgraphs = [int]$manifest.max_subgraphs
    $maxCalls = [int]$manifest.max_calls
}
$arguments = @('-m', 'gap_harness.cli', 'iterate', '--task', $taskPath,
               '--live', '--execute', '--rounds', "$MaxTries", '--max-attempts', "$MaxTries",
               '--max-subgraphs', "$maxSubgraphs", '--max-calls', "$maxCalls")
if ($Session) { $arguments += @('--session', $Session) }
Write-Host "Stop at 10/10 or $MaxTries total candidates. Model call budget: $maxCalls for this session."
Write-Host 'Keep Qwen and the Codex proxy running. Close Isaac and unrelated model workers.'
Push-Location -LiteralPath $repoRoot
try {
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Harness stopped. Inspect the saved error before resuming the same session.' }
} finally {
    Pop-Location
}
