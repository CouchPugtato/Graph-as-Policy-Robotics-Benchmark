param(
    [ValidateSet('libero-object', 'libero-spatial', 'libero-goal', 'libero-10', 'libero-90', 'libero-pro-object', 'libero-pro-swap')]
    [string]$Benchmark = 'libero-object',
    [ValidateSet('list', 'capture', 'run')][string]$Mode = 'capture',
    [int]$TaskId = 0,
    [int]$InitStateId = 0,
    [int]$Seed = 0,
    [switch]$Prepare,
    [string]$Graph,
    [string]$Plan,
    [switch]$EnableMotion,
    [switch]$Record
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot '.venv-libero/Scripts/python.exe'
if ($Prepare) {
    & $python -m gap_libero.benchmarks.prepare --benchmark $Benchmark
    if ($LASTEXITCODE -ne 0) { throw 'Upstream benchmark preparation failed' }
}
$arguments = @('-m', 'gap_libero.cli', 'benchmark', $Mode, '--benchmark', $Benchmark,
               '--task-id', "$TaskId", '--init-state-id', "$InitStateId", '--seed', "$Seed")
if ($Graph) { $arguments += @('--graph', (Resolve-Path -LiteralPath $Graph).Path) }
if ($Plan) { $arguments += @('--plan', (Resolve-Path -LiteralPath $Plan).Path) }
if ($EnableMotion) { $arguments += '--enable-motion' }
if ($Record) { $arguments += '--record' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Upstream benchmark worker failed' }
