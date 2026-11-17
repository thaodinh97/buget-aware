param(
    [string]$Output = 'results/dt_online_confirmation',
    [string]$Config = 'configs/dt_online_confirmation.json',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$runExitCode = 1
Push-Location -LiteralPath $projectRoot
try {
    $configPath = (Resolve-Path -LiteralPath $Config).Path
    $runnerArgs = @($Output, '--config', $configPath)
    if ($Resume) { $runnerArgs += '--resume' }
    # Keep the system awake only for this process; do not keep the display on or
    # change the user's persistent power plan. The OS also clears this on exit.
    @'
import ctypes, json, runpy, sys

kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
kernel.SetThreadExecutionState.restype = ctypes.c_uint
if not kernel.SetThreadExecutionState(0x80000001):
    raise ctypes.WinError(ctypes.get_last_error())
print(json.dumps({'sleep_prevention': 'ES_SYSTEM_REQUIRED', 'active': True}), flush=True)
try:
    sys.argv[0] = 'scripts/dt/run_online_confirmation.py'
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    kernel.SetThreadExecutionState(0x80000000)
'@ | python -u - @runnerArgs
    $runExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $runExitCode
