$ErrorActionPreference = 'Stop'

if ($args.Count -gt 1 -or ($args.Count -eq 1 -and $args[0] -notin @('run', '--dry-run'))) {
    [Console]::Error.WriteLine('Usage: run_nes_ram_capture.cmd [--dry-run]')
    exit 2
}
$mode = if ($args.Count -eq 0) { 'run' } elseif ($args[0] -eq '--dry-run') { '--dry-run' } else { 'run' }

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$bashRunnerPaths = @(
    (Join-Path $projectRoot 'scripts\run_gba_feasibility.sh'),
    (Join-Path $projectRoot 'scripts\run_nes_ram_experiment.sh')
)
try {
    & (Join-Path $PSScriptRoot 'normalize_bash_line_endings.ps1') -Paths $bashRunnerPaths | Out-Host
} catch {
    [Console]::Error.WriteLine("ERROR: Could not normalize Bash runner line endings: $($_.Exception.Message)")
    exit 2
}
$driveRoot = [System.IO.Path]::GetPathRoot($projectRoot)
if (-not $driveRoot -or $driveRoot -notmatch '^([A-Za-z]):\\$') {
    [Console]::Error.WriteLine("ERROR: Expected a drive-mounted project path; got '$projectRoot'.")
    exit 2
}
$driveLetter = $Matches[1].ToLowerInvariant()
$relativeProject = $projectRoot.Substring($driveRoot.Length).Replace('\', '/')
$wslProject = "/mnt/$driveLetter/$relativeProject"
$runnerPath = "$wslProject/scripts/run_nes_ram_experiment.sh"

$wslExe = if ($env:FF1_WSL_EXE) { $env:FF1_WSL_EXE } else { 'wsl.exe' }
$wslArguments = @('--exec', 'bash', $runnerPath, $mode)
Write-Host "Starting FF1 NES runner in WSL ($mode)..."
Write-Host "Project path: $wslProject"
try {
    & $wslExe @wslArguments
    $runnerExit = $LASTEXITCODE
    if ($null -eq $runnerExit) { $runnerExit = 0 }
} catch {
    [Console]::Error.WriteLine("ERROR: Could not start WSL command '$wslExe': $($_.Exception.Message)")
    [Console]::Error.WriteLine('No experiment log exists because the Bash runner did not start.')
    exit 2
}

if ($runnerExit -lt 0) {
    [Console]::Error.WriteLine("ERROR: WSL could not start the Linux process (native exit code $runnerExit). The Bash runner did not start, so no experiment log exists.")
    exit 1
}
if ($runnerExit -ne 0) {
    [Console]::Error.WriteLine("ERROR: WSL/Bash returned exit code $runnerExit.")
    [Console]::Error.WriteLine('Review the WSL or runner error above. A run log is created only after the Bash runner starts.')
    exit $runnerExit
}
exit 0
