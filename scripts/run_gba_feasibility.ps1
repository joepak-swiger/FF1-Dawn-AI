$ErrorActionPreference = 'Stop'

if ($args.Count -gt 1 -or ($args.Count -eq 1 -and $args[0] -ne '--dry-run')) {
    [Console]::Error.WriteLine('Usage: run_gba_feasibility.cmd [--dry-run]')
    exit 2
}
$mode = if ($args.Count -eq 0) { 'run' } else { '--dry-run' }

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$driveRoot = [System.IO.Path]::GetPathRoot($projectRoot)
if (-not $driveRoot -or $driveRoot -notmatch '^([A-Za-z]):\\$') {
    [Console]::Error.WriteLine("ERROR: Expected a drive-mounted project path; got '$projectRoot'.")
    exit 2
}
$driveLetter = $Matches[1].ToLowerInvariant()
$relativeProject = $projectRoot.Substring($driveRoot.Length).Replace('\', '/')
$wslProject = "/mnt/$driveLetter/$relativeProject"
$runnerPath = "$wslProject/scripts/run_gba_feasibility.sh"
$wslExe = if ($env:FF1_WSL_EXE) { $env:FF1_WSL_EXE } else { 'wsl.exe' }
$wslArguments = @('--exec', 'bash', $runnerPath, $mode)

Write-Host "Starting GBA feasibility runner in WSL ($mode)..."
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
if ($runnerExit -ne 0) {
    [Console]::Error.WriteLine("ERROR: WSL/Bash returned exit code $runnerExit.")
    [Console]::Error.WriteLine('Review the probe error above. The Bash log path is printed after the runner starts.')
    exit $runnerExit
}
exit 0
