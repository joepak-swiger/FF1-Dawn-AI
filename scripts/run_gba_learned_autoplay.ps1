$ErrorActionPreference = 'Stop'

if ($args.Count -gt 1 -or ($args.Count -eq 1 -and $args[0] -ne '--dry-run')) {
    [Console]::Error.WriteLine('Usage: run_gba_learned_autoplay.cmd [--dry-run]')
    exit 2
}
$mode = if ($args.Count -eq 0) { 'learned' } else { 'learned-dry-run' }
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$bashRunner = Join-Path $projectRoot 'scripts\run_gba_autoplay.sh'
try {
    & (Join-Path $PSScriptRoot 'normalize_bash_line_endings.ps1') -Paths @($bashRunner) | Out-Host
} catch {
    [Console]::Error.WriteLine("ERROR: Could not normalize the GBA autoplay runner: $($_.Exception.Message)")
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
$runnerPath = "$wslProject/scripts/run_gba_autoplay.sh"
$wslExe = if ($env:FF1_WSL_EXE) { $env:FF1_WSL_EXE } else { 'wsl.exe' }
$wslArguments = @('--exec', 'bash', $runnerPath, $mode)
Write-Host "Starting learned GBA autoplay through WSL ($mode)..."
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
    [Console]::Error.WriteLine("ERROR: WSL could not start Bash (native exit code $runnerExit). The runner did not start, so no experiment log exists.")
    exit 1
}
if ($runnerExit -ne 0) {
    [Console]::Error.WriteLine("ERROR: Learned GBA autoplay runner returned exit code $runnerExit.")
    exit $runnerExit
}
exit 0
