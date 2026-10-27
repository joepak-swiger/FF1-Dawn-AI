$ErrorActionPreference = 'Stop'
$testRoot = Join-Path $PSScriptRoot '.gba_launcher_mock_test'
$testsRoot = (Resolve-Path $PSScriptRoot).Path.TrimEnd('\') + '\'
if (Test-Path -LiteralPath $testRoot) { throw "Refusing to reuse or delete existing test directory: $testRoot" }
New-Item -ItemType Directory -Path $testRoot | Out-Null
try {
    $mockWsl = Join-Path $testRoot 'mock-wsl.cmd'
    $logPath = Join-Path $testRoot 'calls.txt'
    @'
@echo off
>>"%FF1_MOCK_WSL_LOG%" echo %~1^|%~2^|%~3^|%~4^|%~5
exit /b %FF1_MOCK_WSL_EXIT%
'@ | Set-Content -LiteralPath $mockWsl -Encoding Ascii
    $launcher = (Resolve-Path (Join-Path $PSScriptRoot '..\run_gba_feasibility.cmd')).Path
    function Invoke-Launcher([string] $Mode, [int] $MockExit) {
        $psi = [System.Diagnostics.ProcessStartInfo]::new('cmd.exe')
        $psi.Arguments = if ($Mode) { "/d /c call `"$launcher`" $Mode <NUL" } else { "/d /c call `"$launcher`" <NUL" }
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.EnvironmentVariables['FF1_WSL_EXE'] = $mockWsl
        $psi.EnvironmentVariables['FF1_MOCK_WSL_LOG'] = $logPath
        $psi.EnvironmentVariables['FF1_MOCK_WSL_EXIT'] = [string]$MockExit
        $psi.EnvironmentVariables['FF1_NO_PAUSE'] = '1'
        $process = [System.Diagnostics.Process]::Start($psi)
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        [pscustomobject]@{ ExitCode = $process.ExitCode; Output = $stdout + $stderr }
    }
    $default = Invoke-Launcher '' 0
    if ($default.ExitCode -ne 0) { throw "Default invocation failed: $($default.Output)" }
    $dryRun = Invoke-Launcher '--dry-run' 0
    if ($dryRun.ExitCode -ne 0) { throw "Dry-run invocation failed: $($dryRun.Output)" }
    $failed = Invoke-Launcher '--dry-run' 47
    if ($failed.ExitCode -ne 47 -or $failed.Output -notmatch 'exit code 47') { throw "WSL failure was not propagated: $($failed.Output)" }
    $beforeInvalid = (Get-Content -LiteralPath $logPath).Count
    $invalid = Invoke-Launcher '--unknown' 0
    if ($invalid.ExitCode -ne 2 -or $invalid.Output -notmatch 'Usage: run_gba_feasibility.cmd') { throw "Invalid argument was not rejected: $($invalid.Output)" }
    $calls = @(Get-Content -LiteralPath $logPath)
    $expected = '/mnt/e/JOEPAK/FF1-Dawn-AI/scripts/run_gba_feasibility.sh'
    if ($calls.Count -ne 3 -or $calls[0] -ne "--exec|bash|$expected|run|" -or $calls[1] -ne "--exec|bash|$expected|--dry-run|" -or $calls[2] -ne "--exec|bash|$expected|--dry-run|") {
        throw "Unexpected simulated WSL argv: $($calls -join '; ')"
    }
    if ($calls.Count -ne $beforeInvalid) { throw 'Invalid mode should not invoke WSL.' }
    Write-Output 'PASS: GBA launcher default/dry-run argv, invalid arguments, and WSL exit propagation.'
}
finally {
    $resolvedTestRoot = (Resolve-Path $testRoot).Path
    if (-not $resolvedTestRoot.StartsWith($testsRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Refusing to remove path outside tests directory: $resolvedTestRoot" }
    Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
}
