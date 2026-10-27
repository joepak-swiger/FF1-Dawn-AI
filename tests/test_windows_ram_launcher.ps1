$ErrorActionPreference = 'Stop'
$testRoot = Join-Path $PSScriptRoot '.launcher_mock_test'
$testsRoot = (Resolve-Path $PSScriptRoot).Path.TrimEnd('\') + '\'
if (Test-Path -LiteralPath $testRoot) {
    throw "Refusing to reuse or delete existing test directory: $testRoot"
}
New-Item -ItemType Directory -Path $testRoot | Out-Null

try {
    $mockWsl = Join-Path $testRoot 'mock-wsl.cmd'
    $logPath = Join-Path $testRoot 'calls.txt'
    @'
@echo off
>>"%FF1_MOCK_WSL_LOG%" echo %~1^|%~2^|%~3^|%~4^|%~5
if /I "%~2"=="wslpath" (
  echo /mnt/e/JOEPAK/FF1-Dawn-AI
  exit /b 0
)
exit /b %FF1_MOCK_WSL_EXIT%
'@ | Set-Content -LiteralPath $mockWsl -Encoding Ascii

    $launcher = (Resolve-Path (Join-Path $PSScriptRoot '..\run_nes_ram_capture.cmd')).Path
    function Invoke-Launcher([string] $Mode, [int] $MockExit) {
        $psi = [System.Diagnostics.ProcessStartInfo]::new('cmd.exe')
        if ($Mode) {
            $psi.Arguments = "/d /c call `"$launcher`" $Mode <NUL"
        } else {
            $psi.Arguments = "/d /c call `"$launcher`" <NUL"
        }
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.EnvironmentVariables['FF1_WSL_EXE'] = $mockWsl
        $psi.EnvironmentVariables['FF1_MOCK_WSL_LOG'] = $logPath
        $psi.EnvironmentVariables['FF1_MOCK_WSL_EXIT'] = [string]$MockExit
        $process = [System.Diagnostics.Process]::Start($psi)
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        [pscustomobject]@{ ExitCode = $process.ExitCode; Output = $stdout + $stderr }
    }

    $default = Invoke-Launcher '' 0
    if ($default.ExitCode -ne 0 -or $default.Output -notmatch 'Experiment and analysis completed') {
        throw "Default launcher path failed in mock WSL (exit $($default.ExitCode)): $($default.Output)"
    }
    $dryRun = Invoke-Launcher '--dry-run' 0
    if ($dryRun.ExitCode -ne 0 -or $dryRun.Output -notmatch 'Experiment and analysis completed') {
        throw "Dry-run launcher path failed in mock WSL (exit $($dryRun.ExitCode)): $($dryRun.Output)"
    }
    $failed = Invoke-Launcher '--dry-run' 47
    if ($failed.ExitCode -ne 47 -or $failed.Output -notmatch 'exit code 47' -or $failed.Output -match 'See videos/') {
        throw "Launcher did not preserve/report mocked WSL failure (exit $($failed.ExitCode)): $($failed.Output)"
    }
    $beforeInvalid = (Get-Content -LiteralPath $logPath).Count
    $invalid = Invoke-Launcher '--not-a-mode' 0
    if ($invalid.ExitCode -ne 2 -or $invalid.Output -notmatch 'Usage: run_nes_ram_capture.cmd') {
        throw "Invalid argument was not rejected with usage help (exit $($invalid.ExitCode)): $($invalid.Output)"
    }

    $calls = Get-Content -LiteralPath $logPath
    $runnerCalls = @($calls | Where-Object { $_ -match '^--exec\|bash\|' })
    if ($runnerCalls.Count -ne 3) {
        throw "Expected three simulated Bash invocations, found $($runnerCalls.Count): $($calls -join '; ')"
    }
    $expectedScript = '/mnt/e/JOEPAK/FF1-Dawn-AI/scripts/run_nes_ram_experiment.sh'
    if ($runnerCalls[0] -ne "--exec|bash|$expectedScript|run|" -or
        $runnerCalls[1] -ne "--exec|bash|$expectedScript|--dry-run|" -or
        $runnerCalls[2] -ne "--exec|bash|$expectedScript|--dry-run|") {
        throw "Unexpected WSL argv forwarding: $($runnerCalls -join '; ')"
    }
    if ($calls.Count -ne $beforeInvalid) {
        throw 'Invalid arguments should be rejected before invoking WSL.'
    }

    Write-Output 'PASS: mocked WSL path resolution, default run, --dry-run, and exit-code propagation.'
}
finally {
    $resolvedTestRoot = (Resolve-Path $testRoot).Path
    if (-not $resolvedTestRoot.StartsWith($testsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside tests directory: $resolvedTestRoot"
    }
    Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
}
