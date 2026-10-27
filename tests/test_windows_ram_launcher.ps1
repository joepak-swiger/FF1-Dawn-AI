$ErrorActionPreference = 'Stop'
$testRoot = Join-Path $PSScriptRoot '.launcher_mock_test'
$testsRoot = (Resolve-Path $PSScriptRoot).Path.TrimEnd('\') + '\'
if (Test-Path -LiteralPath $testRoot) {
    throw "Refusing to reuse or delete existing test directory: $testRoot"
}
New-Item -ItemType Directory -Path (Join-Path $testRoot 'project\scripts') -Force | Out-Null

try {
    $fixtureScripts = Join-Path $testRoot 'project\scripts'
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot '..\scripts\run_nes_ram_capture.ps1') -Destination $fixtureScripts
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot '..\scripts\normalize_bash_line_endings.ps1') -Destination $fixtureScripts
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot '..\run_nes_ram_capture.cmd') -Destination (Join-Path $testRoot 'project')
    $fixtureNesRunner = Join-Path $fixtureScripts 'run_nes_ram_experiment.sh'
    $fixtureGbaRunner = Join-Path $fixtureScripts 'run_gba_feasibility.sh'
    [System.IO.File]::WriteAllBytes($fixtureNesRunner, [System.Text.Encoding]::ASCII.GetBytes("#!/usr/bin/env bash`r`nset -Eeuo pipefail`r`n"))
    [System.IO.File]::WriteAllBytes($fixtureGbaRunner, [System.Text.Encoding]::ASCII.GetBytes("#!/usr/bin/env bash`r`nset -Eeuo pipefail`r`n"))

    $mockWsl = Join-Path $testRoot 'mock-wsl.cmd'
    $logPath = Join-Path $testRoot 'calls.txt'
    @'
@echo off
>>"%FF1_MOCK_WSL_LOG%" echo %~1^|%~2^|%~3^|%~4^|%~5
exit /b %FF1_MOCK_WSL_EXIT%
'@ | Set-Content -LiteralPath $mockWsl -Encoding Ascii

    $launcher = (Resolve-Path (Join-Path $testRoot 'project\run_nes_ram_capture.cmd')).Path
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
        $psi.EnvironmentVariables['FF1_NO_PAUSE'] = '1'
        $process = [System.Diagnostics.Process]::Start($psi)
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        [pscustomobject]@{ ExitCode = $process.ExitCode; Output = $stdout + $stderr }
    }

    $default = Invoke-Launcher '' 0
    if ($default.ExitCode -ne 0 -or $default.Output -notmatch 'Launcher finished successfully') {
        throw "Default launcher path failed in mock WSL (exit $($default.ExitCode)): $($default.Output)"
    }
    if ([System.IO.File]::ReadAllText($fixtureNesRunner).Contains("`r") -or [System.IO.File]::ReadAllText($fixtureGbaRunner).Contains("`r")) {
        throw 'NES launcher did not normalize CRLF runners before invoking WSL.'
    }
    $dryRun = Invoke-Launcher '--dry-run' 0
    if ($dryRun.ExitCode -ne 0 -or $dryRun.Output -notmatch 'Launcher finished successfully') {
        throw "Dry-run launcher path failed in mock WSL (exit $($dryRun.ExitCode)): $($dryRun.Output)"
    }
    $failed = Invoke-Launcher '--dry-run' 47
    if ($failed.ExitCode -ne 47 -or $failed.Output -notmatch 'exit code 47' -or $failed.Output -match 'See videos/') {
        throw "Launcher did not preserve/report mocked WSL failure (exit $($failed.ExitCode)): $($failed.Output)"
    }
    $startFailure = Invoke-Launcher '--dry-run' -1
    if ($startFailure.ExitCode -ne 1 -or $startFailure.Output -notmatch 'WSL could not start' -or $startFailure.Output -notmatch 'no experiment log exists') {
        throw "Negative WSL startup failure was not explained and normalized: $($startFailure.Output)"
    }
    $beforeInvalid = (Get-Content -LiteralPath $logPath).Count
    $invalid = Invoke-Launcher '--not-a-mode' 0
    if ($invalid.ExitCode -ne 2 -or $invalid.Output -notmatch 'Usage: run_nes_ram_capture.cmd') {
        throw "Invalid argument was not rejected with usage help (exit $($invalid.ExitCode)): $($invalid.Output)"
    }

    $calls = Get-Content -LiteralPath $logPath
    $runnerCalls = @($calls | Where-Object { $_ -match '^--exec\|bash\|' })
    if ($runnerCalls.Count -ne 4) {
        throw "Expected four simulated Bash invocations, found $($runnerCalls.Count): $($calls -join '; ')"
    }
    $fixtureProject = (Resolve-Path (Join-Path $testRoot 'project')).Path
    $fixtureDrive = [System.IO.Path]::GetPathRoot($fixtureProject)
    $expectedScript = "/mnt/$($fixtureDrive.Substring(0, 1).ToLowerInvariant())/$($fixtureProject.Substring($fixtureDrive.Length).Replace('\', '/'))/scripts/run_nes_ram_experiment.sh"
    if ($runnerCalls[0] -ne "--exec|bash|$expectedScript|run|" -or
        $runnerCalls[1] -ne "--exec|bash|$expectedScript|--dry-run|" -or
        $runnerCalls[2] -ne "--exec|bash|$expectedScript|--dry-run|" -or
        $runnerCalls[3] -ne "--exec|bash|$expectedScript|--dry-run|") {
        throw "Unexpected WSL argv forwarding: $($runnerCalls -join '; ')"
    }
    if ($calls.Count -ne $beforeInvalid) {
        throw 'Invalid arguments should be rejected before invoking WSL.'
    }

    Write-Output 'PASS: NES launcher CRLF repair, simulated WSL argv, default/dry-run, invalid arguments, and WSL failure reporting.'
}
finally {
    $resolvedTestRoot = (Resolve-Path $testRoot).Path
    if (-not $resolvedTestRoot.StartsWith($testsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside tests directory: $resolvedTestRoot"
    }
    Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
}
