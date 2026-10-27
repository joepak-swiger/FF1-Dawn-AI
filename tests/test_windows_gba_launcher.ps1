$ErrorActionPreference = 'Stop'
$testRoot = Join-Path $PSScriptRoot '.gba_launcher_mock_test'
$testsRoot = (Resolve-Path $PSScriptRoot).Path.TrimEnd('\') + '\'
if (Test-Path -LiteralPath $testRoot) { throw "Refusing to reuse or delete existing test directory: $testRoot" }
New-Item -ItemType Directory -Path (Join-Path $testRoot 'project\scripts') -Force | Out-Null
try {
    $fixtureScripts = Join-Path $testRoot 'project\scripts'
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot '..\scripts\run_gba_feasibility.ps1') -Destination $fixtureScripts
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot '..\scripts\normalize_bash_line_endings.ps1') -Destination $fixtureScripts
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot '..\run_gba_feasibility.cmd') -Destination (Join-Path $testRoot 'project')
    $fixtureRunner = Join-Path $fixtureScripts 'run_gba_feasibility.sh'
    $fixtureNesRunner = Join-Path $fixtureScripts 'run_nes_ram_experiment.sh'
    function Write-CrlfRunner([bool] $WithBom) {
        $text = "#!/usr/bin/env bash`r`nset -Eeuo pipefail`r`necho CRLF_RUNNER_EXECUTED`r`n"
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
        if ($WithBom) { $bytes = [byte[]](@(0xEF, 0xBB, 0xBF) + $bytes) }
        [System.IO.File]::WriteAllBytes($fixtureRunner, $bytes)
    }
    Write-CrlfRunner $true
    [System.IO.File]::WriteAllBytes($fixtureNesRunner, [System.Text.Encoding]::ASCII.GetBytes("#!/usr/bin/env bash`r`nset -Eeuo pipefail`r`n"))
    $gitBash = Join-Path $env:ProgramFiles 'Git\bin\bash.exe'
    $initialBytes = [System.IO.File]::ReadAllBytes($fixtureRunner)
    $initialText = [System.Text.Encoding]::UTF8.GetString($initialBytes)
    if (-not $initialText.Contains("`r`n") -or $initialBytes[0] -ne 0xEF) { throw 'Fixture was not created with both CRLF and a UTF-8 BOM.' }

    $mockWsl = Join-Path $testRoot 'mock-wsl.cmd'
    $logPath = Join-Path $testRoot 'calls.txt'
    @'
@echo off
>>"%FF1_MOCK_WSL_LOG%" echo %~1^|%~2^|%~3^|%~4^|%~5
exit /b %FF1_MOCK_WSL_EXIT%
'@ | Set-Content -LiteralPath $mockWsl -Encoding Ascii
    $launcher = (Resolve-Path (Join-Path $testRoot 'project\run_gba_feasibility.cmd')).Path
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
    $runnerBytes = [System.IO.File]::ReadAllBytes($fixtureRunner)
    if (($runnerBytes | Where-Object { $_ -eq 13 }).Count -ne 0 -or ($runnerBytes.Length -ge 3 -and $runnerBytes[0] -eq 0xEF)) {
        throw 'Launcher did not normalize the CRLF/BOM runner before invoking WSL.'
    }
    if ($runnerBytes.Length -eq 0 -or [System.Text.Encoding]::UTF8.GetString($runnerBytes) -notmatch "set -Eeuo pipefail`necho CRLF_RUNNER_EXECUTED") {
        throw 'Normalized runner content was not preserved as LF text.'
    }
    if ([System.IO.File]::ReadAllText($fixtureNesRunner).Contains("`r")) { throw 'Launcher did not normalize the NES runner in the existing checkout.' }
    if (Test-Path -LiteralPath $gitBash) {
        & $gitBash -n $fixtureRunner
        if ($LASTEXITCODE -ne 0) { throw 'Git Bash syntax validation failed for normalized fixture runner.' }
        $bashOutput = & $gitBash $fixtureRunner
        if ($LASTEXITCODE -ne 0 -or ($bashOutput -join "`n") -notmatch 'CRLF_RUNNER_EXECUTED') {
            throw 'Normalized fixture runner did not execute through Git Bash.'
        }
        foreach ($source in @('run_gba_feasibility.sh', 'run_nes_ram_experiment.sh')) {
            $path = Join-Path $PSScriptRoot "..\scripts\$source"
            & $gitBash -n $path
            if ($LASTEXITCODE -ne 0) { throw "Bash syntax validation failed for $source." }
        }
    } else {
        Write-Warning 'Git Bash is unavailable; CRLF normalization was tested, but bash -n was skipped.'
    }

    Write-CrlfRunner $false
    $dryRun = Invoke-Launcher '--dry-run' 0
    if ($dryRun.ExitCode -ne 0) { throw "Dry-run invocation failed: $($dryRun.Output)" }
    if (([System.IO.File]::ReadAllText($fixtureRunner)).Contains("`r")) { throw 'Dry-run launcher invocation did not normalize a CRLF runner.' }
    $failed = Invoke-Launcher '--dry-run' 47
    if ($failed.ExitCode -ne 47 -or $failed.Output -notmatch 'exit code 47') { throw "WSL failure was not propagated: $($failed.Output)" }
    $startFailure = Invoke-Launcher '--dry-run' -1
    if ($startFailure.ExitCode -ne 1 -or $startFailure.Output -notmatch 'WSL could not start' -or $startFailure.Output -notmatch 'no experiment log exists') {
        throw "Negative WSL startup failure was not explained and normalized: $($startFailure.Output)"
    }
    $beforeInvalid = (Get-Content -LiteralPath $logPath).Count
    $invalid = Invoke-Launcher '--unknown' 0
    if ($invalid.ExitCode -ne 2 -or $invalid.Output -notmatch 'Usage: run_gba_feasibility.cmd') { throw "Invalid argument was not rejected: $($invalid.Output)" }
    $calls = @(Get-Content -LiteralPath $logPath)
    $fixtureProject = (Resolve-Path (Join-Path $testRoot 'project')).Path
    $fixtureDrive = [System.IO.Path]::GetPathRoot($fixtureProject)
    $expected = "/mnt/$($fixtureDrive.Substring(0, 1).ToLowerInvariant())/$($fixtureProject.Substring($fixtureDrive.Length).Replace('\', '/'))/scripts/run_gba_feasibility.sh"
    if ($calls.Count -ne 4 -or $calls[0] -ne "--exec|bash|$expected|run|" -or $calls[1] -ne "--exec|bash|$expected|--dry-run|" -or $calls[2] -ne "--exec|bash|$expected|--dry-run|" -or $calls[3] -ne "--exec|bash|$expected|--dry-run|") {
        throw "Unexpected simulated WSL argv: $($calls -join '; ')"
    }
    if ($calls.Count -ne $beforeInvalid) { throw 'Invalid mode should not invoke WSL.' }
    Write-Output 'PASS: GBA launcher CRLF/BOM repair, default/dry-run argv, invalid arguments, and WSL failure reporting.'
}
finally {
    $resolvedTestRoot = (Resolve-Path $testRoot).Path
    if (-not $resolvedTestRoot.StartsWith($testsRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Refusing to remove path outside tests directory: $resolvedTestRoot" }
    Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
}
