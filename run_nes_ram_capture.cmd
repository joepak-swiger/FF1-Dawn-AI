@echo off
setlocal EnableExtensions

if "%~2" NEQ "" goto :usage
if "%~1"=="" (
    set "MODE=run"
) else if /I "%~1"=="--dry-run" (
    set "MODE=--dry-run"
) else (
    goto :usage
)

for /f "usebackq delims=" %%P in (`wsl.exe --exec wslpath -a "%~dp0"`) do set "WSL_PROJECT=%%P"
if not defined WSL_PROJECT (
    echo ERROR: Could not locate the project path inside WSL. Check that WSL is installed and available.
    pause
    exit /b 2
)

echo Running NES RAM capture through the project's WSL virtual environment...
wsl.exe --exec bash -lc "cd '%WSL_PROJECT%' && exec bash scripts/run_nes_ram_experiment.sh %MODE%"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo.
    echo ERROR: Experiment launcher failed with exit code %RESULT%. See videos/nes_ram_auto_capture/logs for details.
    pause
    exit /b %RESULT%
)
echo.
echo Experiment and analysis completed. See the summary and log path printed above.
pause
exit /b 0

:usage
echo Usage: %~nx0 [--dry-run]
echo Double-click to run three 1800-decision captures and analyze them.
echo Pass --dry-run from a terminal to validate WSL, dependencies, and local assets without starting the emulator.
exit /b 2
