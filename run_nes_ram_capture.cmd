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

if not defined FF1_WSL_EXE set "FF1_WSL_EXE=wsl.exe"
for /f "usebackq delims=" %%P in (`call "%FF1_WSL_EXE%" --exec wslpath -a "%~dp0"`) do set "WSL_PROJECT=%%P"
if not defined WSL_PROJECT (
    echo ERROR: WSL could not resolve this project path. Review the WSL error above and confirm the project's drive is mounted.
    pause
    exit /b 2
)

echo Running NES RAM capture through the project's WSL virtual environment...
call "%FF1_WSL_EXE%" --exec bash "%WSL_PROJECT%/scripts/run_nes_ram_experiment.sh" %MODE%
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo.
    echo ERROR: WSL runner failed with exit code %RESULT%. Review the WSL or runner error printed above.
    echo A run log exists only if the Bash runner started; its exact path is printed by that runner.
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
