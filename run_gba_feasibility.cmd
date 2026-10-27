@echo off
setlocal

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_gba_feasibility.ps1" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo.
    echo ERROR: GBA launcher exited with code %RESULT%. Read the WSL or probe error above.
    echo If the Bash runner started, it prints the exact log and report paths.
    if not defined FF1_NO_PAUSE pause
    exit /b %RESULT%
)
echo GBA feasibility check finished successfully.
if not defined FF1_NO_PAUSE pause
exit /b 0
