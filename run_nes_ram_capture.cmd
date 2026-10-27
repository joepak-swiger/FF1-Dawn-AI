@echo off
setlocal

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_nes_ram_capture.ps1" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo.
    echo ERROR: Launcher exited with code %RESULT%. Read the WSL or smoke-test error above.
    echo If the Bash runner started, it prints the exact run log and summary paths.
    if not defined FF1_NO_PAUSE pause
    exit /b %RESULT%
)
echo Launcher finished successfully.
if not defined FF1_NO_PAUSE pause
exit /b 0
