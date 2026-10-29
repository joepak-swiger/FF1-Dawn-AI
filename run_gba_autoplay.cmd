@echo off
setlocal

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_gba_autoplay.ps1" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo.
    echo ERROR: GBA autoplay failed with exit code %RESULT%.
    echo The launcher prints a log path only after its WSL runner starts.
    if not defined FF1_NO_PAUSE pause
    exit /b %RESULT%
)
echo GBA autoplay recordings and sanitized reports were created.
if not defined FF1_NO_PAUSE pause
exit /b 0
