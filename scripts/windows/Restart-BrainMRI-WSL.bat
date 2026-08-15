@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Restart-BrainMRI-WSL.ps1"
echo.
echo WSL restart requested. You can close this window.
pause
