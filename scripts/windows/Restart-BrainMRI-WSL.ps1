[CmdletBinding()]
param(
    [ValidateRange(1, 30)]
    [int]$WaitSeconds = 4
)

$ErrorActionPreference = "Stop"

Write-Host "Stopping every WSL distribution..."
wsl.exe --shutdown
Start-Sleep -Seconds $WaitSeconds

Write-Host "Starting the default WSL distribution..."
wsl.exe --exec /bin/bash -lc "true"
Start-Sleep -Seconds 2

Write-Host "WSL restart requested. Wait a few seconds for systemd and Tailscale to come back."
