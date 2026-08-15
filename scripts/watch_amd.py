#!/usr/bin/env python3
"""Monitor an AMD ROCm training run in WSL2 using Linux and Windows counters."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_RUN = Path("runs/glioma-pilot--amd--brats--20260812")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def powershell() -> str | None:
    candidates = [
        shutil.which("powershell.exe"),
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
    ]
    return next((candidate for candidate in candidates if candidate and Path(candidate).exists()), None)


def windows_gpu() -> dict[str, object]:
    executable = powershell()
    if executable is None:
        return {"error": "Windows PowerShell interop is unavailable"}
    command = r"""
      $adapter = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory |
        Sort-Object DedicatedUsage -Descending | Select-Object -First 1
      if ($null -eq $adapter) { throw 'No GPU adapter memory counter found' }
      $id = $adapter.Name -replace '^luid_', ''
      $engines = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine |
        Where-Object { $_.Name -like "*$id*" }
      $compute = $engines | Where-Object { $_.Name -like '*engtype_Compute*' } |
        Measure-Object UtilizationPercentage -Maximum
      $all = $engines | Measure-Object UtilizationPercentage -Maximum
      $computeValue = if ($null -eq $compute.Maximum) { 0 } else { $compute.Maximum }
      $busiestValue = if ($null -eq $all.Maximum) { 0 } else { $all.Maximum }
      [PSCustomObject]@{
        adapter = $adapter.Name
        dedicated_bytes = [Int64]$adapter.DedicatedUsage
        shared_bytes = [Int64]$adapter.SharedUsage
        compute_percent = [double]$computeValue
        busiest_engine_percent = [double]$busiestValue
      } | ConvertTo-Json -Compress
    """
    result = subprocess.run(
        [executable, "-NoProfile", "-Command", command], text=True, capture_output=True, timeout=12
    )
    if result.returncode:
        return {"error": result.stderr.strip() or "Windows GPU counter query failed"}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "Windows GPU counter output was not JSON"}


def linux_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,pcpu,pmem,rss,etime,args", "--no-headers"],
        text=True, capture_output=True, check=True,
    )
    return [line for line in result.stdout.splitlines() if "training/train_glioma.py" in line]


def memory() -> tuple[float, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0])
    total = values["MemTotal"] / 1024**2
    available = values["MemAvailable"] / 1024**2
    return total, total - available


def gib(value: object) -> str:
    return f"{int(value) / 1024**3:.2f} GiB"


def latest_metric(run: Path) -> str:
    path = run / "metrics.jsonl"
    if not path.exists() or not path.stat().st_size:
        return "waiting for first completed epoch"
    return path.read_text().splitlines()[-1]


def render(run: Path) -> str:
    total, used = memory()
    gpu = windows_gpu()
    lines = [
        f"AMD WSL training monitor — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Linux RAM: {used:.1f}/{total:.1f} GiB used",
        "Trainer processes:",
        *(linux_processes() or ["  not running"]),
    ]
    if "error" in gpu:
        lines.append(f"Windows GPU: {gpu['error']}")
    else:
        lines.extend([
            f"Windows GPU adapter: {gpu['adapter']}",
            f"GPU VRAM: {gib(gpu['dedicated_bytes'])} dedicated, {gib(gpu['shared_bytes'])} shared",
            f"GPU activity: compute {gpu['compute_percent']:.0f}%, busiest engine {gpu['busiest_engine_percent']:.0f}%",
        ])
    lines.append(f"Latest metric: {latest_metric(run)}")
    return "\n".join(lines)


def main() -> None:
    args = arguments()
    try:
        while True:
            if sys.stdout.isatty():
                os.system("clear")
            print(render(args.run), flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
