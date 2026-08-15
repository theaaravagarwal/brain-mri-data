#!/usr/bin/env python3
"""Monitor an RTX CUDA glioma run without changing its process state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_RUN = Path("runs/glioma-pilot--cuda--brats--20260812--e10")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def gpu() -> dict[str, str]:
    fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=10,
    )
    if result.returncode:
        return {"error": result.stderr.strip() or "nvidia-smi failed"}
    values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    if len(values) != 6:
        return {"error": "unexpected nvidia-smi output"}
    return dict(zip(("name", "utilization", "used", "total", "temperature", "power"), values, strict=True))


def processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,pcpu,pmem,rss,etime,args", "--no-headers"],
        text=True, capture_output=True, check=True,
    )
    return [line for line in result.stdout.splitlines() if "training/train_glioma.py" in line]


def memory() -> tuple[float, float]:
    values = {line.split(":", 1)[0]: int(line.split(":", 1)[1].split()[0]) for line in Path("/proc/meminfo").read_text().splitlines()}
    total = values["MemTotal"] / 1024**2
    return total, total - values["MemAvailable"] / 1024**2


def latest_metric(run: Path) -> str:
    path = run / "metrics.jsonl"
    if not path.exists() or not path.stat().st_size:
        return "waiting for first completed epoch"
    metric = json.loads(path.read_text().splitlines()[-1])
    validation = metric.get("validation", {})
    return (
        f"epoch={metric.get('epoch')} loss={metric.get('train_loss'):.4f} "
        f"val_dice={validation.get('mean_dice', float('nan')):.4f}"
    )


def render(run: Path) -> str:
    total, used = memory()
    values = gpu()
    lines = [
        f"CUDA training monitor — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Linux RAM: {used:.1f}/{total:.1f} GiB used",
        "Trainer processes:",
        *(processes() or ["  not running"]),
    ]
    if "error" in values:
        lines.append(f"GPU: {values['error']}")
    else:
        lines.append(
            f"GPU: {values['name']} | {values['utilization']}% | "
            f"{values['used']}/{values['total']} MiB | {values['temperature']} C | {values['power']} W"
        )
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
