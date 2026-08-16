"""Small, dependency-free terminal monitor for local CUDA training runs."""

from __future__ import annotations

import argparse
import json
import os
import select
import shlex
import shutil
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any


GPU_FIELDS = ("name", "utilization", "memory_used", "memory_total", "temperature", "power")


def parser_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, help="run directory; defaults to the active CUDA run")
    parser.add_argument("--interval", type=float, default=2.0, help="refresh period in seconds")
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable snapshot")


def _process_lines() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,pcpu,pmem,rss,etime,args", "--no-headers"],
        text=True, capture_output=True, check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if "training/train_glioma.py" in line]


def _active_run() -> Path | None:
    for line in _process_lines():
        try:
            parts = shlex.split(line)
            output = parts.index("--output")
            candidate = Path(parts[output + 1])
        except (ValueError, IndexError):
            continue
        if candidate.exists():
            return candidate
    candidates = [path.parent for path in Path("runs").glob("*cuda*/run.json")]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _gpu() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(('name', 'utilization.gpu', 'memory.used', 'memory.total', 'temperature.gpu', 'power.draw'))}", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=10,
    )
    if result.returncode:
        return {"error": result.stderr.strip() or "nvidia-smi failed"}
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        return {"error": "expected one NVIDIA GPU"}
    values = [value.strip() for value in rows[0].split(",")]
    return dict(zip(GPU_FIELDS, values, strict=True)) if len(values) == len(GPU_FIELDS) else {"error": "unexpected nvidia-smi output"}


def _memory() -> dict[str, float]:
    values = {line.split(":", 1)[0]: int(line.split(":", 1)[1].split()[0]) for line in Path("/proc/meminfo").read_text().splitlines()}
    total = values["MemTotal"] / 1024**2
    return {"used_gib": total - values["MemAvailable"] / 1024**2, "total_gib": total}


def _last_metric(run: Path | None) -> dict[str, Any] | None:
    if run is None:
        return None
    metrics = run / "metrics.jsonl"
    if not metrics.exists() or not metrics.stat().st_size:
        return None
    return json.loads(metrics.read_text().splitlines()[-1])


def _progress(run: Path | None) -> dict[str, Any] | None:
    if run is None:
        return None
    path = run / "progress.json"
    return json.loads(path.read_text()) if path.exists() and path.stat().st_size else None


def snapshot(run: Path | None = None) -> dict[str, Any]:
    run = run or _active_run()
    record: dict[str, Any] = {}
    if run is not None and (run / "run.json").exists():
        record = json.loads((run / "run.json").read_text())
    processes = _process_lines()
    return {
        "schema_version": 1,
        "captured_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "run": str(run) if run else None,
        "status": "running" if run and any(str(run) in process for process in processes) else "idle_or_complete",
        "gpu": _gpu(),
        "memory": _memory(),
        "processes": processes,
        "run_metadata": {key: record.get(key) for key in ("profile_id", "arm", "seed", "epochs", "hardware") if key in record},
        "progress": _progress(run),
        "last_metric": _last_metric(run),
    }


def _fit(value: str, width: int) -> str:
    return value if len(value) <= width else value[: max(width - 1, 0)] + "…"


def _panel(title: str, lines: list[str], width: int) -> list[str]:
    inner = max(width - 2, 20)
    title = _fit(title, inner - 2)
    rendered = [f"╭─ {title} " + "─" * max(inner - len(title) - 3, 0) + "╮"]
    rendered.extend(f"│ {_fit(line, inner - 1):<{inner - 1}}│" for line in lines)
    rendered.append("╰" + "─" * inner + "╯")
    return rendered


def render(data: dict[str, Any], color: bool = True) -> str:
    width = min(max(shutil.get_terminal_size((92, 30)).columns, 56), 118)
    metadata = data["run_metadata"]
    progress = data["progress"] or {}
    title = "brain-mri-data  /  CUDA monitor"
    run_label = data["run"] or "No active CUDA run found"
    gpu = data["gpu"]
    gpu_lines = [gpu["error"]] if "error" in gpu else [
        f"{gpu['name']}  ·  {gpu['utilization']}% compute",
        f"VRAM  {gpu['memory_used']} / {gpu['memory_total']} MiB",
        f"{gpu['temperature']}°C  ·  {gpu['power']} W",
    ]
    metric = data["last_metric"]
    if metric is None:
        metric_lines = ["Waiting for first completed epoch"]
    else:
        validation = metric.get("validation", {})
        metric_lines = [
            f"Epoch {metric.get('epoch')}  ·  loss {metric.get('train_loss', float('nan')):.4f}",
            f"Validation Dice {validation.get('mean_dice', float('nan')):.4f}",
            f"Box IoU {validation.get('mean_derived_box_iou', float('nan')):.4f}  ·  HD95 {validation.get('mean_hd95_mm', float('nan')):.2f} mm",
        ]
    run_lines = [
        f"Status: {data['status'].replace('_', ' ')}",
        f"Run: {run_label}",
        f"Profile: {metadata.get('profile_id', '—')}  ·  arm: {metadata.get('arm', '—')}  ·  seed: {metadata.get('seed', '—')}",
        f"RAM: {data['memory']['used_gib']:.1f} / {data['memory']['total_gib']:.1f} GiB  ·  trainer processes: {len(data['processes'])}",
    ]
    if progress:
        phase = str(progress.get("phase", "working")).replace("_", " ")
        epoch = progress.get("epoch", "—")
        epochs = progress.get("epochs", metadata.get("epochs", "—"))
        if "batches_total" in progress:
            done, total = int(progress.get("batches_complete", 0)), int(progress["batches_total"])
            run_lines.append(f"{phase.title()}: epoch {epoch}/{epochs}  ·  batch {done}/{total} ({done / total:.0%})")
            run_lines.append(f"Live loss: {float(progress.get('train_loss', float('nan'))):.4f}  ·  elapsed {float(progress.get('elapsed_seconds', 0)):.0f}s")
        elif "cases_total" in progress:
            run_lines.append(f"{phase.title()}: epoch {epoch}/{epochs}  ·  case {progress.get('cases_complete', 0)}/{progress['cases_total']}")
        else:
            run_lines.append(f"{phase.title()}: epoch {epoch}/{epochs}")
    heading = f"\x1b[1;36m{title}\x1b[0m" if color else title
    lines = [heading, f"Updated {data['captured_at_local']}  ·  r refresh  ·  q quit", ""]
    lines.extend(_panel("RUN", run_lines, width))
    lines.append("")
    lines.extend(_panel("GPU", gpu_lines, width))
    lines.append("")
    lines.extend(_panel("LATEST METRIC", metric_lines, width))
    return "\n".join(lines)


def tui(run: Path | None, interval: float) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(render(snapshot(run), color=False))
        return
    interval = max(interval, 0.2)
    original = termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            print("\x1b[2J\x1b[H" + render(snapshot(run)), flush=True)
            ready, _, _ = select.select([sys.stdin], [], [], interval)
            if ready:
                key = sys.stdin.read(1).lower()
                if key in {"q", "\x03"}:
                    return
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original)


def main(args: argparse.Namespace) -> None:
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    data = snapshot(args.run)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    elif args.once:
        print(render(data, color=sys.stdout.isatty()))
    else:
        tui(args.run, args.interval)
