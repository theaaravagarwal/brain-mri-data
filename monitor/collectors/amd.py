import json
import shlex
import shutil
import socket
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/b/brain-mri-data")
RUNS = REPO / "runs"
STATE = RUNS / "amd-cnn-exploratory-20260817"


def command(args, timeout=4):
    try:
        return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(args, 127, "", str(error))


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def metric_rows(path, limit=100):
    try:
        with path.open() as handle:
            lines = deque(handle, maxlen=limit)
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
            validation = row.get("validation", {})
            rows.append({"epoch": row.get("epoch"), "trainLoss": row.get("train_loss"), "meanDice": validation.get("mean_dice"), "meanBoxIou": validation.get("mean_derived_box_iou"), "meanHd95Mm": validation.get("mean_hd95_mm")})
        except (ValueError, TypeError):
            pass
    return rows


def process_lines():
    result = command(["ps", "-eo", "args", "--no-headers"])
    return [line.strip() for line in result.stdout.splitlines() if "train_glioma_rocm_exploratory.py" in line]


def active_output(processes):
    for line in processes:
        try:
            parts = shlex.split(line)
            return Path(parts[parts.index("--output") + 1])
        except (ValueError, IndexError):
            pass
    return None


def run_snapshot(path, running=False):
    record = read_json(path / "run.json")
    progress = read_json(path / "progress.json")
    metrics = metric_rows(path / "metrics.jsonl")
    complete = (path / "external.json").exists()
    unit = "batch" if "batches_total" in progress else "case" if "cases_total" in progress else None
    return {
        "name": path.name, "status": "running" if running else "complete" if complete else "unknown",
        "arm": record.get("arm"), "seed": record.get("seed"), "profile": record.get("profile_id") or record.get("profile"),
        "epoch": progress.get("epoch") or (metrics[-1]["epoch"] if metrics else None), "epochs": progress.get("epochs") or record.get("epochs"),
        "phase": progress.get("phase"),
        "completed": progress.get("batches_complete") if unit == "batch" else progress.get("cases_complete") if unit == "case" else None,
        "total": progress.get("batches_total") if unit == "batch" else progress.get("cases_total") if unit == "case" else None,
        "unit": unit, "liveLoss": progress.get("train_loss"),
        "updatedAt": datetime.fromtimestamp((path / "progress.json").stat().st_mtime, timezone.utc).isoformat() if (path / "progress.json").exists() else None,
        "metrics": metrics,
    }


def recent_runs(active):
    candidates = sorted(RUNS.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:40]
    result = []
    for path in candidates:
        run_dir = path.parent
        if active and run_dir == active:
            continue
        record = read_json(path)
        metrics = metric_rows(run_dir / "metrics.jsonl")
        best = max(metrics, key=lambda row: row.get("meanDice") if row.get("meanDice") is not None else -1, default={})
        result.append({"name": run_dir.name, "seed": record.get("seed"), "status": "complete" if (run_dir / "external.json").exists() else "incomplete", "bestEpoch": best.get("epoch"), "bestDice": best.get("meanDice"), "bestBoxIou": best.get("meanBoxIou"), "bestHd95Mm": best.get("meanHd95Mm"), "modifiedAt": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
        if len(result) == 6:
            break
    return result


def latest_completed_run(active):
    candidates = sorted(RUNS.glob("*/external.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for marker in candidates:
        run_dir = marker.parent
        if active and run_dir == active:
            continue
        if (run_dir / "run.json").exists() and (run_dir / "metrics.jsonl").exists():
            return run_snapshot(run_dir, False)
    return None


def memory():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1); values[key] = int(value.split()[0])
        total = values["MemTotal"] / 1024**2
        return {"usedGib": total - values["MemAvailable"] / 1024**2, "totalGib": total}
    except (OSError, KeyError, ValueError):
        return {"usedGib": None, "totalGib": None}


def sessions():
    result = command(["tmux", "list-sessions", "-F", "#{session_name}"])
    return [line for line in result.stdout.splitlines() if line][:12]


def queue_state(active, session_names):
    selections = sorted(STATE.glob("batch-selection-retry*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    selected = read_json(selections[0]).get("selected_batch_size") if selections else None
    complete = sorted(STATE.glob("full-rocm-queue-complete*.state"), key=lambda p: p.stat().st_mtime, reverse=True)
    failed = sorted(STATE.glob("benchmark-*.failed"), key=lambda p: p.stat().st_mtime, reverse=True)
    if active:
        return {"state": "running", "detail": f"batch {selected}" if selected else active.name}
    if complete:
        return {"state": "complete", "detail": complete[0].name}
    if failed:
        return {"state": "attention", "detail": failed[0].name}
    return {"state": "waiting" if session_names else "idle", "detail": session_names[0] if session_names else None}


processes = process_lines()
active = active_output(processes)
session_names = sessions()
usage = shutil.disk_usage(Path("/mnt/c"))
gib = 1024**3
print(json.dumps({
    "id": "amd", "label": "AMD worker", "role": "Language and exploratory ROCm", "hostname": socket.gethostname(),
    "gpu": {"name": "AMD Radeon RX 7900 XTX", "utilizationPercent": None, "memoryUsedMib": None, "memoryTotalMib": 24576, "temperatureC": None, "powerW": None, "active": bool(active)},
    "memory": memory(), "disk": {"usedGib": usage.used / gib, "totalGib": usage.total / gib, "freeGib": usage.free / gib},
    "activeRun": run_snapshot(active, True) if active and active.exists() else None,
    "latestRun": latest_completed_run(active),
    "recentRuns": recent_runs(active), "sessions": session_names,
    "queue": queue_state(active, session_names),
}, separators=(",", ":")))
