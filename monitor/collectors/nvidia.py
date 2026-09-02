import json
import os
import shlex
import shutil
import socket
import subprocess
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

REPO = Path.home() / "Documents" / ".aarav" / "brain"
RUNS = REPO / "runs"
QUEUE_STATUS = RUNS / "queue-logs" / "prototype-cnn-rtx4060-long.status.json"
QUEUE_UNIT = "brain-mri-cnn-4060-queue.service"


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


def text_contains(path, needle):
    try:
        return needle in path.read_text(errors="replace")
    except OSError:
        return False


def metric_rows(path, limit=100):
    try:
        with path.open() as handle:
            lines = deque(handle, maxlen=limit)
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            row = json.loads(line)
            validation = row.get("validation", {})
            result.append({
                "epoch": row.get("epoch"),
                "trainLoss": row.get("train_loss"),
                "meanDice": validation.get("mean_dice"),
                "meanBoxIou": validation.get("mean_derived_box_iou"),
                "meanHd95Mm": validation.get("mean_hd95_mm"),
            })
        except (ValueError, TypeError):
            continue
    return result


def process_lines():
    result = command(["ps", "-eo", "args", "--no-headers"])
    markers = ("training/train_glioma.py", "nnUNetv2_train")
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if any(marker in line for marker in markers)
    ]


def active_output(processes):
    for line in processes:
        try:
            parts = shlex.split(line)
            path = Path(parts[parts.index("--output") + 1])
            return path if path.is_absolute() else REPO / path
        except (ValueError, IndexError):
            pass
    return None


def nnunet_snapshot(processes):
    for line in processes:
        if "nnUNetv2_train" not in line:
            continue
        try:
            parts = shlex.split(line)
            executable_index = next(
                index for index, part in enumerate(parts) if part.endswith("nnUNetv2_train")
            )
            dataset_id = parts[executable_index + 1]
            trainer = parts[parts.index("-tr") + 1]
        except (ValueError, IndexError, StopIteration):
            continue
        candidates = sorted(
            RUNS.glob("overnight/product-v2-*/*/state.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        job_state = next(
            (
                path.parent
                for path in candidates
                if text_contains(path, f"trainer={trainer}")
            ),
            None,
        )
        if job_state is None:
            return {
                "name": f"Dataset{dataset_id} · {trainer}",
                "status": "running",
                "profile": "nnU-Net ResEnc M",
                "metrics": [],
            }
        train_log = job_state / "train.log"
        try:
            tail = train_log.read_text(errors="replace")[-64_000:]
            epochs = [int(value) for value in re.findall(r"\bEpoch\s+(\d+)\b", tail)]
        except OSError:
            epochs = []
        job_id = job_state.name
        fields = job_id.split("--")
        epoch_total = 1 if trainer.endswith("Smoke") else 50 if trainer.endswith("50") or trainer.endswith("Screen") else 250 if trainer.endswith("250") else None
        return {
            "name": job_id,
            "status": "running",
            "arm": fields[0] if fields else None,
            "seed": int(fields[-1]) if fields and fields[-1].isdigit() else None,
            "profile": "nnU-Net ResEnc M",
            "epoch": max(epochs) + 1 if epochs else None,
            "epochs": epoch_total,
            "phase": fields[1] if len(fields) > 2 else None,
            "updatedAt": datetime.fromtimestamp(train_log.stat().st_mtime, timezone.utc).isoformat() if train_log.exists() else None,
            "metrics": [],
        }
    return None


def run_snapshot(path, running=False):
    record = read_json(path / "run.json")
    progress = read_json(path / "progress.json")
    metrics = metric_rows(path / "metrics.jsonl")
    complete = (path / "external.json").exists()
    unit = "batch" if "batches_total" in progress else "case" if "cases_total" in progress else None
    return {
        "name": path.name,
        "status": "running" if running else "complete" if complete else "unknown",
        "arm": record.get("arm"),
        "seed": record.get("seed"),
        "profile": record.get("profile_id") or record.get("profile"),
        "epoch": progress.get("epoch") or (metrics[-1]["epoch"] if metrics else None),
        "epochs": progress.get("epochs") or record.get("epochs"),
        "phase": progress.get("phase"),
        "completed": progress.get("batches_complete") if unit == "batch" else progress.get("cases_complete") if unit == "case" else None,
        "total": progress.get("batches_total") if unit == "batch" else progress.get("cases_total") if unit == "case" else None,
        "unit": unit,
        "liveLoss": progress.get("train_loss"),
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
        result.append({
            "name": run_dir.name,
            "seed": record.get("seed"),
            "status": "complete" if (run_dir / "external.json").exists() else "incomplete",
            "bestEpoch": best.get("epoch"),
            "bestDice": best.get("meanDice"),
            "bestBoxIou": best.get("meanBoxIou"),
            "bestHd95Mm": best.get("meanHd95Mm"),
            "modifiedAt": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        })
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


def gpu():
    fields = ["name", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw"]
    executable = shutil.which("nvidia-smi")
    if executable is None and Path("/usr/bin/nvidia-smi").exists():
        executable = "/usr/bin/nvidia-smi"
    if executable is None and Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    result = command([executable or "nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    try:
        values = [part.strip() for part in result.stdout.strip().split(",")]
        number = lambda value: float(value) if value not in {"", "N/A", "[N/A]"} else None
        utilization = number(values[1])
        return {
            "name": values[0], "utilizationPercent": utilization,
            "memoryUsedMib": number(values[2]), "memoryTotalMib": number(values[3]),
            "temperatureC": number(values[4]), "powerW": number(values[5]),
            "active": utilization is not None and utilization > 0,
        }
    except (IndexError, ValueError):
        return {"name": "NVIDIA GeForce RTX 4060", "utilizationPercent": None, "memoryUsedMib": None, "memoryTotalMib": 8188, "temperatureC": None, "powerW": None, "active": False}


def memory():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0])
        total = values["MemTotal"] / 1024**2
        return {"usedGib": total - values["MemAvailable"] / 1024**2, "totalGib": total}
    except (OSError, KeyError, ValueError):
        return {"usedGib": None, "totalGib": None}


def disk():
    value = shutil.disk_usage(REPO)
    gib = 1024**3
    return {"usedGib": value.used / gib, "totalGib": value.total / gib, "freeGib": value.free / gib}


def sessions():
    result = command(["tmux", "list-sessions", "-F", "#{session_name}"])
    return [line for line in result.stdout.splitlines() if line][:12]


def queue_snapshot():
    value = read_json(QUEUE_STATUS)
    service = command(["systemctl", "--user", "is-active", QUEUE_UNIT]).stdout.strip() or "unknown"
    valid = (
        value.get("schemaVersion") == "research-training-queue/v1"
        and value.get("state") in {"waiting", "running", "complete", "attention"}
        and isinstance(value.get("queuedRuns"), list)
        and all(isinstance(run, str) for run in value["queuedRuns"])
        and all(isinstance(value.get(key), int) for key in ("completedCount", "totalCount", "failedCount"))
    )
    if not valid:
        return {
            "state": "attention", "detail": "Queue status unavailable", "serviceState": service,
            "currentRun": None, "queuedRuns": [], "completedCount": 0, "totalCount": 0,
            "failedCount": 0, "updatedAt": None, "lastError": "status_unavailable",
        }
    current = value.get("currentRun")
    queued = value["queuedRuns"]
    return {
        **value,
        "serviceState": service,
        "detail": current or f"{len(queued)} queued · {value['completedCount']}/{value['totalCount']} complete",
    }


processes = process_lines()
active = active_output(processes)
active_nnunet = nnunet_snapshot(processes)
session_names = sessions()
print(json.dumps({
    "id": "nvidia",
    "label": "NVIDIA worker",
    "role": "CNN training",
    "hostname": socket.gethostname(),
    "gpu": gpu(),
    "memory": memory(),
    "disk": disk(),
    "activeRun": active_nnunet or (run_snapshot(active, True) if active and active.exists() else None),
    "latestRun": latest_completed_run(active),
    "recentRuns": recent_runs(active),
    "sessions": session_names,
    "queue": queue_snapshot(),
}, separators=(",", ":")))
