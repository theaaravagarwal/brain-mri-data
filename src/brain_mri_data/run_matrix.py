from __future__ import annotations

import hashlib
import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .study import load_yaml


def _matrix_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand_matrix(path: Path) -> list[dict[str, Any]]:
    matrix = load_yaml(path)
    if matrix.get("schema_version") != 1:
        raise ValueError("Unsupported run-matrix schema_version")
    profiles = matrix.get("profiles", [])
    arms = matrix.get("arms", [])
    seeds = matrix.get("seeds", [])
    if not profiles or not arms or not seeds:
        raise ValueError("Run matrix needs non-empty profiles, arms, and seeds")
    jobs = []
    for profile in profiles:
        for arm in arms:
            for seed in seeds:
                jobs.append({
                    "run_id": f"{matrix['study_id']}--{profile}--{arm}--{seed}",
                    "profile": profile,
                    "arm": arm,
                    "seed": int(seed),
                })
    return jobs


def claim_run(matrix_path: Path, run_id: str, profile: str, state_root: Path) -> dict[str, Any]:
    jobs = {job["run_id"]: job for job in expand_matrix(matrix_path)}
    try:
        job = jobs[run_id]
    except KeyError as error:
        raise ValueError(f"Unknown run ID: {run_id}") from error
    if job["profile"] != profile:
        raise ValueError(f"{run_id} is assigned to {job['profile']}, not {profile}")
    state_root.mkdir(parents=True, exist_ok=True)
    claim_path = state_root / f"{run_id}.claim.json"
    claim = {
        **job,
        "matrix_sha256": _matrix_hash(matrix_path),
        "claimed_at_utc": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
    }
    try:
        with claim_path.open("x") as file:
            json.dump(claim, file, indent=2, sort_keys=True)
            file.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"Run is already claimed: {claim_path}") from error
    return {"claim": str(claim_path), **claim}
