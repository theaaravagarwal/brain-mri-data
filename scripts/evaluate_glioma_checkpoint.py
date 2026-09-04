#!/usr/bin/env python3
"""Re-evaluate one saved glioma checkpoint on every locked validation case."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from training.pamc import PamcSegResNet
from training.train_glioma import (
    evaluate,
    file_sha256,
    load_profile,
    loader,
    make_transforms,
    manifest_items,
    set_seed,
    validate_cnn_accelerator,
    validate_profile_against_study,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("brats", "pooled", "pamc"), default="brats")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists() or args.workers < 0:
        raise ValueError("Output must be new and workers must be non-negative")
    study = json.loads(args.study.read_text())
    profile = load_profile(args.profile)
    validate_cnn_accelerator(profile, torch.version.hip)
    validate_profile_against_study(profile, study)
    if study.get("evaluation_status") != "pilot_internal_only":
        raise ValueError("This evaluator is restricted to locked internal validation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    set_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    expected_gpu = str(profile.get("expected_gpu_name_contains", ""))
    if expected_gpu and expected_gpu.lower() not in device_name.lower():
        raise RuntimeError(f"GPU {device_name!r} does not match {expected_gpu!r}")

    patch_size = tuple(int(value) for value in profile["patch_size"])
    records = manifest_items(study, args.data_root.resolve() / "raw", args.arm, "val")
    data = loader(records, make_transforms(False, patch_size), profile, False, workers=args.workers, persistent_workers=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PamcSegResNet(int(study["study"]["training"]["init_filters"]), len(study["study"]["train_sources"])).to(device)
    model.load_state_dict(checkpoint["model"])
    started = time.monotonic()
    metrics = evaluate(model, data, device, patch_size)
    torch.cuda.synchronize(device)
    payload = {
        "schema_version": 1,
        "evaluation_scope": "full_locked_internal_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study_sha256": file_sha256(args.study),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "profile_sha256": file_sha256(args.profile),
        "device": device_name,
        "validation_cases": len(records),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: payload[key] for key in ("checkpoint_sha256", "device", "validation_cases", "runtime_seconds")} | {key: metrics[key] for key in ("mean_dice", "mean_derived_box_iou", "mean_hd95_mm")}, sort_keys=True))


if __name__ == "__main__":
    main()
