#!/usr/bin/env python3
"""Locked-manifest trainer for the BraTS, pooled, and PAMC glioma arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    SpatialPadd,
)

from brain_mri_data.indexer import resolve_case_path
if __package__:
    from .pamc import PamcSegResNet, mask_one_modality, modality_consistency_loss
    from .evaluation import box_iou_per_case, deterministic_mask_one_modality, dice_per_case, hd95_mm_per_case
else:
    from pamc import PamcSegResNet, mask_one_modality, modality_consistency_loss
    from evaluation import box_iou_per_case, deterministic_mask_one_modality, dice_per_case, hd95_mm_per_case


MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")


class WholeLesiond(MapTransform):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        values = torch.as_tensor(result["positive_values"], dtype=torch.float32)
        for key in self.key_iterator(result):
            result[key] = torch.isin(torch.as_tensor(result[key]), values).to(torch.float32)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="locked study manifest")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--arm", choices=("brats", "pooled", "pamc"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--init-filters", type=int, default=32)
    parser.add_argument("--validation-interval", type=int, default=1)
    return parser.parse_args()


def load_profile(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        profile = yaml.safe_load(stream)
    required = {"accelerator", "batch_size", "effective_batch_size", "num_workers", "prefetch_factor", "pin_memory", "patch_size"}
    if not isinstance(profile, dict) or profile.get("schema_version") != 1 or required - set(profile):
        raise ValueError(f"Invalid runtime profile: {path}")
    if profile["batch_size"] != 1 or profile["effective_batch_size"] % profile["batch_size"]:
        raise ValueError("This trainer requires a profile with batch size one and integral accumulation")
    return profile


def validate_profile_against_study(profile: dict[str, Any], study: dict[str, Any]) -> None:
    """Reject a runtime profile that changes a locked scientific setting."""
    locked = study["study"]
    expected_patch = [int(value) for value in locked["study_patch_size"]]
    actual_patch = [int(value) for value in profile["patch_size"]]
    if actual_patch != expected_patch:
        raise ValueError("Runtime profile patch_size must match the locked study")
    if int(profile["effective_batch_size"]) != int(locked["effective_batch_size"]):
        raise ValueError("Runtime profile effective_batch_size must match the locked study")
    if profile["mixed_precision"] != locked["training"]["mixed_precision"]:
        raise ValueError("Runtime profile mixed_precision must match the locked study")


def validate_cnn_accelerator(profile: dict[str, Any], hip_version: str | None) -> None:
    """This project reserves AMD compute for the bounded language layer."""
    if profile["accelerator"] != "cuda":
        raise ValueError("The frozen ISEF CNN study is restricted to the CUDA runtime profile")
    if hip_version is not None:
        raise ValueError("The cuda profile requires a CUDA PyTorch build")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def manifest_items(study: dict[str, Any], raw_root: Path, arm: str, split: str) -> list[dict[str, Any]]:
    mappings = study["label_mappings"]
    source_ids = sorted(study["study"]["train_sources"])
    source_index = {source_id: index for index, source_id in enumerate(source_ids)}
    records = study["external_test"] if split == "locked_test" else study["development"]
    output = []
    for item in records:
        if item["split"] != split:
            continue
        source_id = item["source_id"]
        if arm == "brats" and split != "locked_test" and source_id != "brats2020_kaggle":
            continue
        record = item["record"]
        positive = mappings[source_id]["whole_lesion"]["positive_values"]
        output.append({
            "image": [
                str(resolve_case_path(record, record["modalities"][modality], raw_root))
                for modality in MODALITY_ORDER
            ],
            "label": str(resolve_case_path(record, record["segmentation"], raw_root)),
            "positive_values": positive,
            "source_index": source_index.get(source_id, -1),
            "case_id": f"{source_id}:{record['case_id']}",
        })
    if not output:
        raise ValueError(f"No {split} cases are available for arm={arm}")
    return output


def make_transforms(training: bool, patch_size: tuple[int, int, int]) -> Compose:
    transforms: list[Any] = [
        LoadImaged(keys=("image", "label")),
        EnsureChannelFirstd(keys=("image", "label")),
        WholeLesiond(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if training:
        transforms.extend([
            SpatialPadd(keys=("image", "label"), spatial_size=patch_size),
            RandSpatialCropd(keys=("image", "label"), roi_size=patch_size, random_size=False),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=1),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=2),
            RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3),
        ])
    transforms.extend([EnsureTyped(keys=("image", "label"), dtype=torch.float32)])
    return Compose(transforms)


def loader(items: list[dict[str, Any]], transforms: Compose, profile: dict[str, Any], shuffle: bool) -> DataLoader:
    workers = int(profile["num_workers"])
    options: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(profile["pin_memory"]),
        "persistent_workers": workers > 0,
    }
    if workers:
        options["prefetch_factor"] = int(profile["prefetch_factor"])
    return DataLoader(Dataset(items, transforms), batch_size=1, shuffle=shuffle, **options)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_progress(path: Path, **values: Any) -> None:
    """Atomically publish live run state for the terminal monitor."""
    payload = {"schema_version": 1, "updated_at_unix": time.time(), **values}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def evaluate(
    model: PamcSegResNet,
    data: DataLoader,
    device: torch.device,
    patch_size: tuple[int, int, int],
    corruption_seed: int | None = None,
    progress_path: Path | None = None,
    progress_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        iterator = tqdm(
            data, total=len(data), desc=(progress_fields or {}).get("phase", "evaluate"),
            unit="case", dynamic_ncols=True, disable=not sys.stderr.isatty(), leave=False,
        )
        for case_number, batch in enumerate(iterator, start=1):
            image = batch["image"].to(device, non_blocking=True)
            case_ids = [str(case_id) for case_id in batch["case_id"]]
            masked_modalities: list[str | None] = [None] * len(case_ids)
            if corruption_seed is not None:
                image, masked_modalities = deterministic_mask_one_modality(image, case_ids, corruption_seed)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = sliding_window_inference(image, patch_size, 1, lambda values: model(values)[0], overlap=0.5)
            labels = batch["label"].to(device, non_blocking=True)
            dice = dice_per_case(logits, labels)
            box_iou = box_iou_per_case(logits, labels)
            hd95 = hd95_mm_per_case(logits, labels)
            per_case.extend({
                "case_id": case_id,
                "whole_lesion_dice": score,
                "derived_box_iou": box_score,
                "hd95_mm": hd95_score,
                **({"masked_modality": modality} if modality is not None else {}),
            } for case_id, score, box_score, hd95_score, modality in zip(case_ids, dice, box_iou, hd95, masked_modalities, strict=True))
            if progress_path is not None:
                write_progress(
                    progress_path,
                    **(progress_fields or {}),
                    cases_complete=case_number,
                    cases_total=len(data),
                )
    if not per_case:
        raise ValueError("Evaluation loader yielded no cases")
    return {
        "mean_dice": float(np.mean([item["whole_lesion_dice"] for item in per_case])),
        "mean_derived_box_iou": float(np.mean([item["derived_box_iou"] for item in per_case])),
        "mean_hd95_mm": float(np.mean([item["hd95_mm"] for item in per_case if item["hd95_mm"] is not None])) if any(item["hd95_mm"] is not None for item in per_case) else None,
        "hd95_defined_cases": sum(item["hd95_mm"] is not None for item in per_case),
        "per_case": per_case,
    }


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    if not torch.cuda.is_available():
        raise SystemExit("Selected training environment cannot access a GPU")
    validate_cnn_accelerator(profile, torch.version.hip)
    study = json.loads(args.study.read_text())
    if study.get("study", {}).get("study_id") != "glioma":
        raise ValueError("This trainer only accepts the locked glioma study")
    if args.arm not in study["study"]["arms"]:
        raise ValueError(f"Arm is not present in the locked study: {args.arm}")
    training = study["study"].get("training", {})
    required_training = {"architecture", "init_filters", "validation_interval", "optimizer", "learning_rate", "weight_decay", "mixed_precision"}
    if required_training - set(training):
        raise ValueError("Locked study has incomplete training configuration")
    if training["architecture"] != "monai_segresnet" or training["optimizer"] != "adamw" or training["mixed_precision"] != "fp16":
        raise ValueError("Locked study requests an unsupported training configuration")
    if args.init_filters != int(training["init_filters"]) or args.validation_interval != int(training["validation_interval"]):
        raise ValueError("Command-line model settings must match the locked study")
    validate_profile_against_study(profile, study)
    if study["evaluation_status"] == "external_test_locked":
        if args.epochs != int(training.get("epochs", -1)):
            raise ValueError("External-study epoch budget must match the locked study")

    set_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    patch_size = tuple(int(value) for value in profile["patch_size"])
    raw_root = args.data_root.resolve() / "raw"
    train_items = manifest_items(study, raw_root, args.arm, "train")
    val_items = manifest_items(study, raw_root, args.arm, "val")
    external_items = (
        manifest_items(study, raw_root, args.arm, "locked_test")
        if study.get("external_test")
        else []
    )
    train_loader = loader(train_items, make_transforms(True, patch_size), profile, True)
    val_loader = loader(val_items, make_transforms(False, patch_size), profile, False)
    external_loader = (
        loader(external_items, make_transforms(False, patch_size), profile, False)
        if external_items
        else None
    )

    source_count = len(study["study"]["train_sources"])
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    expected_gpu = str(profile.get("expected_gpu_name_contains", ""))
    if expected_gpu and expected_gpu.lower() not in device_name.lower():
        raise SystemExit(f"Runtime GPU {device_name!r} does not match profile expectation {expected_gpu!r}")
    model = PamcSegResNet(args.init_filters, source_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    segmentation_loss = DiceCELoss(sigmoid=True, squared_pred=True, to_onehot_y=False)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    settings = study["study"].get("method", {}).get("pamc", {})
    accumulation = int(profile["effective_batch_size"])
    args.output.mkdir(parents=True, exist_ok=True)
    run = {
        "schema_version": 1,
        "study_id": study["study"]["study_id"],
        "evaluation_status": study["evaluation_status"],
        "study": str(args.study.resolve()),
        "study_sha256": file_sha256(args.study),
        "profile": profile,
        "profile_id": profile["profile_id"],
        "profile_sha256": file_sha256(args.profile),
        "arm": args.arm,
        "seed": args.seed,
        "epochs": args.epochs,
        "init_filters": args.init_filters,
        "training_config": training,
        "trainer_sha256": file_sha256(Path(__file__)),
        "pamc_sha256": file_sha256(Path(__file__).with_name("pamc.py")),
        "evaluation_sha256": file_sha256(Path(__file__).with_name("evaluation.py")),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "hardware": {
            "device_name": device_name,
            "device_count": torch.cuda.device_count(),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    progress_path = args.output / "progress.json"

    best_dice = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        epoch_started = time.monotonic()
        train_iterator = tqdm(
            train_loader, total=len(train_loader), desc=f"train {epoch}/{args.epochs}",
            unit="batch", dynamic_ncols=True, disable=not sys.stderr.isatty(), leave=True,
        )
        for step, batch in enumerate(train_iterator, start=1):
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, domain_logits = model(image, 1.0 if args.arm == "pamc" else 0.0)
                loss = segmentation_loss(logits, label)
                if args.arm == "pamc":
                    domain_loss = torch.nn.functional.cross_entropy(domain_logits, batch["source_index"].to(device))
                    masked, _ = mask_one_modality(image, float(settings["modality_mask_probability"]))
                    masked_logits, _ = model(masked)
                    consistency = modality_consistency_loss(logits, masked_logits)
                    loss = loss + float(settings["domain_loss_weight"]) * domain_loss + float(settings["modality_consistency_weight"]) * consistency
                loss = loss / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item() * accumulation)
            mean_loss = float(np.mean(losses))
            train_iterator.set_postfix(loss=f"{mean_loss:.4f}")
            write_progress(
                progress_path,
                phase="training",
                epoch=epoch,
                epochs=args.epochs,
                batches_complete=step,
                batches_total=len(train_loader),
                train_loss=mean_loss,
                elapsed_seconds=round(time.monotonic() - epoch_started, 1),
            )
        if len(losses) % accumulation:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        report: dict[str, Any] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if epoch % args.validation_interval == 0:
            report["validation"] = evaluate(
                model, val_loader, device, patch_size,
                progress_path=progress_path,
                progress_fields={"phase": "validation", "epoch": epoch, "epochs": args.epochs},
            )
            if report["validation"]["mean_dice"] > best_dice:
                best_dice = report["validation"]["mean_dice"]
                torch.save({"epoch": epoch, "model": model.state_dict(), "report": report}, args.output / "best.pt")
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(report, sort_keys=True) + "\n")
        write_progress(
            progress_path,
            phase="epoch_complete",
            epoch=epoch,
            epochs=args.epochs,
            train_loss=report["train_loss"],
            validation_dice=report.get("validation", {}).get("mean_dice"),
            elapsed_seconds=round(time.monotonic() - epoch_started, 1),
        )

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    final: dict[str, Any] = {"schema_version": 1, "run": run, "checkpoint_epoch": checkpoint["epoch"], "checkpoint_sha256": file_sha256(args.output / "best.pt")}
    if external_loader is None:
        final["external_evaluation"] = "not_run: pilot_internal_only"
    else:
        final.update({
            "external_clean": evaluate(
                model, external_loader, device, patch_size, progress_path=progress_path,
                progress_fields={"phase": "external_clean", "epoch": args.epochs, "epochs": args.epochs},
            ),
            "external_one_modality_masked": evaluate(
                model, external_loader, device, patch_size, corruption_seed=args.seed, progress_path=progress_path,
                progress_fields={"phase": "external_masked", "epoch": args.epochs, "epochs": args.epochs},
            ),
        })
    (args.output / "external.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    write_progress(progress_path, phase="complete", epoch=args.epochs, epochs=args.epochs)


if __name__ == "__main__":
    main()
