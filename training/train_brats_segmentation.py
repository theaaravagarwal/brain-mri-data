#!/usr/bin/env python3
"""Memory-conscious MONAI SegResNet baseline for validated BraTS manifests."""

from __future__ import annotations

import argparse
import json
import os
import random
import hashlib
import platform
import subprocess
from pathlib import Path

import yaml

import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    SpatialPadd,
    MapTransform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist", type=Path, default=Path("data/manifests/brats2020_kaggle.monai.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/brats-segresnet"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--init-filters", type=int, default=32, help="SegResNet base width; higher values use more VRAM")
    parser.add_argument("--num-workers", default="auto", help="auto, or an explicit non-negative integer")
    parser.add_argument("--max-workers", type=int, default=6, help="safety ceiling used by --num-workers auto")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--limit-train-batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--limit-val-batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--test-datalist", type=Path, help="internal held-out MONAI datalist; defaults to test split in --datalist")
    parser.add_argument("--external-datalist", type=Path, help="locked external MONAI datalist; evaluated once after training")
    parser.add_argument("--external-label-schema", default="brats_africa_123")
    parser.add_argument("--profile", type=Path, help="hardware runtime profile under training/profiles")
    return apply_profile(parser.parse_args())


def apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    if not args.profile:
        args.accelerator = "amd"
        args.roi_size = (160, 160, 160)
        args.pin_memory = False
        args.prefetch_factor = 1
        return args
    with args.profile.open() as stream:
        profile = yaml.safe_load(stream)
    if profile.get("schema_version") != 1:
        raise ValueError(f"Unsupported runtime profile: {args.profile}")
    required = {"profile_id", "accelerator", "batch_size", "effective_batch_size", "num_workers", "prefetch_factor", "pin_memory", "patch_size"}
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"Profile is missing: {', '.join(missing)}")
    if args.batch_size != profile["batch_size"]:
        raise ValueError("Runtime profile and --batch-size must agree")
    if profile["effective_batch_size"] % args.batch_size:
        raise ValueError("effective_batch_size must be divisible by batch_size")
    args.accumulation_steps = profile["effective_batch_size"] // args.batch_size
    args.num_workers = str(profile["num_workers"])
    args.max_workers = int(profile["num_workers"])
    args.roi_size = tuple(int(value) for value in profile["patch_size"])
    args.pin_memory = bool(profile["pin_memory"])
    args.prefetch_factor = int(profile["prefetch_factor"])
    args.accelerator = str(profile["accelerator"])
    args.runtime_profile = profile
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CanonicalizeBraTSLabeld(MapTransform):
    """Map source-specific labels into the legacy {0, 1, 2, 4} convention."""

    def __init__(self, keys: str | tuple[str, ...], schema: str) -> None:
        super().__init__(keys)
        self.schema = schema

    def __call__(self, data):
        result = dict(data)
        for key in self.key_iterator(result):
            label = result[key]
            values = set(torch.as_tensor(label).unique().detach().cpu().tolist())
            if self.schema == "brats_legacy_124":
                allowed = {0, 1, 2, 4}
            elif self.schema == "brats_africa_123":
                allowed = {0, 1, 2, 3}
                label = torch.where(torch.as_tensor(label) == 3, 4, torch.as_tensor(label))
            else:
                raise ValueError(f"Unsupported legacy baseline label schema: {self.schema}")
            if not values <= allowed:
                raise ValueError(f"Unexpected labels for {self.schema}: {sorted(values - allowed)}")
            result[key] = label
        return result


def explicit_workers(value: str) -> int | None:
    if value == "auto":
        return None
    workers = int(value)
    if workers < 0:
        raise ValueError("--num-workers must be non-negative")
    return workers


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def rss_bytes(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def calibrated_workers(dataset: Dataset, cpu_count: int, max_workers: int) -> int:
    """Measure one real loader worker, then size the stable worker pool."""
    probe = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=False,
                       persistent_workers=True, prefetch_factor=1)
    iterator = iter(probe)
    try:
        next(iterator)  # load and transform a real 4-sequence case
        worker_rss = max((rss_bytes(worker.pid) for worker in iterator._workers), default=0)
        available = mem_available_bytes()
    finally:
        iterator._shutdown_workers()
    reserve = 4 * 2**30
    # Full-volume NIfTI loading can temporarily exceed steady-state RSS in WSL.
    per_worker = max(int(worker_rss * 1.5), 512 * 2**20)
    estimated = max(1, (available - reserve) // per_worker)
    return min(int(estimated), max_workers, max(cpu_count - 2, 1))


def resolve_workers(value: str, dataset: Dataset, max_workers: int) -> int:
    workers = explicit_workers(value)
    return workers if workers is not None else calibrated_workers(dataset, os.cpu_count() or 1, max_workers)


def transforms(training: bool, roi_size: tuple[int, int, int], label_schema: str = "brats_legacy_124") -> Compose:
    items = [
        LoadImaged(keys=("image", "label")),
        EnsureChannelFirstd(keys=("image", "label")),
        CanonicalizeBraTSLabeld(keys="label", schema=label_schema),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if training:
        items.extend(
            [
                SpatialPadd(keys=("image", "label"), spatial_size=roi_size),
                RandSpatialCropd(keys=("image", "label"), roi_size=roi_size, random_size=False),
                RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
                RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=1),
                RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=2),
                RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ]
        )
    items.extend(
        [
            EnsureTyped(keys="image", dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.float32),
        ]
    )
    return Compose(items)


def limited(loader: DataLoader, limit: int):
    for index, batch in enumerate(loader):
        if limit and index >= limit:
            break
        yield batch


def evaluate(model, loader: DataLoader, device: torch.device, limit: int, roi_size: tuple[int, int, int]) -> dict[str, float | list[float]]:
    metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    model.eval()
    with torch.inference_mode():
        for batch in limited(loader, limit):
            image, label = batch["image"].to(device), batch["label"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                prediction = sliding_window_inference(image, roi_size, 1, model, overlap=0.5)
            metric(y_pred=[(x > 0.5) for x in decollate_batch(torch.sigmoid(prediction))], y=decollate_batch(label))
    scores = metric.aggregate().detach().cpu().tolist()
    return {"dice_tc_wt_et": scores, "mean_dice": float(np.mean(scores))}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA-compatible PyTorch GPU is available; verify the selected worker environment.")
    if args.accelerator == "amd" and torch.version.hip is None:
        raise SystemExit("The AMD profile requires a ROCm/HIP PyTorch build.")
    if args.accelerator == "cuda" and torch.version.hip is not None:
        raise SystemExit("The CUDA profile requires an NVIDIA CUDA PyTorch build.")
    if args.accelerator not in {"amd", "cuda", "rocm"}:
        raise SystemExit(f"Unsupported accelerator in runtime profile: {args.accelerator}")
    if args.batch_size != 1:
        raise SystemExit("This baseline is deliberately fixed to batch size 1 for 16 GB host RAM.")
    set_seed(args.seed)
    # WSL can exhaust the default file-descriptor based worker handoff with
    # multiple NIfTI loader workers; file-system sharing is stable here.
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    records = json.loads(args.datalist.read_text())
    train_dataset = Dataset(records["training"], transforms(True, args.roi_size))
    workers = resolve_workers(args.num_workers, train_dataset, args.max_workers)
    print(json.dumps({"data_workers": workers, "mem_available_gib": round(mem_available_bytes() / 2**30, 2)}), flush=True)
    loader_options = {"num_workers": workers, "pin_memory": args.pin_memory,
                      "persistent_workers": workers > 0}
    if workers:
        loader_options["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              **loader_options)
    val_loader = DataLoader(Dataset(records["validation"], transforms(False, args.roi_size)), batch_size=1, shuffle=False,
                            **loader_options)
    model = SegResNet(spatial_dims=3, init_filters=args.init_filters, in_channels=4, out_channels=3, dropout_prob=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, to_onehot_y=False)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    args.output.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "init_filters": args.init_filters,
        "requested_workers": args.num_workers,
        "resolved_workers": workers,
        "datalist": str(args.datalist.resolve()),
        "datalist_sha256": file_sha256(args.datalist),
        "test_datalist": str(args.test_datalist.resolve()) if args.test_datalist else "embedded_test_split",
        "external_datalist": str(args.external_datalist.resolve()) if args.external_datalist else None,
        "external_label_schema": args.external_label_schema if args.external_datalist else None,
        "code_revision": code_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "accelerator": args.accelerator,
        "roi_size": args.roi_size,
        "runtime_profile": getattr(args, "runtime_profile", None),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n")
    best_dice = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0
        for step, batch in enumerate(limited(train_loader, args.limit_train_batches), start=1):
            image, label = batch["image"].to(device), batch["label"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(image), label) / args.accumulation_steps
            scaler.scale(loss).backward()
            if step % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * args.accumulation_steps
            batches = step
        if batches % args.accumulation_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        report = {"epoch": epoch, "train_loss": total_loss / max(batches, 1), "data_workers": workers}
        if epoch % args.val_interval == 0:
            report.update(evaluate(model, val_loader, device, args.limit_val_batches, args.roi_size))
            if report["mean_dice"] > best_dice:
                best_dice = report["mean_dice"]
                torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "report": report}, args.output / "best.pt")
        report["gpu_max_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        report["gpu_max_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
        print(json.dumps(report), flush=True)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")

    checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    internal_records = json.loads(args.test_datalist.read_text()) if args.test_datalist else records
    internal_items = internal_records.get("testing") or internal_records.get("test")
    if not internal_items:
        raise SystemExit("Internal test split is missing from the test datalist.")
    internal_loader = DataLoader(Dataset(internal_items, transforms(False, args.roi_size)), batch_size=1, shuffle=False, **loader_options)
    final_report = {"checkpoint_epoch": checkpoint["epoch"], "internal_test": evaluate(model, internal_loader, device, args.limit_val_batches, args.roi_size)}
    if args.external_datalist:
        external_records = json.loads(args.external_datalist.read_text())
        external_items = external_records.get("testing")
        if not external_items:
            raise SystemExit("External datalist must contain a testing split.")
        external_loader = DataLoader(Dataset(external_items, transforms(False, args.roi_size, args.external_label_schema)), batch_size=1, shuffle=False, **loader_options)
        final_report["locked_external_test"] = evaluate(model, external_loader, device, args.limit_val_batches, args.roi_size)
    (args.output / "final_evaluation.json").write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"final_evaluation": final_report}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
