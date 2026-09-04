#!/usr/bin/env python3
"""Validate and run the fixed RTX 4060 adult-glioma research segmenter.

The runner accepts one four-volume NIfTI study, verifies the frozen checkpoint,
and emits a binary research segmentation plus immutable provenance artifacts.
It never sends image data or paths to the optional language model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from brain_mri_data.language_contracts import DISCLAIMER, ResearchSegmentationResultV1
from brain_mri_data.language_gateway import generate_result_explanation


MODALITIES = ("t1", "t1ce", "t2", "flair")
MODALITY_SUFFIXES = ("0000", "0001", "0002", "0003")
MODEL_ID = "glioma-segresnet-20260828"
EXPECTED_CHECKPOINT_SHA256 = "121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5"
TRAINING_GIT_REVISION = "570c65ac4709dac3b05f48314ddd5aef70589a7d"
STUDY_SHA256 = "e53f85b429449585089133b1d9f680c3d80125b58da3042e5510522e2b333f6d"
PROFILE_SHA256 = "9ec821920b6a08e914306d1651101dd52693d02c185f2750410297ec1c43fc7e"
TRAINER_SHA256 = "bf5dede3b5b1ee5d916cd6f046ca7eda8ea579f0f730db6f9201e2523b0456d9"
MAX_AXIS = 512
MAX_VOXELS = 64_000_000
REFERENCE_NAMES = ("research_reference.nii.gz", "research_reference.nii")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="directory containing one *_0000…*_0003 NIfTI set")
    parser.add_argument("output", type=Path, nargs="?", help="new output directory")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--job-id", default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/glioma-pilot--cuda-4060--brats--20260828--e100/best.pt"),
    )
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(80, 80, 80))
    parser.add_argument("--ollama-host", default=os.environ.get("BRAIN_MRI_OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--ollama-model", default=os.environ.get("BRAIN_MRI_LLM_MODEL"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def write_json_once(path: Path, value: Any) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def input_paths(directory: Path) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Input must be a regular directory")
    paths: list[Path] = []
    for suffix in MODALITY_SUFFIXES:
        matches = sorted([*directory.glob(f"*_{suffix}.nii.gz"), *directory.glob(f"*_{suffix}.nii")])
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one volume for modality suffix {suffix}; found {len(matches)}")
        if matches[0].is_symlink() or matches[0].parent.resolve() != directory.resolve():
            raise ValueError("Input volumes must be regular, non-symlink files")
        paths.append(matches[0])
    reference = reference_path(directory)
    allowed = {path.resolve() for path in paths}
    if reference is not None:
        allowed.add(reference.resolve())
    supplied = {
        path.resolve()
        for path in [*directory.glob("*.nii"), *directory.glob("*.nii.gz")]
        if path.is_file()
    }
    if supplied != allowed:
        raise ValueError("Input directory must contain four scan volumes and, optionally, one reference mask")
    stems = {
        path.name.removesuffix(f"_{suffix}.nii.gz").removesuffix(f"_{suffix}.nii")
        for path, suffix in zip(paths, MODALITY_SUFFIXES, strict=True)
    }
    if len(stems) != 1:
        raise ValueError("All four volumes must share one research input stem")
    return paths


def reference_path(directory: Path) -> Path | None:
    matches = [directory / name for name in REFERENCE_NAMES if (directory / name).is_file()]
    if len(matches) > 1:
        raise ValueError("Expected at most one reference mask")
    if matches and (matches[0].is_symlink() or matches[0].parent.resolve() != directory.resolve()):
        raise ValueError("Reference mask must be a regular, non-symlink file")
    return matches[0] if matches else None


def validate_reference(path: Path, image: nib.Nifti1Image) -> tuple[np.ndarray, dict[str, Any]]:
    reference = nib.load(path)
    if tuple(reference.shape) != tuple(image.shape) or not np.allclose(reference.affine, image.affine, rtol=0, atol=1e-5):
        raise ValueError("Reference mask geometry does not match the scan")
    values = np.asarray(reference.dataobj)
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError("Reference mask must contain finite integer labels")
    labels = sorted(int(value) for value in np.unique(values))
    if not set(labels).issubset({0, 1, 2, 3, 4}):
        raise ValueError("Reference mask contains unsupported labels")
    binary = values != 0
    nonzero = int(np.count_nonzero(binary))
    if nonzero == 0:
        raise ValueError("Reference mask must contain a non-empty lesion outline")
    return binary, {
        "status": "pass",
        "geometry_match": True,
        "sha256": sha256(path),
        "labels": labels,
        "nonzero_voxels": nonzero,
    }


def evaluation_metrics(prediction: np.ndarray, truth: np.ndarray, spacing: tuple[float, ...]) -> dict[str, Any]:
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    predicted = prediction.astype(bool, copy=False)
    expected = truth.astype(bool, copy=False)
    true_positive = int(np.count_nonzero(predicted & expected))
    false_positive = int(np.count_nonzero(predicted & ~expected))
    false_negative = int(np.count_nonzero(~predicted & expected))
    dice_denominator = 2 * true_positive + false_positive + false_negative
    union = true_positive + false_positive + false_negative
    predicted_count = true_positive + false_positive
    truth_count = true_positive + false_negative
    hd95: float | None = None
    if predicted_count and truth_count:
        structure = generate_binary_structure(3, 1)
        predicted_surface = predicted ^ binary_erosion(predicted, structure=structure, border_value=0)
        expected_surface = expected ^ binary_erosion(expected, structure=structure, border_value=0)
        to_expected = distance_transform_edt(~expected_surface, sampling=spacing)[predicted_surface]
        to_predicted = distance_transform_edt(~predicted_surface, sampling=spacing)[expected_surface]
        hd95 = float(np.percentile(np.concatenate((to_expected, to_predicted)), 95))
    return {
        "status": "complete",
        "scope": "single_user_supplied_reference",
        "whole_lesion_dice": float(2 * true_positive / dice_denominator),
        "whole_lesion_iou": float(true_positive / union),
        "precision": float(true_positive / predicted_count) if predicted_count else 0.0,
        "recall": float(true_positive / truth_count),
        "hd95_mm": hd95,
        "true_positive_voxels": true_positive,
        "false_positive_voxels": false_positive,
        "false_negative_voxels": false_negative,
    }


def validate_and_normalize(paths: list[Path]) -> tuple[np.ndarray, nib.Nifti1Image, dict[str, Any]]:
    images = [nib.load(path) for path in paths]
    reference = images[0]
    shape = tuple(int(size) for size in reference.shape)
    if len(shape) != 3 or any(size < 1 or size > MAX_AXIS for size in shape) or int(np.prod(shape)) > MAX_VOXELS:
        raise ValueError("Each volume must be 3D and within the fixed inference bounds")
    affine = np.asarray(reference.affine, dtype=np.float64)
    spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    if not np.isfinite(affine).all() or any(not np.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("Reference geometry contains invalid affine or spacing values")

    normalized: list[np.ndarray] = []
    for modality, path, image in zip(MODALITIES, paths, images, strict=True):
        if tuple(image.shape) != shape or not np.allclose(image.affine, affine, rtol=0, atol=1e-5):
            raise ValueError(f"Geometry mismatch for {modality}")
        volume = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(volume).all():
            raise ValueError(f"Non-finite voxel values in {modality}")
        foreground = volume != 0
        if foreground.any():
            mean, std = float(volume[foreground].mean()), float(volume[foreground].std())
            volume[foreground] = (volume[foreground] - mean) / std if std > 0 else 0
        normalized.append(volume)

    geometry = {
        "shape": list(shape),
        "spacing_mm": [round(value, 8) for value in spacing],
        "affine": [[round(float(value), 8) for value in row] for row in affine],
    }
    validation = {
        "schema_version": "research-study-validation/v1",
        "status": "pass",
        "modality_count": 4,
        "modalities": list(MODALITIES),
        "geometry_match": True,
        "shape": list(shape),
        "spacing_mm": list(spacing),
        "geometry_sha256": canonical_sha256(geometry),
        "modality_sha256": {modality: sha256(path) for modality, path in zip(MODALITIES, paths, strict=True)},
    }
    return np.stack(normalized, axis=0), reference, validation


def validate_study(directory: Path) -> dict[str, Any]:
    _, image, validation = validate_and_normalize(input_paths(directory))
    reference = reference_path(directory)
    if reference is not None:
        _, validation["reference_mask"] = validate_reference(reference, image)
    return validation


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    import monai
    import torch
    from monai.inferers import sliding_window_inference

    from training.pamc import PamcSegResNet

    if args.output is None:
        raise ValueError("Output directory is required unless --validate-only is used")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this fixed RTX 4060 research runner")

    paths = input_paths(args.input)
    image, reference, validation = validate_and_normalize(paths)
    truth = None
    reference_mask = reference_path(args.input)
    if reference_mask is not None:
        truth, validation["reference_mask"] = validate_reference(reference_mask, reference)
    checkpoint = args.checkpoint.resolve(strict=True)
    observed_checkpoint_sha256 = sha256(checkpoint)
    if args.expected_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("The configured checkpoint digest is not the frozen product digest")
    if observed_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen checkpoint digest mismatch")

    job_id = str(UUID(args.job_id)) if args.job_id else str(uuid4())
    if UUID(job_id).version != 4:
        raise ValueError("job-id must be a UUIDv4")
    device = torch.device("cuda:0")
    model = PamcSegResNet(init_filters=32, source_count=1).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        logits = sliding_window_inference(
            tensor,
            tuple(args.patch_size),
            1,
            lambda values: model(values)[0],
            overlap=0.5,
        )
        segmentation = (
            (torch.sigmoid(logits) >= 0.5)
            .to(torch.uint8)
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    if tuple(segmentation.shape) != tuple(reference.shape) or not set(np.unique(segmentation)).issubset({0, 1}):
        raise RuntimeError("Inference produced an invalid segmentation contract")
    args.output.mkdir(parents=True, mode=0o700)
    output = args.output / "research_segmentation.nii.gz"
    output_header = reference.header.copy()
    output_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(segmentation, reference.affine, output_header), output)
    os.chmod(output, 0o600)
    reloaded = nib.load(output)
    if tuple(reloaded.shape) != tuple(reference.shape) or not np.allclose(reloaded.affine, reference.affine, rtol=0, atol=1e-5):
        raise RuntimeError("Saved segmentation did not preserve input geometry")
    if reloaded.get_data_dtype() != np.dtype(np.uint8) or not set(np.unique(np.asarray(reloaded.dataobj))).issubset({0, 1}):
        raise RuntimeError("Saved segmentation did not preserve the binary uint8 label contract")

    result_data = {
        "schema_version": "research-segmentation-result/v1",
        "job_id": job_id,
        "study_id": "glioma",
        "protocol": "glioma_4seq_v1",
        "disclaimer": DISCLAIMER,
        "input_qc": validation,
        "segmentation": {
            "status": "complete",
            "output_sha256": sha256(output),
            "output_shape": list(segmentation.shape),
            "geometry_preserved": True,
            "labels": [0, 1],
            "label_count": 2,
            "nonzero_voxels": int(np.count_nonzero(segmentation)),
        },
        "evaluation": evaluation_metrics(segmentation, truth, tuple(validation["spacing_mm"])) if truth is not None else None,
        "provenance": {
            "model_id": MODEL_ID,
            "model_scope": "internal_research_only",
            "checkpoint_sha256": observed_checkpoint_sha256,
            "training_git_revision": TRAINING_GIT_REVISION,
            "study_sha256": STUDY_SHA256,
            "profile_sha256": PROFILE_SHA256,
            "trainer_sha256": TRAINER_SHA256,
            "inference_script_sha256": sha256(Path(__file__)),
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "monai_version": monai.__version__,
            "nibabel_version": nib.__version__,
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }
    result = ResearchSegmentationResultV1.model_validate(result_data)
    result_payload = result.model_dump(mode="json")
    receipt = {
        "schema_version": "research-inference-receipt/v1",
        "disclaimer": DISCLAIMER,
        "job_id": job_id,
        "input_qc": validation,
        "segmentation": result_payload["segmentation"],
        "evaluation": result_payload["evaluation"],
        "model": result_payload["provenance"],
    }
    explanation = generate_result_explanation(
        result,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
    )
    write_json_once(args.output / "result.json", result_payload)
    write_json_once(args.output / "receipt.json", receipt)
    write_json_once(args.output / "explanation.json", explanation)
    return result_payload


def main() -> None:
    args = arguments()
    if args.validate_only:
        if args.output is not None:
            raise ValueError("--validate-only does not accept an output directory")
        print(json.dumps(validate_study(args.input), sort_keys=True))
        return
    print(json.dumps(run_inference(args), sort_keys=True))


if __name__ == "__main__":
    main()
