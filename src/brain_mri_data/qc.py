from __future__ import annotations

from pathlib import Path
from typing import Any

from .indexer import read_jsonl, resolve_case_path


def _box(mask: Any, affine: Any) -> dict[str, list[float] | list[int]] | None:
    import numpy as np

    points = np.argwhere(mask > 0)
    if not len(points):
        return None
    lower, upper = points.min(axis=0), points.max(axis=0)
    corners = np.array([lower, upper])
    world = (affine @ np.c_[corners, np.ones(2)].T).T[:, :3]
    return {
        "voxel_min": lower.astype(int).tolist(),
        "voxel_max_inclusive": upper.astype(int).tolist(),
        "world_min": world.min(axis=0).round(4).tolist(),
        "world_max": world.max(axis=0).round(4).tolist(),
    }


def _label_mapping(source: dict[str, Any] | None) -> tuple[set[int] | None, set[int] | None]:
    """Return allowed and positive labels only after a source mapping is approved."""
    mapping = (source or {}).get("label_mapping", {})
    whole_lesion = mapping.get("whole_lesion", {})
    if mapping.get("status") != "approved":
        return None, None
    positive = {int(value) for value in whole_lesion.get("positive_values", [])}
    if not positive:
        return None, None
    return {0, *positive}, positive


def validate_cases(cases_path: Path, raw_root: Path, source: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as error:
        raise RuntimeError("QC needs the optional dependencies: pip install -e '.[qc]'") from error

    allowed_labels, positive_labels = _label_mapping(source)
    results = []
    for case in read_jsonl(cases_path):
        try:
            image_paths = [resolve_case_path(case, path, raw_root) for path in case["modalities"].values()]
            segmentation_path = resolve_case_path(case, case["segmentation"], raw_root)
            images = [nib.load(path) for path in image_paths]
            seg = nib.load(segmentation_path)
            # Do not populate nibabel's float cache: a full 240×240×155 BraTS
            # mask is large, and QC must stay bounded on a 32-GB WSL worker.
            # Segmentation labels are kept in their on-disk integer dtype.
            mask = np.asanyarray(seg.dataobj)
        except Exception as error:
            results.append({"case_id": case["case_id"], "valid": False, "reasons": [f"read_error:{error}"]})
            continue

        reasons: list[str] = []
        if any(image.ndim != 3 for image in images) or seg.ndim != 3:
            reasons.append("volume_not_3d")
        shapes_match = all(image.shape == seg.shape for image in images)
        if not shapes_match:
            reasons.append("shape_mismatch")
        affines_match = all(np.allclose(image.affine, seg.affine, rtol=1e-5, atol=1e-5) for image in images)
        if not affines_match:
            reasons.append("affine_mismatch")
        spacing = list(seg.header.get_zooms()[:3])
        if len(spacing) != 3 or not all(np.isfinite(spacing)) or not all(value > 0 for value in spacing):
            reasons.append("invalid_spacing")
        finite_mask = np.isfinite(mask)
        values = sorted({int(value) for value in np.unique(mask[finite_mask])})
        is_integer_dtype = np.issubdtype(mask.dtype, np.integer)
        if not finite_mask.all() or (not is_integer_dtype and not np.allclose(mask, np.rint(mask))):
            reasons.append("non_integer_mask_values")
        if allowed_labels is None or positive_labels is None:
            reasons.append("label_mapping_unapproved")
        elif not set(values).issubset(allowed_labels):
            reasons.append("unexpected_mask_values")
        try:
            box = _box(mask if positive_labels is None else np.isin(mask, list(positive_labels)), seg.affine)
        except Exception as error:
            box = None
            reasons.append(f"mask_read:{error}")
        if box is None:
            reasons.append("empty_whole_lesion_mask")
        valid = not reasons
        results.append({
            "case_id": case["case_id"], "valid": valid,
            "reasons": ["ok"] if valid else reasons,
            "shape": list(seg.shape), "spacing": spacing, "mask_values": values, "box": box,
        })
        for image in [*images, seg]:
            image.uncache()
    return results
