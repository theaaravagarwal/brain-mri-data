from __future__ import annotations

from pathlib import Path
from typing import Any

from .indexer import read_jsonl


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


def validate_cases(cases_path: Path) -> list[dict[str, Any]]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as error:
        raise RuntimeError("QC needs the optional dependencies: pip install -e '.[qc]'") from error

    results = []
    for case in read_jsonl(cases_path):
        image_paths = list(case["modalities"].values())
        images = [nib.load(path) for path in image_paths]
        seg = nib.load(case["segmentation"])
        shapes_match = all(image.shape == seg.shape for image in images)
        affines_match = all(np.allclose(image.affine, seg.affine, rtol=1e-5, atol=1e-5) for image in images)
        try:
            box = _box(seg.get_fdata(dtype=np.float32), seg.affine)
        except Exception as error:
            box = None
            results.append({"case_id": case["case_id"], "valid": False, "reason": f"mask_read:{error}"})
            continue
        valid = shapes_match and affines_match and box is not None
        results.append({
            "case_id": case["case_id"], "valid": valid,
            "reason": "ok" if valid else "geometry_mismatch_or_empty_mask",
            "shape": list(seg.shape), "box": box,
        })
    return results
