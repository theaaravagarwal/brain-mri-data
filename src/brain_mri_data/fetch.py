from __future__ import annotations

from pathlib import Path
from typing import Any


def fetch_source(source_id: str, source: dict[str, Any], raw_root: Path, dry_run: bool) -> Path:
    destination = raw_root / source_id
    if dry_run:
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing source: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        if source["provider"] == "kaggle":
            import kagglehub

            kagglehub.dataset_download(source["locator"], output_dir=str(destination))
        elif source["provider"] == "huggingface":
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=source["locator"],
                repo_type="dataset",
                revision=source.get("revision"),
                allow_patterns=source.get("include_patterns"),
                local_dir=destination,
            )
        else:
            raise ValueError(f"Unsupported provider: {source['provider']}")
    except Exception:
        destination.rmdir()
        raise
    return destination
