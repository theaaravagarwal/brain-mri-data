from __future__ import annotations

from pathlib import Path
from typing import Any

AUTOMATIC_PROVIDERS = {"kaggle", "huggingface"}


def fetch_source(source_id: str, source: dict[str, Any], raw_root: Path, dry_run: bool, *, resume: bool = False) -> Path:
    destination = raw_root / source_id
    if dry_run:
        return destination
    if destination.exists() and any(destination.iterdir()):
        if not (resume and source["provider"] == "huggingface"):
            raise FileExistsError(f"Refusing to overwrite existing source: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=True)
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
        if destination.exists() and not any(destination.iterdir()):
            destination.rmdir()
        raise
    return destination


def automatic_source_ids(catalog: dict[str, Any], requested: list[str] | None = None) -> list[str]:
    sources = catalog["sources"]
    if requested:
        unknown = sorted(set(requested) - set(sources))
        if unknown:
            raise KeyError("Unknown source IDs: " + ", ".join(unknown))
        manual = sorted(item for item in requested if sources[item]["provider"] not in AUTOMATIC_PROVIDERS)
        if manual:
            raise ValueError("These sources require manual acquisition: " + ", ".join(manual))
        return requested
    return sorted(item for item, source in sources.items() if source["provider"] in AUTOMATIC_PROVIDERS)


def fetch_automatic(
    catalog: dict[str, Any], raw_root: Path, dry_run: bool, requested: list[str] | None = None
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for source_id in automatic_source_ids(catalog, requested):
        source = catalog["sources"][source_id]
        destination = raw_root / source_id
        if destination.exists() and any(destination.iterdir()):
            results.append({"source_id": source_id, "status": "skipped_existing", "path": str(destination)})
            continue
        fetch_source(source_id, source, raw_root, dry_run)
        results.append({"source_id": source_id, "status": "planned" if dry_run else "downloaded", "path": str(destination)})
    return results
