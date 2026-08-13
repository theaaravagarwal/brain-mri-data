from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog_path = path or project_root() / "datasets" / "catalog.yaml"
    with catalog_path.open() as file:
        return yaml.safe_load(file)


def get_source(source_id: str, catalog: dict[str, Any]) -> dict[str, Any]:
    try:
        return catalog["sources"][source_id]
    except KeyError as error:
        available = ", ".join(sorted(catalog["sources"]))
        raise ValueError(f"Unknown source '{source_id}'. Available: {available}") from error
