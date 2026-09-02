#!/usr/bin/env python3
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest_path = root / "data/manifests/glioma.pilot.4060.json"
marker = root / "data/.4090-transfer-complete"
manifest = json.loads(manifest_path.read_text())
verified = 0
for item in manifest["development"]:
    record = item["record"]
    paths = {**record["modalities"], "seg": record["segmentation"]}
    for kind, relative in paths.items():
        path = root / "data/raw" / item["source_id"] / relative
        expected = record["provenance"]["files"][kind]["sha256"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"transfer incomplete: {item['case_id']}:{kind}")
        verified += 1
receipt = {
    "schemaVersion": "research-transfer-gate/v1",
    "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "cases": len(manifest["development"]),
    "files": verified,
    "verifiedAt": datetime.now(UTC).isoformat(),
}
temporary = marker.with_suffix(".tmp")
temporary.write_text(json.dumps(receipt, sort_keys=True) + "\n")
os.replace(temporary, marker)
print(json.dumps(receipt, sort_keys=True))
