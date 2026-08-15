from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from brain_mri_data.indexer import resolve_case_path, verify_case_files


class PortableManifestTests(unittest.TestCase):
    def test_resolves_only_under_declared_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            target = raw_root / "source_a" / "nested" / "t1.nii.gz"
            target.parent.mkdir(parents=True)
            target.touch()
            record = {"source_id": "source_a"}
            self.assertEqual(resolve_case_path(record, "nested/t1.nii.gz", raw_root), target.resolve())

    def test_rejects_absolute_and_parent_paths(self) -> None:
        record = {"source_id": "source_a"}
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary)
            with self.assertRaises(ValueError):
                resolve_case_path(record, "/etc/passwd", raw_root)
            with self.assertRaises(ValueError):
                resolve_case_path(record, "../other/scan.nii.gz", raw_root)

    def test_verifies_manifest_fingerprints_after_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_root = Path(temporary) / "raw"
            file = raw_root / "source_a" / "scan_t1.nii.gz"
            file.parent.mkdir(parents=True)
            file.write_bytes(b"original")
            record = {
                "case_id": "a", "source_id": "source_a", "modalities": {"t1": "scan_t1.nii.gz"},
                "provenance": {"files": {"t1": {"bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}}},
            }
            manifest = Path(temporary) / "cases.jsonl"
            manifest.write_text(json.dumps(record) + "\n")
            self.assertEqual(verify_case_files(manifest, raw_root)["failed_cases"], 0)
            file.write_bytes(b"altered")
            self.assertEqual(verify_case_files(manifest, raw_root)["failed_cases"], 1)
