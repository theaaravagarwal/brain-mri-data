from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from brain_mri_data.indexer import resolve_case_path


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
