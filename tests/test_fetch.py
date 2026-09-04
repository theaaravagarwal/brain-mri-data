from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from brain_mri_data.fetch import fetch_source


class FetchTests(unittest.TestCase):
    def test_huggingface_resume_reuses_an_interrupted_destination(self) -> None:
        source = {"provider": "huggingface", "locator": "owner/dataset", "revision": "abc"}
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            destination = raw / "external"
            destination.mkdir()
            (destination / "partial.nii.gz").write_bytes(b"partial")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                fetch_source("external", source, raw, False)

            download = Mock(return_value=str(destination))
            module = SimpleNamespace(snapshot_download=download)
            with patch.dict(sys.modules, {"huggingface_hub": module}):
                self.assertEqual(fetch_source("external", source, raw, False, resume=True), destination)
            download.assert_called_once_with(
                repo_id="owner/dataset",
                repo_type="dataset",
                revision="abc",
                allow_patterns=None,
                local_dir=destination,
            )


if __name__ == "__main__":
    unittest.main()
