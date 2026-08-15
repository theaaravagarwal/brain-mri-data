from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from brain_mri_data.run_matrix import claim_run, expand_matrix


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "config" / "run-matrix" / "glioma.yaml"


class RunMatrixTests(unittest.TestCase):
    def test_matrix_expands_each_profile_arm_and_seed(self) -> None:
        jobs = expand_matrix(MATRIX)
        self.assertEqual(len(jobs), 9)
        self.assertEqual(len({job["run_id"] for job in jobs}), 9)

    def test_claim_is_profile_bound_and_immutable(self) -> None:
        job = expand_matrix(MATRIX)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            claim = claim_run(MATRIX, job["run_id"], job["profile"], root)
            self.assertTrue(Path(claim["claim"]).exists())
            with self.assertRaises(FileExistsError):
                claim_run(MATRIX, job["run_id"], job["profile"], root)
            with self.assertRaises(ValueError):
                claim_run(MATRIX, job["run_id"], "wrong-profile", root)
