import unittest

from scripts.run_serving_language_eval import envelope


class ServingLanguageEvalTests(unittest.TestCase):
    def test_fixture_builds_the_exact_serving_contract(self) -> None:
        result = envelope({
            "id": "fixture", "job_id": "a4ee2cd5-cb1f-4d56-bb68-c794f401ecab",
            "shape": [64, 64, 64], "spacing_mm": [1.0, 1.0, 1.0], "nonzero_voxels": 12,
        })
        self.assertEqual(result.segmentation.nonzero_voxels, 12)
        self.assertEqual(result.provenance.model_id, "glioma-segresnet-20260828")

    def test_fixture_rejects_impossible_nonzero_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid nonzero_voxels"):
            envelope({
                "id": "fixture", "job_id": "a4ee2cd5-cb1f-4d56-bb68-c794f401ecab",
                "shape": [2, 2, 2], "spacing_mm": [1.0, 1.0, 1.0], "nonzero_voxels": 9,
            })


if __name__ == "__main__":
    unittest.main()
