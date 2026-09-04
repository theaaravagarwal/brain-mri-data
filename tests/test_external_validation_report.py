import json
import unittest

from scripts.generate_external_validation_report import analyze_failures


class ExternalValidationReportTests(unittest.TestCase):
    def test_failure_analysis_is_aggregate_only_and_retraining_safe(self) -> None:
        summary = {"benchmark_id": "fixed-test", "case_count": 3}
        private = {"cases": [
            {"case_token": "case_001", "predicted_voxels": 0, "reference_voxels": 100, "metrics": {"whole_lesion_dice": 0.0, "hd95_mm": None}},
            {"case_token": "case_002", "predicted_voxels": 200, "reference_voxels": 100, "metrics": {"whole_lesion_dice": 0.6, "hd95_mm": 12.0}},
            {"case_token": "case_003", "predicted_voxels": 100, "reference_voxels": 100, "metrics": {"whole_lesion_dice": 0.9, "hd95_mm": 2.0}},
        ]}
        result = analyze_failures(summary, private)
        self.assertEqual(result["weak_case_count"], 2)
        self.assertEqual(result["clusters"]["empty_prediction"], 1)
        self.assertEqual(result["clusters"]["substantial_oversegmentation"], 1)
        self.assertFalse(result["retraining_decision"]["immediate_retraining"])
        self.assertNotIn("case_001", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
