"""Read-only checks for versioned cleanup archive references."""
import csv
import json
import sys
import unittest
from pathlib import Path


AI_DIR = Path(__file__).resolve().parents[1] / "ai"
ROOT = AI_DIR.parent
sys.path.insert(0, str(AI_DIR))

from cleanup_snapshot_contract import resolve_snapshot, verify_historical_visual_indices


class CleanupSnapshotContractTests(unittest.TestCase):
    def test_visual_reference_indices_are_restorable(self):
        counts = verify_historical_visual_indices()
        self.assertEqual(counts["D23_unique_previews"], 40)
        self.assertEqual(counts["D24_combined_manifest.csv_rows"], 34)
        self.assertEqual(counts["D24_random30_manifest.csv_rows"], 30)
        self.assertEqual(counts["D24_bad_stability4_manifest.csv_rows"], 4)
        self.assertEqual(counts["D24_unique_overlays"], 34)
        self.assertEqual(counts["D25_capture_rows"], 11)
        self.assertEqual(counts["D25_unique_overlays"], 11)

    def test_retired_validation_prediction_is_in_snapshot(self):
        path = resolve_snapshot(
            "AI/dataset/mlp_v2_validation_comparison_v2/"
            "v2_debug_5pct_validation_predictions.csv"
        )
        self.assertTrue(path.is_file())
        with (
            ROOT / "ai/dataset/deep_cleanup_audit_v1/final_quarantine_manifest_v2.csv"
        ).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        row = next(item for item in rows if item["path"].endswith(
            "mlp_v2_validation_comparison_v2/v2_debug_5pct_validation_predictions.csv"
        ))
        self.assertEqual(set(row["prerequisite_reference_ids"].split(";")), {"D21", "R02"})

    def test_all_reference_groups_have_a_disposition(self):
        plan = json.loads((ROOT / "ai/dataset/deep_cleanup_audit_v1/final_cleanup_plan_v2.json").read_text())
        contract = json.loads((ROOT / "ai/dataset/deep_cleanup_audit_v1/reference_retirement_contract_v1.json").read_text())
        self.assertEqual(
            {item["id"] for item in plan["reference_actions"]},
            set(contract["reference_dispositions"]),
        )


if __name__ == "__main__":
    unittest.main()
