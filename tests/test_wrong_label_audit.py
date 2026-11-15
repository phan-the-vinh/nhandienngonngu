import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai"))
import audit_wrong_labels as audit


class WrongLabelAuditTests(unittest.TestCase):
    def test_nearest_alternative_preserves_distance_order_not_mode(self):
        nearest, modal, status = audit.alternative_neighbor_evidence(
            ["T", "A", "S", "S", "G", "G", "G"], "A"
        )
        self.assertEqual((nearest, modal, status), ("T", "G", "FOUND_IN_TOP_K"))

    def test_no_alternative_is_explicitly_limited_to_top_k(self):
        self.assertEqual(
            audit.alternative_neighbor_evidence(["A"] * 7, "A"),
            ("", "", "NOT_OBSERVED_IN_TOP_K"),
        )

    def test_reporting_repair_preserves_scores_and_input(self):
        original = pd.DataFrame({
            "knn_labels": ["T|A|S|S|G|G|G"],
            "knn_nearest_alternative_class": ["G"],
            "predicted_probability": ["0.99000000000000001"],
            "label_consistency_status": ["LABEL_CONSISTENT"],
        })
        repaired = audit.repair_knn_alternative_columns(original, np.array(["A"]))
        self.assertEqual(repaired.loc[0, "knn_nearest_alternative_class"], "T")
        self.assertEqual(repaired.loc[0, "knn_modal_alternative_class"], "G")
        self.assertEqual(original.loc[0, "knn_nearest_alternative_class"], "G")
        for column in ["predicted_probability", "label_consistency_status", "knn_labels"]:
            self.assertTrue(original[column].equals(repaired[column]))

    def test_reporting_repair_rejects_incomplete_neighbors(self):
        with self.assertRaises(ValueError):
            audit.repair_knn_alternative_columns(
                pd.DataFrame({"knn_labels": ["A|T"]}), np.array(["A"])
            )

    def test_entropy_is_zero_for_certain_prediction(self):
        values = audit._entropy(np.array([[1.0, 0.0], [0.5, 0.5]]))
        self.assertAlmostEqual(values[0], 0.0, places=10)
        self.assertAlmostEqual(values[1], np.log(2.0), places=10)

    def test_group_split_has_no_overlap_and_one_validation_hit(self):
        rows = []
        for label in ["A", "B"]:
            for group_number in range(10):
                for member in range(2):
                    rows.append({
                        "label": label,
                        "group_id": f"{label}-{group_number}",
                        "file": f"{label}-{group_number}-{member}.jpg",
                    })
        frame = pd.DataFrame(rows)
        splits, report = audit._build_splits(frame)
        hits = np.zeros(len(frame), dtype=int)
        groups = frame["group_id"].to_numpy()
        for train, validation in splits:
            self.assertFalse(set(groups[train]).intersection(groups[validation]))
            hits[validation] += 1
        self.assertTrue(np.all(hits == 1))
        self.assertTrue((report["group_intersection_count"] == 0).all())

    def test_distance_percentile_uses_reference_distribution(self):
        reference = {
            "A": {
                "centroid": np.array([0.0, 0.0]),
                "distribution": np.array([0.0, 1.0, 2.0, 3.0]),
            }
        }
        distance, percentile = audit._distance_and_percentile(
            np.array([[1.5, 0.0], [4.0, 0.0]]), np.array(["A", "A"]), reference
        )
        np.testing.assert_allclose(distance, [1.5, 4.0])
        np.testing.assert_allclose(percentile, [0.5, 1.0])

    def test_diverse_sample_is_deterministic_and_capped(self):
        frame = pd.DataFrame({
            "label": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "file": [f"f{i}" for i in range(30)],
        })
        first = audit._diverse_sample(frame, 12, 42)
        second = audit._diverse_sample(frame, 12, 42)
        self.assertEqual(first["file"].tolist(), second["file"].tolist())
        self.assertEqual(len(first), 12)
        self.assertEqual(set(first["label"]), {"A", "B", "C"})


if __name__ == "__main__":
    unittest.main()
