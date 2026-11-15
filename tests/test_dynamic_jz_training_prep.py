"""Read-only J/Z baseline gates; these tests never create splits or fit models."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from training.dynamic import train_dynamic as candidate
from training.dynamic import evaluate_dynamic as evaluator


class DynamicJzTrainingPrepTests(unittest.TestCase):
    def test_source_key_requires_matching_jz_path(self):
        self.assertEqual(candidate.source_key({"source_path": candidate.SOURCE_PREFIX + "J/1.avi",
                                               "label": "J"}), "J/1.avi")
        with self.assertRaises(ValueError):
            candidate.source_key({"source_path": candidate.SOURCE_PREFIX + "Z/1.avi",
                                  "label": "J"})

    def test_read_only_data_contract_and_rights_gate(self):
        config = candidate.load_config()
        records, X, y, audit = candidate.load_checked_data(config, materialize=False)
        self.assertEqual(len(records), 640)
        self.assertEqual(audit["final_counts"], {"J": 265, "Z": 375})
        self.assertEqual(audit["excluded"], ["J/103.avi", "J/250.avi", "J/99.avi"])
        self.assertIsNone(X)
        self.assertIsNone(y)
        self.assertFalse(candidate.rights_verified(config))

    def test_check_only_cannot_split_or_fit(self):
        with patch.object(candidate, "make_split", side_effect=AssertionError("split executed")):
            with patch.object(candidate, "train", side_effect=AssertionError("training executed")):
                with contextlib.redirect_stdout(io.StringIO()) as stream:
                    self.assertEqual(candidate.main(["--check-only"]), 0)
        status = json.loads(stream.getvalue())
        self.assertEqual(status["status"], "DATA_PASS_RIGHTS_BLOCKED")
        self.assertFalse(status["training_performed"])

    def test_synthetic_evaluation_reports_confident_errors(self):
        class FakeModel:
            classes_ = np.asarray(["J", "Z"])

            def predict(self, X):
                return np.where(X[:, 0] > 0, "Z", "J")

            def predict_proba(self, X):
                p_z = np.where(X[:, 0] > 0, 0.8, 0.2)
                return np.column_stack((1 - p_z, p_z))

        X = np.zeros((6, 1104))
        X[[1, 3], 0] = 1
        y = np.asarray(["J", "Z", "J", "Z", "J", "Z"])
        records = [{"source_path": f"synthetic/{index}.avi"} for index in range(6)]
        info = {"indices": {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}}
        report = evaluator.evaluate(FakeModel(), records, X, y, info)
        self.assertEqual(report["splits"]["test"]["error_count"], 1)
        self.assertEqual(report["splits"]["test"]["errors"][0]["confidence"], 0.8)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
