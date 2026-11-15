"""Read-only safety checks for the user-run J40 filtered-JSONL utility."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from tools import prepare_dynamic_j_clean_dataset as clean


class PrepareDynamicJCleanDatasetTests(unittest.TestCase):
    def test_preflight_exact_manifest_and_record_match(self):
        plan = clean.preflight()
        self.assertEqual(len(plan["confirmed"]), 40)
        self.assertEqual(plan["original"]["counts"], {"J": 308, "Z": 375})
        self.assertEqual(plan["expected_after"], {"J": 268, "Z": 375})
        self.assertEqual(len(plan["remaining_review"]), 38)
        self.assertEqual(len(plan["missing_sequence"]), 8)
        self.assertFalse(set(plan["remaining_review"]) & set(plan["confirmed"]))
        self.assertEqual({row["batch"] for row in plan["confirmed"].values()},
                         {"INITIAL", "BATCH_01", "BATCH_02", "BATCH_03", "BATCH_04"})

    def test_manifest_rejects_duplicate_and_non_j(self):
        plan = clean.preflight()
        rows = list(plan["confirmed"].values())
        inventory = {row["video"]: {"classification": "EXCLUDE_CONFIRMED"}
                     for row in rows}
        batches = clean._batch_sets()
        raw = {f"J/{path.name}" for path in clean.RAW_J.glob("*.avi")}
        duplicate = [dict(row) for row in rows]
        duplicate[-1] = dict(duplicate[0])
        with self.assertRaises(ValueError):
            clean.validate_manifest_rows(duplicate, batches, raw, inventory)
        wrong_label = [dict(row) for row in rows]
        wrong_label[0]["label"] = "Z"
        with self.assertRaises(ValueError):
            clean.validate_manifest_rows(wrong_label, batches, raw, inventory)

    def test_check_only_does_not_write_filtered_jsonl(self):
        with patch.object(clean, "create_filtered", side_effect=AssertionError("filter executed")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(clean.main(["--check-only"]), 0)
        self.assertFalse(clean.OUTPUT.exists())
        self.assertFalse(clean.OUTPUT_MANIFEST.exists())

    def test_line_policy_only_excludes_manifested_j_and_preserves_z(self):
        j = (b'{"label":"J","source_path":"' + clean.SOURCE_PREFIX.encode()
             + b'J/100.avi"}\n')
        z = (b'{"label":"Z","source_path":"' + clean.SOURCE_PREFIX.encode()
             + b'Z/1.avi"}\n')
        excluded = {clean.SOURCE_PREFIX + "J/100.avi"}
        self.assertEqual(clean.line_action(j, excluded),
                         (False, "J", clean.SOURCE_PREFIX + "J/100.avi"))
        self.assertEqual(clean.line_action(z, excluded),
                         (True, "Z", clean.SOURCE_PREFIX + "Z/1.avi"))
        with self.assertRaises(ValueError):
            clean.line_action(z, {clean.SOURCE_PREFIX + "Z/1.avi"})


if __name__ == "__main__":
    unittest.main()
