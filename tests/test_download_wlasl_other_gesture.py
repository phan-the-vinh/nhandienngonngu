"""Offline safety checks for the WLASL candidate downloader."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import download_wlasl_other_gesture as downloader


class WlaslDownloaderTests(unittest.TestCase):
    def test_real_manifest_is_limited_and_review_required(self):
        rows, methods = downloader.load_candidates()
        self.assertEqual(len(rows), 220)
        self.assertEqual(sum(methods.values()), 220)
        self.assertTrue(all(row["review_status"] == "REVIEW_REQUIRED" for row in rows))
        self.assertTrue(all(row["training_eligible"] == "False" for row in rows))

    def test_url_methods(self):
        self.assertEqual(downloader.download_method("https://example.org/clip.mp4"), "direct")
        self.assertEqual(downloader.download_method("https://youtu.be/abc"), "yt_dlp")
        self.assertEqual(downloader.download_method("https://example.org/clip.swf"), "unsupported_swf")
        with self.assertRaises(downloader.ManifestError):
            downloader.download_method("file:///private/video.mp4")

    def test_duplicate_video_id_fails_before_download(self):
        rows, _ = downloader.load_candidates()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "candidates.csv"
            audit = root / "audit.json"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows([rows[0], rows[0], *rows[2:]])
            audit.write_text(json.dumps({"candidate_manifest_sha256": downloader._file_sha256(manifest)}), encoding="utf-8")
            with self.assertRaisesRegex(downloader.ManifestError, "Duplicate video_id"):
                downloader.load_candidates(manifest, audit)

    def test_check_only_does_not_download_or_write(self):
        output = io.StringIO()
        with (patch.object(sys, "argv", ["download_wlasl_other_gesture.py", "--check-only"]),
              patch.object(downloader, "_download_direct", side_effect=AssertionError("network")),
              patch.object(downloader, "_download_with_yt_dlp", side_effect=AssertionError("network")),
              patch.object(downloader, "execute", side_effect=AssertionError("write")),
              contextlib.redirect_stdout(output)):
            self.assertEqual(downloader.main(), 0)
        self.assertEqual(json.loads(output.getvalue())["candidate_count"], 220)

    def test_execute_requires_confirmation(self):
        output = io.StringIO()
        with (patch.object(sys, "argv", ["download_wlasl_other_gesture.py", "--execute"]),
              patch.object(downloader.importlib.util, "find_spec", return_value=object()),
              patch("builtins.input", return_value="n"),
              patch.object(downloader, "execute", side_effect=AssertionError("download")),
              contextlib.redirect_stdout(output)):
            self.assertEqual(downloader.main(), 0)
        self.assertIn("Cancelled", output.getvalue())

    def test_one_failed_url_does_not_stop_later_candidate(self):
        sample, _ = downloader.load_candidates()
        rows = []
        for index, original in enumerate(sample[:2], start=1):
            row = dict(original)
            row["video_id"] = f"TEST{index}"
            row["url"] = f"https://example.org/test{index}.mp4"
            rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            audit = root / "audit.csv"
            calls = 0

            def fake_download(url, video_id, staging):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("unavailable")
                path = staging / f"WLASL_{video_id}.temporary.mp4"
                path.write_bytes(b"\x00\x00\x00\x18ftyp" + b"0" * 2048)
                return path

            with (patch.object(downloader, "PROJECT_ROOT", root),
                  patch.object(downloader, "RAW_DIR", raw),
                  patch.object(downloader, "DOWNLOAD_AUDIT", audit),
                  patch.object(downloader, "_download_direct", side_effect=fake_download),
                  contextlib.redirect_stdout(io.StringIO())):
                counts = downloader.execute(rows)
            self.assertEqual(counts["DOWNLOAD_FAILED"], 1)
            self.assertEqual(counts["DOWNLOADED"], 1)
            with audit.open(newline="", encoding="utf-8") as stream:
                events = list(csv.DictReader(stream))
            self.assertEqual([event["download_status"] for event in events], ["DOWNLOAD_FAILED", "DOWNLOADED"])
            self.assertTrue(all(event["review_status"] == "REVIEW_REQUIRED" for event in events))

    def test_retry_plan_uses_only_original_youtube_format_failures(self):
        rows, _ = downloader.load_candidates()
        eligible, original, counts = downloader.build_retry_plan(rows)
        self.assertEqual(len(original), 220)
        self.assertEqual(len(eligible), 40)
        self.assertEqual(counts["already_downloaded"], 83)
        self.assertEqual(counts["http_403_source_review"], 10)
        self.assertEqual(counts["html_source_review"], 10)
        self.assertEqual(counts["ssl_hostname_source_review"], 15)
        self.assertEqual(counts["private_video"], 5)
        self.assertEqual(counts["video_unavailable"], 28)
        self.assertEqual(counts["http_404"], 14)
        self.assertEqual(counts["unsupported_swf"], 15)
        self.assertTrue(all("youtube" in row["url"] or "youtu.be" in row["url"] for row in eligible))

    def test_check_retry_does_not_download_or_write(self):
        original_sha = downloader._file_sha256(downloader.DOWNLOAD_AUDIT)
        output = io.StringIO()
        with (patch.object(sys, "argv", ["download_wlasl_other_gesture.py", "--check-retry"]),
              patch.object(downloader, "_download_with_yt_dlp", side_effect=AssertionError("network")),
              patch.object(downloader, "execute_retry", side_effect=AssertionError("write")),
              contextlib.redirect_stdout(output)):
            self.assertEqual(downloader.main(), 0)
        self.assertEqual(json.loads(output.getvalue())["retry_eligible_youtube_format"], 40)
        self.assertEqual(downloader._file_sha256(downloader.DOWNLOAD_AUDIT), original_sha)

    def test_retry_requires_confirmation(self):
        output = io.StringIO()
        with (patch.object(sys, "argv", ["download_wlasl_other_gesture.py", "--retry"]),
              patch("builtins.input", return_value="n"),
              patch.object(downloader, "execute_retry", side_effect=AssertionError("download")),
              contextlib.redirect_stdout(output)):
            self.assertEqual(downloader.main(), 0)
        self.assertIn("Cancelled", output.getvalue())

    def test_retry_audit_is_versioned_and_one_failure_does_not_stop_next(self):
        rows, _ = downloader.load_candidates()
        eligible, original, _ = downloader.build_retry_plan(rows)
        selected = eligible[:2]
        original_subset = {row["video_id"]: original[row["video_id"]] for row in selected}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            retry_audit = root / "download_audit_v2.csv"
            calls = 0

            def fake_download(url, video_id, staging, format_selector):
                nonlocal calls
                self.assertEqual(format_selector, "bv*")
                calls += 1
                if calls == 1:
                    raise OSError("unavailable")
                path = staging / f"WLASL_{video_id}.temporary.mp4"
                path.write_bytes(b"\x00\x00\x00\x18ftyp" + b"0" * 2048)
                return path

            with (patch.object(downloader, "PROJECT_ROOT", root),
                  patch.object(downloader, "RAW_DIR", raw),
                  patch.object(downloader, "_download_with_yt_dlp", side_effect=fake_download),
                  contextlib.redirect_stdout(io.StringIO())):
                counts = downloader.execute_retry(selected, original_subset, retry_audit)
            self.assertEqual(counts["RETRY_FAILED"], 1)
            self.assertEqual(counts["RETRY_SUCCESS"], 1)
            with retry_audit.open(newline="", encoding="utf-8") as stream:
                events = list(csv.DictReader(stream))
            self.assertEqual(len(events), 4)  # 2 preserved baseline rows + 2 retry events
            self.assertEqual([event["retry_status"] for event in events[-2:]], ["RETRY_FAILED", "RETRY_SUCCESS"])
            self.assertTrue(all(event["review_status"] == "REVIEW_REQUIRED" for event in events))


if __name__ == "__main__":
    unittest.main()
