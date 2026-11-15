"""User-operated, manifest-limited WLASL OTHER_GESTURE video downloader.

--check-only is offline and read-only. --execute requires confirmation. Downloaded
videos remain REVIEW_REQUIRED; this tool does not create NONE training samples.
--check-retry is offline and read-only. --retry only tries approved recoverable
format failures from the first download audit, after separate confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "data" / "experiments" / "dynamic_none_v1"
MANIFEST = EXPERIMENT_ROOT / "metadata" / "wlasl_other_gesture_candidates_v1.csv"
MANIFEST_AUDIT = EXPERIMENT_ROOT / "metadata" / "wlasl_metadata_audit_v1.json"
RAW_DIR = EXPERIMENT_ROOT / "raw" / "wlasl_other_gesture"
DOWNLOAD_AUDIT = EXPERIMENT_ROOT / "metadata" / "wlasl_other_gesture_download_audit_v1.csv"
RETRY_AUDIT = EXPERIMENT_ROOT / "metadata" / "wlasl_other_gesture_download_audit_v2.csv"
EXPECTED_COUNT = 220
DIRECT_EXTENSIONS = {".mp4", ".mov"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
REQUIRED_COLUMNS = {
    "video_id", "gloss", "signer_id", "source", "url", "frame_start",
    "frame_end", "split", "review_status", "training_eligible", "none_subtype",
}
AUDIT_COLUMNS = (
    "attempted_at_utc", "video_id", "gloss", "signer_id", "source", "url",
    "frame_start", "frame_end", "split", "local_path", "download_status",
    "failure_reason", "review_status", "file_size_bytes", "sha256",
)
RETRY_COLUMNS = ("retry_attempted", "retry_method", "retry_status", "retry_failure_reason")
RETRY_AUDIT_COLUMNS = (*AUDIT_COLUMNS, *RETRY_COLUMNS)
YOUTUBE_FORMAT_FALLBACK = "bv*"  # best available format containing video; audio is not required
RETRY_METHOD = "YTDLP_BEST_AVAILABLE_VIDEO_NO_AUDIO_REQUIRED"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class ManifestError(ValueError):
    """Candidate manifest is unsafe or inconsistent."""


def download_method(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise ManifestError(f"Invalid public HTTP(S) URL: {url}")
    if parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}:
        raise ManifestError("Local URLs are not allowed")
    extension = Path(parsed.path).suffix.lower()
    if extension == ".swf":
        return "unsupported_swf"
    if extension in DIRECT_EXTENSIONS:
        return "direct"
    return "yt_dlp"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidates(
    manifest_path: Path = MANIFEST, audit_path: Path = MANIFEST_AUDIT,
) -> tuple[list[dict[str, str]], Counter[str]]:
    if not manifest_path.is_file() or not audit_path.is_file():
        raise ManifestError("Candidate manifest or metadata audit report is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_sha = audit.get("candidate_manifest_sha256")
    if _file_sha256(manifest_path) != expected_sha:
        raise ManifestError("Candidate manifest SHA256 differs from metadata audit")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ManifestError("Missing required candidate provenance/review columns")
        rows = list(reader)
    if len(rows) != EXPECTED_COUNT:
        raise ManifestError(f"Expected {EXPECTED_COUNT} candidates, found {len(rows)}")

    seen_ids: set[str] = set()
    seen_stems: set[str] = set()
    seen_urls: set[str] = set()
    methods: Counter[str] = Counter()
    for row in rows:
        video_id = row["video_id"].strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            raise ManifestError(f"Unsafe video_id: {video_id!r}")
        stem = f"WLASL_{video_id}".casefold()
        if video_id in seen_ids or stem in seen_stems:
            raise ManifestError(f"Duplicate video_id/output stem: {video_id}")
        seen_ids.add(video_id)
        seen_stems.add(stem)
        url = row["url"].strip()
        if url in seen_urls:
            raise ManifestError(f"Duplicate source URL: {url}")
        seen_urls.add(url)
        if (row["review_status"] != "REVIEW_REQUIRED"
                or row["none_subtype"] != "OTHER_GESTURE"
                or row["training_eligible"].strip().lower() != "false"):
            raise ManifestError(f"Candidate {video_id} is not pending review")
        if row["gloss"].strip().lower() in {"j", "z"}:
            raise ManifestError(f"J/Z gloss is not an OTHER_GESTURE candidate: {video_id}")
        try:
            start, end = int(row["frame_start"]), int(row["frame_end"])
        except ValueError as exc:
            raise ManifestError(f"Invalid frame range for {video_id}") from exc
        if start < 0 or (end != -1 and end < start):
            raise ManifestError(f"Invalid frame range for {video_id}")
        methods[download_method(url)] += 1
    return rows, methods


def existing_videos(video_id: str) -> list[Path]:
    if not RAW_DIR.exists():
        return []
    stem = f"WLASL_{video_id}"
    return sorted(
        path for path in RAW_DIR.iterdir()
        if path.is_file() and path.stem.casefold() == stem.casefold()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _video_header_ok(path: Path) -> bool:
    if path.stat().st_size < 1024:
        return False
    with path.open("rb") as stream:
        header = stream.read(16)
    if path.suffix.lower() in {".mp4", ".mov"}:
        return header[4:8] == b"ftyp"
    if path.suffix.lower() in {".mkv", ".webm"}:
        return header.startswith(b"\x1a\x45\xdf\xa3")
    if path.suffix.lower() == ".avi":
        return header.startswith(b"RIFF") and header[8:12] == b"AVI "
    return False


def _download_direct(url: str, video_id: str, staging: Path) -> Path:
    extension = Path(urlparse(url).path).suffix.lower()
    target = staging / f"WLASL_{video_id}.{uuid.uuid4().hex}{extension}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (WLASL research candidate audit)"})
    with urlopen(request, timeout=40) as response, target.open("xb") as stream:
        if "text/html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("URL returned HTML rather than video")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
    if not _video_header_ok(target):
        raise ValueError("Downloaded file is too small or has an invalid video header")
    return target


def _download_with_yt_dlp(
    url: str, video_id: str, staging: Path, format_selector: str = "best[ext=mp4]/best",
) -> Path:
    import yt_dlp  # type: ignore[import-not-found]  # optional; checked before --execute

    token = uuid.uuid4().hex
    prefix = f"WLASL_{video_id}.{token}"
    template = str(staging / f"{prefix}.%(ext)s")
    options = {
        "outtmpl": template,
        "format": format_selector,
        "noplaylist": True,
        "overwrites": False,
        "continuedl": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 40,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
    if info is None:
        raise ValueError("yt-dlp returned no downloadable video")
    files = [
        path for path in staging.glob(f"{prefix}.*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if len(files) != 1:
        raise ValueError(f"yt-dlp produced {len(files)} supported video files")
    if not _video_header_ok(files[0]):
        raise ValueError("yt-dlp output is too small or has an invalid video header")
    return files[0]


def _load_prior_successes() -> dict[str, dict[str, str]]:
    if not DOWNLOAD_AUDIT.exists():
        return {}
    with DOWNLOAD_AUDIT.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(AUDIT_COLUMNS):
            raise ManifestError("Existing download audit has an unexpected schema")
        successes = {}
        for row in reader:
            if row["download_status"] in {"DOWNLOADED", "ALREADY_PRESENT_VERIFIED"}:
                successes[row["video_id"]] = row
        return successes


def _audit_event(row: dict[str, str], status: str, reason: str = "", path: Path | None = None) -> dict[str, str]:
    size = path.stat().st_size if path and path.is_file() else ""
    sha = _file_sha256(path) if path and path.is_file() else ""
    return {
        "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
        **{field: row[field] for field in ("video_id", "gloss", "signer_id", "source", "url", "frame_start", "frame_end", "split")},
        "local_path": path.relative_to(PROJECT_ROOT).as_posix() if path else "",
        "download_status": status,
        "failure_reason": reason[:500],
        "review_status": "REVIEW_REQUIRED",
        "file_size_bytes": size,
        "sha256": sha,
    }


def execute(rows: list[dict[str, str]]) -> Counter[str]:
    # Audit is append-only. Failed partial files stay in _staging for manual review;
    # neither this tool nor yt-dlp overwrites completed files.
    prior = _load_prior_successes()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    staging = RAW_DIR / "_staging"
    staging.mkdir(exist_ok=True)
    DOWNLOAD_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    is_new = not DOWNLOAD_AUDIT.exists()
    totals: Counter[str] = Counter()
    with DOWNLOAD_AUDIT.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS)
        if is_new:
            writer.writeheader()
        for row in rows:
            video_id = row["video_id"].strip()
            found = existing_videos(video_id)
            if found:
                if len(found) != 1:
                    event = _audit_event(row, "DOWNLOAD_FAILED", "Multiple existing files share this video_id; manual review required")
                else:
                    previous = prior.get(video_id)
                    is_verified = (
                        previous is not None and previous["local_path"] == found[0].relative_to(PROJECT_ROOT).as_posix()
                        and previous["sha256"] == _file_sha256(found[0]) and _video_header_ok(found[0])
                    )
                    if is_verified:
                        event = _audit_event(row, "ALREADY_PRESENT_VERIFIED", path=found[0])
                    else:
                        event = _audit_event(row, "DOWNLOAD_FAILED", "Existing file is not verified against a successful prior audit; not overwritten", found[0])
            elif download_method(row["url"]) == "unsupported_swf":
                event = _audit_event(row, "DOWNLOAD_FAILED", "SWF is not a directly usable video; no conversion attempted")
            else:
                try:
                    method = download_method(row["url"])
                    staged = (_download_direct(row["url"], video_id, staging)
                              if method == "direct" else _download_with_yt_dlp(row["url"], video_id, staging))
                    final = RAW_DIR / f"WLASL_{video_id}{staged.suffix.lower()}"
                    if final.exists():
                        raise FileExistsError(f"Final path already exists: {final}")
                    staged.rename(final)
                    event = _audit_event(row, "DOWNLOADED", path=final)
                    prior[video_id] = event
                except Exception as exc:  # one broken/private URL must not stop other candidates
                    event = _audit_event(row, "DOWNLOAD_FAILED", f"{type(exc).__name__}: {exc}")
            writer.writerow(event)
            stream.flush()
            totals[event["download_status"]] += 1
            print(f"{video_id}: {event['download_status']}")
    return totals


def _read_latest_audit(
    path: Path, expected_columns: tuple[str, ...], candidates: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(expected_columns):
            raise ManifestError(f"Unexpected download audit schema: {path}")
        latest: dict[str, dict[str, str]] = {}
        for event in reader:
            video_id = event["video_id"]
            candidate = candidates.get(video_id)
            if candidate is None:
                raise ManifestError(f"Audit references unknown video_id: {video_id}")
            for field in ("gloss", "signer_id", "source", "url", "frame_start", "frame_end", "split"):
                if event[field] != candidate[field]:
                    raise ManifestError(f"Audit provenance mismatch: {video_id}/{field}")
            if event["review_status"] != "REVIEW_REQUIRED":
                raise ManifestError(f"Audit changed candidate review status: {video_id}")
            latest[video_id] = event
    return latest


def _failure_category(event: dict[str, str]) -> str:
    reason = ANSI_ESCAPE.sub("", event["failure_reason"]).casefold()
    host = (urlparse(event["url"]).hostname or "").casefold()
    youtube = host in {"youtube.com", "www.youtube.com", "youtu.be", "www.youtu.be", "m.youtube.com"}
    if youtube and "requested format is not available" in reason:
        return "youtube_format_fallback"
    if "private video" in reason:
        return "private_video"
    if "this video is unavailable" in reason:
        return "video_unavailable"
    if "http error 404" in reason:
        return "http_404"
    if "swf is not a directly usable video" in reason:
        return "unsupported_swf"
    if "http error 403" in reason:
        return "http_403_source_review"
    if "url returned html rather than video" in reason:
        return "html_source_review"
    if "certificate_verify_failed" in reason and "hostname mismatch" in reason:
        return "ssl_hostname_source_review"
    return "unclassified_manual_review"


def build_retry_plan(
    rows: list[dict[str, str]], audit_v1: Path = DOWNLOAD_AUDIT,
    audit_v2: Path = RETRY_AUDIT,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], Counter[str]]:
    if not audit_v1.is_file():
        raise ManifestError(f"Missing original download audit: {audit_v1}")
    candidates = {row["video_id"]: row for row in rows}
    original = _read_latest_audit(audit_v1, AUDIT_COLUMNS, candidates)
    if set(original) != set(candidates):
        raise ManifestError("Original download audit does not cover all 220 candidates")
    retry_history = (
        _read_latest_audit(audit_v2, RETRY_AUDIT_COLUMNS, candidates)
        if audit_v2.is_file() else {}
    )
    if retry_history and set(retry_history) != set(candidates):
        raise ManifestError("Retry audit does not include all 220 baseline records")
    eligible: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        video_id = row["video_id"]
        event = original[video_id]
        if event["download_status"] in {"DOWNLOADED", "ALREADY_PRESENT_VERIFIED"}:
            counts["already_downloaded"] += 1
            continue
        if event["download_status"] != "DOWNLOAD_FAILED":
            raise ManifestError(f"Unexpected original download status: {video_id}")
        previous_retry = retry_history.get(video_id)
        if previous_retry and previous_retry["retry_attempted"] == "True":
            counts["already_retry_attempted"] += 1
            continue
        category = _failure_category(event)
        if category == "youtube_format_fallback":
            if existing_videos(video_id):
                counts["existing_file_manual_review"] += 1
                continue
            eligible.append(row)
            counts["youtube_format_fallback"] += 1
        else:
            counts[category] += 1
    return eligible, original, counts


def _initialize_retry_audit(
    original: dict[str, dict[str, str]], path: Path = RETRY_AUDIT,
) -> None:
    # New version; the historical v1 audit is never overwritten or reserialized.
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RETRY_AUDIT_COLUMNS)
        writer.writeheader()
        for event in original.values():
            writer.writerow({**event, **{field: "" for field in RETRY_COLUMNS}, "retry_attempted": "False"})


def execute_retry(
    eligible: list[dict[str, str]], original: dict[str, dict[str, str]],
    audit_v2: Path = RETRY_AUDIT,
) -> Counter[str]:
    _initialize_retry_audit(original, audit_v2)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    staging = RAW_DIR / "_staging"
    staging.mkdir(exist_ok=True)
    totals: Counter[str] = Counter()
    with audit_v2.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RETRY_AUDIT_COLUMNS)
        for row in eligible:
            video_id = row["video_id"]
            if existing_videos(video_id):
                # Never download over an unexpected file, even if it appeared
                # between --check-retry and --retry.
                event = _audit_event(row, "DOWNLOAD_FAILED", "Existing video found during retry; not overwritten")
                event.update({"retry_attempted": "False", "retry_method": "", "retry_status": "SKIPPED_EXISTING_FILE", "retry_failure_reason": event["failure_reason"]})
            else:
                try:
                    staged = _download_with_yt_dlp(
                        row["url"], video_id, staging, format_selector=YOUTUBE_FORMAT_FALLBACK,
                    )
                    final = RAW_DIR / f"WLASL_{video_id}{staged.suffix.lower()}"
                    if final.exists():
                        raise FileExistsError(f"Final path already exists: {final}")
                    staged.rename(final)
                    event = _audit_event(row, "DOWNLOADED", path=final)
                    event.update({"retry_attempted": "True", "retry_method": RETRY_METHOD, "retry_status": "RETRY_SUCCESS", "retry_failure_reason": ""})
                except Exception as exc:  # a failed URL must not stop the remaining eligible rows
                    reason = f"{type(exc).__name__}: {ANSI_ESCAPE.sub('', str(exc))}"[:500]
                    event = _audit_event(row, "DOWNLOAD_FAILED", reason)
                    event.update({"retry_attempted": "True", "retry_method": RETRY_METHOD, "retry_status": "RETRY_FAILED", "retry_failure_reason": reason})
            writer.writerow(event)
            stream.flush()
            totals[event["retry_status"]] += 1
            print(f"{video_id}: {event['retry_status']}")
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Offline manifest/dependency audit; no files written")
    mode.add_argument("--execute", action="store_true", help="Prompt and then download manifest candidates")
    mode.add_argument("--check-retry", action="store_true", help="Offline audit of recoverable failures; no files written")
    mode.add_argument("--retry", action="store_true", help="Prompt and retry only approved recoverable failures")
    args = parser.parse_args()
    try:
        rows, methods = load_candidates()
        if args.check_retry or args.retry:
            eligible, original, counts = build_retry_plan(rows)
            yt_dlp_available = importlib.util.find_spec("yt_dlp") is not None
            summary = {
                "candidate_count": len(rows),
                "already_downloaded_protected": counts["already_downloaded"],
                "retry_eligible_youtube_format": counts["youtube_format_fallback"],
                "retry_already_attempted": counts["already_retry_attempted"],
                "manual_source_review": {
                    "http_403": counts["http_403_source_review"],
                    "html_response": counts["html_source_review"],
                    "ssl_hostname_mismatch": counts["ssl_hostname_source_review"],
                    "existing_file": counts["existing_file_manual_review"],
                    "unclassified": counts["unclassified_manual_review"],
                },
                "permanent_skip": {
                    "private_video": counts["private_video"],
                    "video_unavailable": counts["video_unavailable"],
                    "http_404": counts["http_404"],
                    "unsupported_swf": counts["unsupported_swf"],
                },
                "youtube_format_fallback": YOUTUBE_FORMAT_FALLBACK,
                "yt_dlp_installed": yt_dlp_available,
                "ssl_verification_disabled": False,
                "retry_audit_path": RETRY_AUDIT.relative_to(PROJECT_ROOT).as_posix(),
                "all_candidates_review_status": "REVIEW_REQUIRED",
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            if args.check_retry:
                return 0
            if not eligible:
                print("No eligible, never-retried candidates; no files written.")
                return 0
            if not yt_dlp_available:
                print("STOP: yt-dlp is required for YouTube retry; no files written.", file=sys.stderr)
                return 2
            if input(f"Retry {len(eligible)} YouTube format failures only? [y/N] ").strip().lower() != "y":
                print("Cancelled; no retry attempted.")
                return 0
            print(json.dumps(dict(execute_retry(eligible, original)), ensure_ascii=False, sort_keys=True))
            return 0
        existing = {row["video_id"]: existing_videos(row["video_id"]) for row in rows}
        collisions = [video_id for video_id, paths in existing.items() if len(paths) > 1]
        if collisions:
            raise ManifestError(f"Ambiguous existing output paths for video_id: {collisions}")
        yt_dlp_available = importlib.util.find_spec("yt_dlp") is not None
        summary = {
            "candidate_count": len(rows),
            "direct_urls": methods["direct"],
            "yt_dlp_urls": methods["yt_dlp"],
            "unsupported_swf_urls": methods["unsupported_swf"],
            "existing_video_count": sum(bool(paths) for paths in existing.values()),
            "yt_dlp_installed": yt_dlp_available,
            "all_candidates_review_status": "REVIEW_REQUIRED",
            "download_root": RAW_DIR.relative_to(PROJECT_ROOT).as_posix(),
            "audit_path": DOWNLOAD_AUDIT.relative_to(PROJECT_ROOT).as_posix(),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.check_only:
            return 0
        if methods["yt_dlp"] and not yt_dlp_available:
            print("STOP: yt-dlp is required for YouTube/non-direct URLs; install it in this Python environment.", file=sys.stderr)
            return 2
        print("These are unreviewed WLASL candidates. Check source licensing and URLs before download.")
        if input(f"Download up to {len(rows)} manifest candidates? [y/N] ").strip().lower() != "y":
            print("Cancelled; no download attempted.")
            return 0
        print(json.dumps(dict(execute(rows)), ensure_ascii=False, sort_keys=True))
        return 0
    except (ManifestError, OSError, json.JSONDecodeError) as exc:
        print(f"PRECHECK_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
