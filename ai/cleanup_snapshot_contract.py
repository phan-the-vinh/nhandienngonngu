"""Read-only resolver for retired, byte-exact cleanup artefacts.

Historical manifests remain immutable. Their file references can be resolved
through the verified snapshot without restoring into the live project.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_MANIFEST = ROOT / "ai/dataset/deep_cleanup_snapshot_v1/snapshot_manifest.json"
PROMOTION_DIR = Path("AI/final_promotion_audit/run_20260917_181104_976231")
# NOTE: PROMOTION_DIR / LIVE_DEBUG_DIR intentionally keep the historical "AI/"
# prefix: they are matched against the frozen snapshot manifest keys
# (original_relative_path) which still record "AI/...".
LIVE_DEBUG_DIR = Path("AI/camera_debug_uv_v2_live/V_front")


def _key(path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    parts = PurePosixPath(value).parts
    if not parts or any(part in ("..", ".") for part in parts) or value.startswith("/"):
        raise ValueError(f"Unsafe project-relative path: {path}")
    return "/".join(parts)


def snapshot_index(manifest_path: Path = SNAPSHOT_MANIFEST) -> dict[str, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("verification", {}).get("original_vs_snapshot_sha256") != "PASS_212_OF_212":
        raise ValueError("Snapshot verification contract is not PASS_212_OF_212")
    entries = {_key(item["original_relative_path"]): item for item in manifest["files"]}
    if len(entries) != 212:
        raise ValueError("Snapshot entries are missing or duplicate")
    return entries


def resolve_snapshot(
    original_relative_path: str | Path,
    *,
    manifest_path: Path = SNAPSHOT_MANIFEST,
    verify: bool = True,
) -> Path:
    """Return a verified snapshot copy for an original project-relative path."""
    key = _key(original_relative_path)
    entry = snapshot_index(manifest_path).get(key)
    if entry is None:
        raise FileNotFoundError(f"Not covered by snapshot: {key}")
    relative = _key(entry["snapshot_relative_path"])
    if not relative.startswith("payload/"):
        raise ValueError(f"Unexpected snapshot payload path: {relative}")
    path = manifest_path.parent / relative
    if verify:
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"Snapshot size mismatch: {key}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {key}")
    return path


def verify_historical_visual_indices() -> dict[str, int]:
    """Check D23/D24/D25 references against snapshot, without editing indices."""
    counts: dict[str, int] = {}
    with (ROOT / "ai/audit_landmark_previews/audit_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        paths = {_key(row["preview_file"]) for row in csv.DictReader(handle)}
    for path in paths:
        resolve_snapshot(path)
    counts["D23_unique_previews"] = len(paths)

    promotion_paths: set[str] = set()
    for name in ("combined_manifest.csv", "random30_manifest.csv", "bad_stability4_manifest.csv"):
        with (ROOT / PROMOTION_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            promotion_paths.add(_key(PROMOTION_DIR / row["overlay_file"]))
        counts[f"D24_{name}_rows"] = len(rows)
    for path in promotion_paths:
        resolve_snapshot(path)
    counts["D24_unique_overlays"] = len(promotion_paths)

    debug_paths: set[str] = set()
    with (ROOT / LIVE_DEBUG_DIR / "captures.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        debug_paths.add(_key(LIVE_DEBUG_DIR / row["overlay_image"]))
        details = json.loads((ROOT / LIVE_DEBUG_DIR / row["details_json"]).read_text(encoding="utf-8"))
        debug_paths.add(_key(LIVE_DEBUG_DIR / details["overlay_image"]))
    for path in debug_paths:
        resolve_snapshot(path)
    counts["D25_capture_rows"] = len(rows)
    counts["D25_unique_overlays"] = len(debug_paths)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-indices", action="store_true")
    parser.add_argument("--resolve", metavar="PROJECT_RELATIVE_PATH")
    args = parser.parse_args()
    if args.verify_indices:
        print(json.dumps(verify_historical_visual_indices(), sort_keys=True))
    if args.resolve:
        print(resolve_snapshot(args.resolve))
