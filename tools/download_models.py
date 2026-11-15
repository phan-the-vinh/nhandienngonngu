"""Model Asset Verification and Download Utility.

Downloads official third-party model assets (MediaPipe Hand Landmarker)
and verifies SHA-256 checksums of all runtime models.

Usage:
    python tools/download_models.py
    python tools/download_models.py --check-only
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]

# Registry of downloadable external models
DOWNLOADABLE_ASSETS: dict[str, dict[str, Any]] = {
    "hand_landmarker.task": {
        "description": "Google MediaPipe Hand Landmarker Model Bundle (float16)",
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "destination": ROOT / "models" / "mediapipe" / "hand_landmarker.task",
        "expected_sha256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        "expected_size_bytes": 7819105,
        "license": "Apache-2.0",
        "source": "Google MediaPipe official storage"
    }
}

# Registry of local custom production assets to verify
RUNTIME_LOCAL_ASSETS: dict[str, dict[str, Any]] = {
    "letters_model": {
        "description": "ASL Static Letters Model (uvw_medium)",
        "path": ROOT / "models" / "letters" / "production" / "uvw_medium_model.joblib",
        "expected_sha256": "88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45",
    },
    "digits_model": {
        "description": "Static Digits Model (digits_final)",
        "path": ROOT / "models" / "digits" / "production" / "digits_final.joblib",
        "expected_sha256": "43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c",
    },
    "space_model": {
        "description": "SPACE Gesture Model (final_space_model)",
        "path": ROOT / "models" / "space" / "production" / "final_space_model.joblib",
        "expected_sha256": "453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121",
    },
    "jz_model": {
        "description": "Dynamic J/Z Baseline Model (model_pipeline)",
        "path": ROOT / "data" / "experiments" / "dynamic_jz_binary_baseline_v1" / "model_pipeline.joblib",
        "expected_sha256": "8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497",
    },
    "english_dictionary": {
        "description": "Offline English Word Suggestion Dictionary (25k words)",
        "path": ROOT / "data" / "dictionary" / "english_words_25k.tsv",
        "expected_sha256": "986eead866a09df979d0c95a8250f235cb4bdf0bdadd5b3699db610e3ad4525e",
    }
}


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(asset_name: str, meta: dict[str, Any], check_only: bool = False) -> bool:
    dst: Path = meta["destination"]
    expected_sha: str = meta["expected_sha256"]

    if dst.is_file():
        actual_sha = compute_sha256(dst)
        if actual_sha == expected_sha:
            print(f"[EXISTS] {asset_name}: Verified matching SHA-256 at {dst.relative_to(ROOT)}")
            return True
        else:
            print(f"[CORRUPT] {asset_name}: File exists but SHA-256 mismatch!\n  Expected: {expected_sha}\n  Actual:   {actual_sha}", file=sys.stderr)
            if check_only:
                return False
    elif check_only:
        print(f"[MISSING] {asset_name}: File not found at {dst.relative_to(ROOT)}", file=sys.stderr)
        return False

    url: str = meta["url"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")

    print(f"[DOWNLOAD] Downloading {asset_name} from {url} ...")
    try:
        urllib.request.urlretrieve(url, tmp_dst)
    except urllib.error.URLError as err:
        print(f"[ERROR] Download failed for {asset_name}: {err}", file=sys.stderr)
        if tmp_dst.exists():
            tmp_dst.unlink()
        return False

    actual_sha = compute_sha256(tmp_dst)
    if actual_sha != expected_sha:
        print(f"[ERROR] Downloaded file SHA-256 mismatch!\n  Expected: {expected_sha}\n  Actual:   {actual_sha}", file=sys.stderr)
        tmp_dst.unlink()
        return False

    tmp_dst.replace(dst)
    print(f"[OK] {asset_name} downloaded and verified successfully: {dst.relative_to(ROOT)}")
    return True


def check_local_assets() -> bool:
    print("\n--- Verifying Project Production Runtime Assets ---")
    all_ok = True
    for key, info in RUNTIME_LOCAL_ASSETS.items():
        p: Path = info["path"]
        desc: str = info["description"]
        expected_sha: str = info["expected_sha256"]
        if not p.is_file():
            print(f"[MISSING] {desc}: NOT found at {p.relative_to(ROOT)}", file=sys.stderr)
            all_ok = False
            continue
        actual_sha = compute_sha256(p)
        if actual_sha != expected_sha:
            print(f"[CORRUPT] {desc}: Hash mismatch at {p.relative_to(ROOT)}", file=sys.stderr)
            all_ok = False
        else:
            print(f"[OK] {desc} ({p.relative_to(ROOT)}): Verified SHA-256")
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify NhanDienCuChi models.")
    parser.add_argument("--check-only", action="store_true", help="Verify existence and checksums without downloading.")
    args = parser.parse_args(argv)

    print("=== NhanDienCuChi Model Asset Manager ===")
    downloads_ok = True
    for name, meta in DOWNLOADABLE_ASSETS.items():
        if not download_asset(name, meta, check_only=args.check_only):
            downloads_ok = False

    local_ok = check_local_assets()

    if downloads_ok and local_ok:
        print("\n[SUCCESS] All runtime models and assets are present and verified!")
        return 0
    else:
        print("\n[FAILURE] One or more assets are missing or invalid.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
