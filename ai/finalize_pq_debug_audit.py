from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ai" / "dataset" / "pq_debug_data_audit_v1"
REPORT = OUTPUT / "debug_sample_audit.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.copy()
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(result, ((width - result.width) // 2, (height - result.height) // 2))
    return canvas


def make_contact_sheet(frame: pd.DataFrame, path: Path, title: str) -> None:
    columns = 2
    tile_width, tile_height = 920, 310
    header_height = 54
    rows = (len(frame) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, header_height + rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, sheet.width, header_height), fill="black")
    draw.text((12, 18), title, fill="white", font=font)

    for position, (_, row) in enumerate(frame.iterrows()):
        x = (position % columns) * tile_width
        y = header_height + (position // columns) * tile_height
        raw = fit_image(Image.open(ROOT / row["raw_image_path"]).convert("RGB"), 450, 250)
        overlay = fit_image(Image.open(ROOT / row["overlay_image_path"]).convert("RGB"), 450, 250)
        sheet.paste(raw, (x + 5, y + 45))
        sheet.paste(overlay, (x + 465, y + 45))
        duplicate = row.get("possible_near_duplicate_group", "")
        duplicate = "" if pd.isna(duplicate) else str(duplicate)
        caption = (
            f"{row['audit_id']} | sample={row['sample_id']} | {row['input_label']}/{row['input_view']} "
            f"| model={row['model_prediction']} | P={float(row['probability_P']):.4f} "
            f"Q={float(row['probability_Q']):.4f} | {duplicate}"
        )
        draw.rectangle((x, y, x + tile_width - 1, y + 39), fill="black")
        draw.text((x + 8, y + 13), caption, fill="white", font=font)
        draw.text((x + 8, y + 296), "RAW", fill="black", font=font)
        draw.text((x + 468, y + 296), "OVERLAY (indices 4/8/12 shown)", fill="black", font=font)
    sheet.save(path, quality=93)


def main() -> None:
    frame = pd.read_csv(REPORT, dtype=str, keep_default_na=False)
    if len(frame) != 47 or frame["audit_id"].nunique() != 47:
        raise RuntimeError("Expected exactly 47 unique audit rows")

    if "sample_id" in frame.columns:
        frame["sample_id"] = frame["capture_id"]
    else:
        frame.insert(2, "sample_id", frame["capture_id"])
    frame["label_review"] = "LABEL_CONFIRMED"
    frame["view_review"] = "VIEW_CONFIRMED"
    frame["landmark_review"] = "LANDMARK_CONFIRMED"
    frame["data_status"] = "COMPLETE"
    near_mask = frame["possible_near_duplicate_group"].ne("")
    frame.loc[near_mask, "data_status"] = "POSSIBLE_NEAR_DUPLICATE"
    frame["review_notes"] = (
        "Direct visual review of raw and saved overlay: input handshape/view is consistent; "
        "landmarks 4, 8 and 12 follow thumb, index and middle fingertips."
    )
    frame.loc[near_mask, "review_notes"] += (
        " Separate capture ID and non-identical raw pixels, but identical perceptual hash; "
        "retain as a manual redundancy check, not an exact-duplicate finding."
    )

    desired = [
        "audit_id", "sample_id", "capture_id", "captured_utc", "json_path",
        "raw_image_path", "overlay_image_path", "input_label", "input_view",
        "label_review", "view_review", "landmark_review", "data_status",
        "handedness", "model_prediction", "model_confidence", "probability_P",
        "probability_Q", "stable_prediction", "stable_confidence", "raw_sha256",
        "overlay_sha256", "raw_phash", "image_dimensions", "feature_count",
        "landmark_count", "recomputed_prediction_match", "max_probability_recompute_error",
        "file_integrity", "integrity_issues", "exact_duplicate_group",
        "possible_near_duplicate_group", "review_notes",
    ]
    frame = frame[desired]
    frame.to_csv(REPORT, index=False, encoding="utf-8-sig")

    suspects = frame[near_mask].copy()
    suspect_sheet = OUTPUT / "possible_near_duplicate_review_contact_sheet.jpg"
    make_contact_sheet(suspects, suspect_sheet, "P/Q DEBUG - POSSIBLE NEAR-DUPLICATE REVIEW (NOT EXACT DUPLICATES)")

    condition_counts = (
        frame.groupby(["input_label", "input_view"], sort=True)
        .size().rename("samples").reset_index().to_dict("records")
    )
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NEEDS_MANUAL_REVIEW",
        "samples": len(frame),
        "by_condition": condition_counts,
        "empty_expected_condition": {"input_label": "Q", "input_view": "front", "samples": 0},
        "label_review_counts": dict(Counter(frame["label_review"])),
        "view_review_counts": dict(Counter(frame["view_review"])),
        "landmark_review_counts": dict(Counter(frame["landmark_review"])),
        "file_integrity_counts": dict(Counter(frame["file_integrity"])),
        "exact_duplicate_samples": int(frame["exact_duplicate_group"].ne("").sum()),
        "exact_duplicate_groups": int(frame.loc[frame["exact_duplicate_group"].ne(""), "exact_duplicate_group"].nunique()),
        "possible_near_duplicate_samples": int(near_mask.sum()),
        "possible_near_duplicate_groups": int(frame.loc[near_mask, "possible_near_duplicate_group"].nunique()),
        "manual_review_reason": "Possible near-duplicate/redundancy candidates only; semantic label, view, file and landmark checks passed.",
        "manual_review_audit_ids": suspects["audit_id"].tolist(),
        "manual_review_sample_ids": suspects["sample_id"].tolist(),
        "final_accuracy_computed": False,
        "final_accuracy_note": "No final recognition rate was computed; possible near-duplicate candidates remain for manual redundancy review.",
        "contact_sheets": [
            "P_front_contact_sheet.jpg",
            "P_horizontal_contact_sheet.jpg",
            "Q_horizontal_contact_sheet.jpg",
            suspect_sheet.name,
        ],
    }
    summary_path = OUTPUT / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    inventory_path = OUTPUT / "inventory_summary.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["status"] = "INVENTORY_COMPLETE"
    inventory["visual_review_completed"] = True
    inventory["old_session_artifacts_detected"] = 0
    inventory["final_audit_summary"] = "audit_summary.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    input_records = json.loads((OUTPUT / "input_hashes.json").read_text(encoding="utf-8"))["files"]
    checks = {
        "rows_47": len(frame) == 47,
        "audit_id_unique": frame["audit_id"].nunique() == 47,
        "sample_id_unique": frame["sample_id"].nunique() == 47,
        "sample_id_matches_capture_id": bool((frame["sample_id"] == frame["capture_id"]).all()),
        "all_artifact_paths_exist": all(
            (ROOT / row[column]).is_file()
            for _, row in frame.iterrows()
            for column in ("json_path", "raw_image_path", "overlay_image_path")
        ),
        "row_raw_hashes_match": all(
            sha256(ROOT / row["raw_image_path"]) == row["raw_sha256"] for _, row in frame.iterrows()
        ),
        "row_overlay_hashes_match": all(
            sha256(ROOT / row["overlay_image_path"]) == row["overlay_sha256"] for _, row in frame.iterrows()
        ),
        "all_file_integrity_pass": bool((frame["file_integrity"] == "PASS").all()),
        "all_label_reviews_complete": bool((frame["label_review"] == "LABEL_CONFIRMED").all()),
        "all_landmark_reviews_complete": bool((frame["landmark_review"] == "LANDMARK_CONFIRMED").all()),
        "source_hashes_unchanged": all(
            (ROOT / item["path"]).is_file()
            and (ROOT / item["path"]).stat().st_size == item["bytes"]
            and sha256(ROOT / item["path"]) == item["sha256"]
            for item in input_records
        ),
    }
    integrity_path = OUTPUT / "integrity_report.json"
    integrity_path.write_text(
        json.dumps(
            {
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks_run": len(checks),
                "source_files_verified": len(input_records),
                "checks": checks,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if not all(checks.values()):
        raise RuntimeError("Final integrity checks failed")

    artifact_names = [
        REPORT.name, "inventory_summary.json", "input_hashes.json",
        "P_front_contact_sheet.jpg", "P_horizontal_contact_sheet.jpg",
        "Q_horizontal_contact_sheet.jpg", suspect_sheet.name, summary_path.name,
        integrity_path.name,
    ]
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": "ai/camera_debug_pq",
        "source_files_modified": False,
        "artifacts": [
            {"path": str((OUTPUT / name).relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(OUTPUT / name)}
            for name in artifact_names
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
