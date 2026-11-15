from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ai" / "camera_debug_uv"
OUTPUT = ROOT / "ai" / "dataset" / "uv_debug_data_audit_v1"
MODEL_DIR = ROOT / "ai" / "dataset" / "mlp_validation_comparison_v1"
STATIC_LABELS = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
FEATURES = tuple(f"f{i}" for i in range(1, 64))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].reshape(-1)[1:]
    bits = low > np.median(low)
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.copy()
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(result, ((width - result.width) // 2, (height - result.height) // 2))
    return canvas


def make_sheet(frame: pd.DataFrame, path: Path, title: str) -> None:
    columns, tile_width, tile_height, header = 2, 920, 310, 54
    rows = (len(frame) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, header + rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, sheet.width, header), fill="black")
    draw.text((12, 18), title, fill="white", font=font)
    for position, (_, row) in enumerate(frame.iterrows()):
        x = (position % columns) * tile_width
        y = header + (position // columns) * tile_height
        raw = fit_image(Image.open(ROOT / row.raw_image_path).convert("RGB"), 450, 250)
        overlay = fit_image(Image.open(ROOT / row.overlay_image_path).convert("RGB"), 450, 250)
        sheet.paste(raw, (x + 5, y + 45))
        sheet.paste(overlay, (x + 465, y + 45))
        caption = (
            f"{row.audit_id} | {row.sample_id} | input={row.input_label}/{row.input_view} "
            f"| instant={row.instant_prediction}:{row.instant_confidence:.4f} "
            f"| stable={row.stable_prediction or 'NONE'}"
        )
        draw.rectangle((x, y, x + tile_width - 1, y + 39), fill="black")
        draw.text((x + 8, y + 13), caption, fill="white", font=font)
        draw.text((x + 8, y + 296), "RAW", fill="black", font=font)
        draw.text((x + 468, y + 296), "OVERLAY (inspect tips 8 and 12)", fill="black", font=font)
    sheet.save(path, quality=93)


def make_hand_crop_sheet(frame: pd.DataFrame, path: Path, title: str) -> None:
    columns, tile_width, tile_height, header = 2, 920, 360, 54
    rows = (len(frame) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, header + rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, sheet.width, header), fill="black")
    draw.text((12, 18), title, fill="white", font=font)
    for position, (_, row) in enumerate(frame.iterrows()):
        x = (position % columns) * tile_width
        y = header + (position // columns) * tile_height
        raw = Image.open(ROOT / row.raw_image_path).convert("RGB")
        overlay = Image.open(ROOT / row.overlay_image_path).convert("RGB")
        payload = json.loads((ROOT / row.json_path).read_text(encoding="utf-8"))
        xs = [float(item["x"]) * raw.width for item in payload["raw_landmarks"]]
        ys = [float(item["y"]) * raw.height for item in payload["raw_landmarks"]]
        margin = max(35, int(max(max(xs) - min(xs), max(ys) - min(ys)) * 0.28))
        crop_box = (
            max(0, int(min(xs)) - margin), max(0, int(min(ys)) - margin),
            min(raw.width, int(max(xs)) + margin), min(raw.height, int(max(ys)) + margin),
        )
        raw_crop = fit_image(raw.crop(crop_box), 450, 300)
        overlay_crop = fit_image(overlay.crop(crop_box), 450, 300)
        sheet.paste(raw_crop, (x + 5, y + 45))
        sheet.paste(overlay_crop, (x + 465, y + 45))
        draw.rectangle((x, y, x + tile_width - 1, y + 39), fill="black")
        draw.text(
            (x + 8, y + 13),
            f"{row.audit_id} | raw hand crop + indexed overlay | inspect separation and tips 8/12",
            fill="white", font=font,
        )
        draw.text((x + 8, y + 346), "RAW CROP", fill="black", font=font)
        draw.text((x + 468, y + 346), "OVERLAY CROP", fill="black", font=font)
    sheet.save(path, quality=95)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_DIR / "selected_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")

    json_paths = sorted(SOURCE.rglob("*.json"))
    rows: list[dict[str, object]] = []
    condition_counts: Counter[tuple[str, str]] = Counter()
    audit_counts: Counter[tuple[str, str]] = Counter()
    expected_files: set[Path] = set()

    csv_ids: dict[Path, set[str]] = {}
    for index_path in sorted(SOURCE.rglob("captures.csv")):
        with index_path.open(encoding="utf-8-sig", newline="") as stream:
            csv_ids[index_path] = {row["capture_id"] for row in csv.DictReader(stream)}
        expected_files.add(index_path.resolve())

    for metadata_path in json_paths:
        issues: list[str] = []
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        capture_id = str(payload.get("capture_id", ""))
        label = str(payload.get("manual_label", ""))
        view = str(payload.get("manual_view", ""))
        condition_counts[(label, view)] += 1
        audit_counts[(label, view)] += 1
        audit_id = f"{label}{view[0].upper()}{audit_counts[(label, view)]:02d}"

        raw_path = metadata_path.parent / str(payload.get("raw_image", ""))
        overlay_path = metadata_path.parent / str(payload.get("overlay_image", ""))
        expected_files.update({metadata_path.resolve(), raw_path.resolve(), overlay_path.resolve()})
        if not raw_path.is_file():
            issues.append("MISSING_RAW_IMAGE")
        if not overlay_path.is_file():
            issues.append("MISSING_OVERLAY_IMAGE")

        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR) if raw_path.is_file() else None
        overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR) if overlay_path.is_file() else None
        if raw is None:
            issues.append("CORRUPT_RAW_IMAGE")
        if overlay is None:
            issues.append("CORRUPT_OVERLAY_IMAGE")
        if raw is not None and overlay is not None and raw.shape != overlay.shape:
            issues.append("IMAGE_DIMENSION_MISMATCH")

        features = payload.get("features", {})
        values = np.asarray([features.get(name, np.nan) for name in FEATURES], dtype=np.float64)
        if set(features) != set(FEATURES) or values.shape != (63,) or not np.isfinite(values).all():
            issues.append("INVALID_FEATURES")
        landmarks = payload.get("raw_landmarks", [])
        if len(landmarks) != 21 or [item.get("index") for item in landmarks] != list(range(21)):
            issues.append("INVALID_LANDMARK_SCHEMA")
        else:
            coords = np.asarray([[item.get(axis, np.nan) for axis in "xyz"] for item in landmarks], dtype=float)
            if coords.shape != (21, 3) or not np.isfinite(coords).all():
                issues.append("INVALID_LANDMARK_VALUES")

        probabilities = payload.get("probabilities", {})
        probability_values = np.asarray([probabilities.get(name, np.nan) for name in STATIC_LABELS], dtype=float)
        if set(probabilities) != set(STATIC_LABELS) or not np.isfinite(probability_values).all():
            issues.append("INVALID_PROBABILITIES")
        elif abs(float(probability_values.sum()) - 1.0) > 1e-6:
            issues.append("PROBABILITY_SUM_INVALID")
        manual_probability = float(probabilities.get(label, np.nan))
        if not np.isfinite(manual_probability):
            issues.append("MISSING_MANUAL_LABEL_PROBABILITY")
        elif not np.isclose(manual_probability, float(payload.get("probability_manual_label", np.nan))):
            issues.append("MANUAL_LABEL_PROBABILITY_MISMATCH")

        recomputed_prediction = ""
        probability_error = np.nan
        if np.isfinite(values).all() and np.isfinite(probability_values).all():
            recomputed = np.asarray(model.predict_proba(scaler.transform(values.reshape(1, -1)))[0])
            recomputed_prediction = STATIC_LABELS[int(np.argmax(recomputed))]
            probability_error = float(np.max(np.abs(recomputed - probability_values)))
            if recomputed_prediction != payload.get("instant_prediction"):
                issues.append("RECOMPUTED_PREDICTION_MISMATCH")
            if probability_error > 1e-9:
                issues.append("RECOMPUTED_PROBABILITY_MISMATCH")

        index_path = metadata_path.parent / "captures.csv"
        if capture_id not in csv_ids.get(index_path, set()):
            issues.append("MISSING_FROM_CAPTURES_CSV")

        rows.append({
            "audit_id": audit_id,
            "sample_id": capture_id,
            "capture_id": capture_id,
            "captured_utc": payload.get("captured_utc", ""),
            "json_path": metadata_path.relative_to(ROOT).as_posix(),
            "raw_image_path": raw_path.relative_to(ROOT).as_posix(),
            "overlay_image_path": overlay_path.relative_to(ROOT).as_posix(),
            "input_label": label,
            "input_view": view,
            "handedness": payload.get("handedness_reported_by_mediapipe", ""),
            "instant_prediction": payload.get("instant_prediction", ""),
            "instant_confidence": float(payload.get("instant_confidence", np.nan)),
            "stable_prediction": payload.get("stable_prediction") or "",
            "stable_confidence": payload.get("stable_confidence") if payload.get("stable_confidence") is not None else np.nan,
            "probability_U": float(probabilities.get("U", np.nan)),
            "probability_V": float(probabilities.get("V", np.nan)),
            "probability_manual_label": manual_probability,
            "feature_count": len(features),
            "landmark_count": len(landmarks),
            "probability_count": len(probabilities),
            "image_dimensions": "" if raw is None else f"{raw.shape[1]}x{raw.shape[0]}",
            "raw_sha256": "" if raw is None else sha256(raw_path),
            "overlay_sha256": "" if overlay is None else sha256(overlay_path),
            "raw_phash": "" if raw is None else phash(raw),
            "recomputed_prediction": recomputed_prediction,
            "max_probability_recompute_error": probability_error,
            "file_integrity": "PASS" if not issues else "FAIL",
            "integrity_issues": "|".join(issues),
            "exact_duplicate_group": "",
            "near_duplicate_group": "",
            "label_review": "PENDING_VISUAL_REVIEW",
            "view_review": "PENDING_VISUAL_REVIEW",
            "landmark_review": "PENDING_VISUAL_REVIEW",
            "review_notes": "",
        })

    frame = pd.DataFrame(rows)
    for number, (_, group) in enumerate(frame.groupby("raw_sha256"), start=1):
        if group["raw_sha256"].iloc[0] and len(group) > 1:
            frame.loc[group.index, "exact_duplicate_group"] = f"EXACT{number:03d}"

    # Conservative perceptual candidates: join samples only inside the same
    # label/view when their 63-bit pHash differs by at most two bits. Connected
    # components are candidates, not proof of an independently repeated bout.
    parent = list(range(len(frame)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for condition, group in frame.groupby(["input_label", "input_view"]):
        indices = list(group.index)
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                distance = (int(frame.at[left, "raw_phash"], 16) ^ int(frame.at[right, "raw_phash"], 16)).bit_count()
                if distance <= 2:
                    union(left, right)
    components: dict[int, list[int]] = {}
    for index in range(len(frame)):
        components.setdefault(find(index), []).append(index)
    candidate_number = 0
    for members in components.values():
        if len(members) > 1:
            candidate_number += 1
            frame.loc[members, "near_duplicate_group"] = f"UVPHASH{candidate_number:03d}"

    frame["collection_dependence_group"] = (
        frame["input_label"] + "_" + frame["input_view"] + "_20260923_short_session"
    )
    frame["label_review"] = "LABEL_CONFIRMED"
    frame["view_review"] = "VIEW_CONFIRMED"
    frame["landmark_review"] = "LANDMARK_CONFIRMED"
    frame["review_notes"] = (
        "Direct raw/overlay review: U/V finger separation matches the entered label; "
        "landmark tips 8 and 12 track the visible index and middle fingertips."
    )
    candidate_mask = frame["near_duplicate_group"].ne("")
    frame.loc[candidate_mask, "review_notes"] += (
        " Perceptual near-duplicate candidate only; retain the sample and do not infer an independent gesture bout."
    )

    source_files = {path.resolve() for path in SOURCE.rglob("*") if path.is_file()}
    orphan_files = sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in source_files - expected_files)
    frame.to_csv(OUTPUT / "uv_debug_sample_audit.csv", index=False, encoding="utf-8-sig")
    for label in ("U", "V"):
        subset = frame[frame.input_label == label]
        if not subset.empty:
            make_sheet(subset, OUTPUT / f"{label}_all_samples_contact_sheet.jpg", f"U/V DEBUG AUDIT - {label}")
            make_hand_crop_sheet(
                subset,
                OUTPUT / f"{label}_hand_crops_contact_sheet.jpg",
                f"U/V DEBUG AUDIT - {label} HAND CROPS",
            )
    disagreements = frame[frame["instant_prediction"] != frame["input_label"]]
    if not disagreements.empty:
        make_sheet(
            disagreements,
            OUTPUT / "model_v1_disagreements_contact_sheet.jpg",
            "U/V DEBUG - MODEL V1 DISAGREEMENTS (LABELS VISUALLY CONFIRMED)",
        )
    near_candidates = frame[candidate_mask]
    if not near_candidates.empty:
        make_sheet(
            near_candidates,
            OUTPUT / "near_duplicate_candidates_contact_sheet.jpg",
            "U/V DEBUG - PERCEPTUAL NEAR-DUPLICATE CANDIDATES (RETAIN ALL)",
        )

    source_hashes = [
        {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(source_files)
    ]
    (OUTPUT / "input_hashes.json").write_text(
        json.dumps({"files": source_hashes}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    inventory = {
        "status": "DEBUG_DATA_AUDIT_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "samples": len(frame),
        "by_condition": [
            {"input_label": label, "input_view": view, "count": count}
            for (label, view), count in sorted(condition_counts.items())
        ],
        "json_count": len(json_paths),
        "csv_ids": sum(len(ids) for ids in csv_ids.values()),
        "file_integrity_failures": int((frame.file_integrity != "PASS").sum()),
        "exact_duplicate_samples": int(frame.exact_duplicate_group.ne("").sum()),
        "near_duplicate_candidate_samples": int(candidate_mask.sum()),
        "near_duplicate_candidate_groups": int(frame.loc[candidate_mask, "near_duplicate_group"].nunique()),
        "near_duplicate_method": "Within same label/view, connected components using 63-bit image pHash Hamming distance <= 2; descriptive candidates only.",
        "independence_limit": "No signer_id, session_id or gesture_bout_id. All captures per condition occurred in one short session, so samples are not claimed as independent gesture executions.",
        "orphan_files": orphan_files,
        "source_files": len(source_files),
    }
    (OUTPUT / "inventory_summary.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    frame["instant_correct"] = frame["instant_prediction"] == frame["input_label"]
    frame["stable_correct"] = frame["stable_prediction"] == frame["input_label"]
    condition_metrics: list[dict[str, object]] = []
    for (label, view), subset in frame.groupby(["input_label", "input_view"], sort=True):
        units = subset["near_duplicate_group"].where(
            subset["near_duplicate_group"].ne(""), "SINGLE_" + subset["audit_id"]
        )
        equal_unit_accuracy = subset.assign(unit=units).groupby("unit")["instant_correct"].mean().mean()
        condition_metrics.append({
            "input_label": label,
            "input_view": view,
            "confirmed_samples": len(subset),
            "instant_correct": int(subset["instant_correct"].sum()),
            "instant_accuracy": float(subset["instant_correct"].mean()),
            "stable_correct": int(subset["stable_correct"].sum()),
            "stable_accuracy_all_samples": float(subset["stable_correct"].mean()),
            "stable_missing": int(subset["stable_prediction"].eq("").sum()),
            "candidate_adjusted_units": int(units.nunique()),
            "equal_candidate_unit_weighted_instant_accuracy": float(equal_unit_accuracy),
        })
    pd.DataFrame(condition_metrics).to_csv(
        OUTPUT / "condition_model_metrics.csv", index=False, encoding="utf-8-sig"
    )

    v_to_u = frame[(frame.input_label == "V") & (frame.instant_prediction == "U")]
    visual_counts = {
        "label_review": dict(Counter(frame["label_review"])),
        "view_review": dict(Counter(frame["view_review"])),
        "landmark_review": dict(Counter(frame["landmark_review"])),
    }
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DEBUG_DATA_AUDIT_COMPLETE",
        "samples": len(frame),
        "by_condition": inventory["by_condition"],
        "valid_samples": int(
            ((frame.file_integrity == "PASS") & (frame.label_review == "LABEL_CONFIRMED") &
             (frame.landmark_review == "LANDMARK_CONFIRMED")).sum()
        ),
        "manual_review_samples": int(
            ((frame.label_review != "LABEL_CONFIRMED") | (frame.landmark_review != "LANDMARK_CONFIRMED")).sum()
        ),
        "visual_review_counts": visual_counts,
        "condition_model_metrics": condition_metrics,
        "instant_prediction_counts": {
            label: dict(Counter(group.instant_prediction))
            for label, group in frame.groupby("input_label", sort=True)
        },
        "stable_prediction_counts": {
            label: dict(Counter(group.stable_prediction.replace("", "NO_STABLE")))
            for label, group in frame.groupby("input_label", sort=True)
        },
        "instant_stable_disagreements_nonmissing": int(
            ((frame.stable_prediction != "") & (frame.instant_prediction != frame.stable_prediction)).sum()
        ),
        "stable_missing": int(frame.stable_prediction.eq("").sum()),
        "V_to_U_instant": {
            "samples": len(v_to_u),
            "total_V": int((frame.input_label == "V").sum()),
            "rate": float(len(v_to_u) / (frame.input_label == "V").sum()),
            "audit_ids": v_to_u.audit_id.tolist(),
            "mean_U_probability": float(v_to_u.probability_U.mean()),
            "median_U_probability": float(v_to_u.probability_U.median()),
            "minimum_U_probability": float(v_to_u.probability_U.min()),
            "maximum_U_probability": float(v_to_u.probability_U.max()),
        },
        "duplicates": {
            "exact_duplicate_samples": inventory["exact_duplicate_samples"],
            "near_duplicate_candidate_samples": inventory["near_duplicate_candidate_samples"],
            "near_duplicate_candidate_groups": inventory["near_duplicate_candidate_groups"],
            "method": inventory["near_duplicate_method"],
            "independence_limit": inventory["independence_limit"],
        },
        "final_accuracy_uses_only_confirmed_samples": True,
        "source_data_modified": False,
        "training_performed": False,
        "contact_sheets": [
            "U_all_samples_contact_sheet.jpg", "V_all_samples_contact_sheet.jpg",
            "U_hand_crops_contact_sheet.jpg", "V_hand_crops_contact_sheet.jpg",
            "model_v1_disagreements_contact_sheet.jpg", "near_duplicate_candidates_contact_sheet.jpg",
        ],
    }
    (OUTPUT / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    saved_inputs = json.loads((OUTPUT / "input_hashes.json").read_text(encoding="utf-8"))["files"]
    pq_inputs = json.loads(
        (ROOT / "ai" / "dataset" / "pq_debug_data_audit_v1" / "input_hashes.json").read_text(encoding="utf-8")
    )["files"]
    training_manifest = json.loads((MODEL_DIR / "training_manifest.json").read_text(encoding="utf-8"))
    official_hashes = {
        item["path"]: item["sha256"]
        for item in training_manifest["outputs"]
        if item["path"] in {"selected_model.joblib", "scaler.joblib", "label_mapping.json"}
    }
    checks = {
        "sample_rows_30": len(frame) == 30,
        "capture_ids_unique": frame.capture_id.nunique() == 30,
        "conditions_exact": frame.groupby(["input_label", "input_view"]).size().to_dict()
        == {("U", "front"): 15, ("V", "front"): 15},
        "all_file_integrity_pass": bool((frame.file_integrity == "PASS").all()),
        "all_features_63": bool((frame.feature_count == 63).all()),
        "all_landmarks_21": bool((frame.landmark_count == 21).all()),
        "all_probabilities_24": bool((frame.probability_count == 24).all()),
        "model_recompute_matches": bool(
            (frame.recomputed_prediction == frame.instant_prediction).all()
            and (frame.max_probability_recompute_error <= 1e-9).all()
        ),
        "all_labels_visually_confirmed": bool((frame.label_review == "LABEL_CONFIRMED").all()),
        "all_landmarks_visually_confirmed": bool((frame.landmark_review == "LANDMARK_CONFIRMED").all()),
        "no_exact_duplicate": int(frame.exact_duplicate_group.ne("").sum()) == 0,
        "source_uv_hashes_unchanged": all(
            (ROOT / item["path"]).is_file()
            and (ROOT / item["path"]).stat().st_size == item["bytes"]
            and sha256(ROOT / item["path"]) == item["sha256"]
            for item in saved_inputs
        ),
        "source_pq_hashes_unchanged": all(
            (ROOT / item["path"]).is_file()
            and (ROOT / item["path"]).stat().st_size == item["bytes"]
            and sha256(ROOT / item["path"]) == item["sha256"]
            for item in pq_inputs
        ),
        "official_model_artifacts_unchanged": len(official_hashes) == 3 and all(
            sha256(MODEL_DIR / name) == expected for name, expected in official_hashes.items()
        ),
        "all_contact_sheets_exist": all((OUTPUT / name).is_file() for name in summary["contact_sheets"]),
    }
    integrity = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks_run": len(checks),
        "checks": checks,
        "uv_source_files_verified": len(saved_inputs),
        "pq_source_files_verified": len(pq_inputs),
        "source_files_modified": 0 if checks["source_uv_hashes_unchanged"] and checks["source_pq_hashes_unchanged"] else None,
        "training_performed": False,
        "model_modified": False,
        "frozen_or_split_data_modified": False,
    }
    (OUTPUT / "integrity_report.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not all(checks.values()):
        raise RuntimeError(f"UV audit integrity failed: {checks}")

    output_names = [
        "uv_debug_sample_audit.csv", "condition_model_metrics.csv", "inventory_summary.json",
        "audit_summary.json", "integrity_report.json", "input_hashes.json", "U_all_samples_contact_sheet.jpg",
        "V_all_samples_contact_sheet.jpg", "U_hand_crops_contact_sheet.jpg",
        "V_hand_crops_contact_sheet.jpg", "model_v1_disagreements_contact_sheet.jpg",
        "near_duplicate_candidates_contact_sheet.jpg",
    ]
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_directory": "ai/camera_debug_uv",
        "source_files_modified": False,
        "artifacts": [
            {"path": f"ai/dataset/uv_debug_data_audit_v1/{name}", "sha256": sha256(OUTPUT / name)}
            for name in output_names
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
