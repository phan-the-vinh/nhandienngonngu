"""Create a reproducible visual audit of saved landmark quality decisions.

The utility is read-only with respect to source images and dataset CSV files. It
draws the exact raw landmarks used by the quality pipeline; MediaPipe is not run
again, so previews cannot silently differ from the audited records.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


AI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AI_DIR.parent
REPORT_FILE = AI_DIR / "dataset" / "landmark_quality_report.csv"
RAW_FILE = AI_DIR / "raw_data" / "raw_landmarks_chucai_all.csv"
OUTPUT_DIR = AI_DIR / "audit_landmark_previews"

RANDOM_STATE = 42
SAMPLE_SIZE = 20
MAX_IMAGE_SIDE = 1100

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def safe_number(value, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def make_samples(report: pd.DataFrame) -> pd.DataFrame:
    extraction_rejected = safe_bool(report["extraction_rejected"])
    quality_excluded = report[
        report["final_status"].eq("EXCLUDE_FROM_TRAIN") & ~extraction_rejected
    ].copy()
    if len(quality_excluded) < SAMPLE_SIZE:
        raise ValueError("Không đủ 20 quality-filtering EXCLUDE để audit")
    exclude_sample = quality_excluded.sample(
        n=SAMPLE_SIZE, random_state=RANDOM_STATE
    ).copy()
    exclude_sample["sample_group"] = "QUALITY_EXCLUDE_RANDOM"

    review = report[report["final_status"].eq("REVIEW")].copy()
    high_risk_mask = (
        safe_bool(review["bad_geometry"])
        | safe_bool(review["bad_image_quality"])
        | safe_bool(review["bad_stability"])
    )
    high_risk_pool = review[high_risk_mask].copy()
    if len(high_risk_pool) < SAMPLE_SIZE:
        raise ValueError("Không đủ 20 REVIEW có cờ bad_* để audit")
    review_sample = high_risk_pool.sample(
        n=SAMPLE_SIZE, random_state=RANDOM_STATE
    ).copy()
    review_sample["sample_group"] = "HIGH_RISK_REVIEW_RANDOM"

    samples = pd.concat([exclude_sample, review_sample], ignore_index=True)
    score_columns = ["geometry_score", "image_quality_score", "stability_score"]
    samples["audit_risk_score"] = samples[score_columns].apply(
        pd.to_numeric, errors="coerce"
    ).max(axis=1)
    samples["sample_order"] = samples.groupby("sample_group").cumcount() + 1
    return samples


def resize_for_preview(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIDE / max(height, width))
    if scale == 1.0:
        return image.copy()
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def draw_preview(image: np.ndarray, raw_row: pd.Series, row: pd.Series) -> np.ndarray:
    canvas = resize_for_preview(image)
    height, width = canvas.shape[:2]
    raw_points = [
        (float(raw_row[f"x{i}"]), float(raw_row[f"y{i}"])) for i in range(21)
    ]
    points = [
        (
            int(np.clip(x * width, 0, width - 1)),
            int(np.clip(y * height, 0, height - 1)),
        )
        for x, y in raw_points
    ]

    thickness = max(1, round(min(height, width) / 350))
    radius = max(3, round(min(height, width) / 130))
    for start, end in CONNECTIONS:
        cv2.line(canvas, points[start], points[end], (0, 220, 0), thickness)
    tips = {4, 8, 12, 16, 20}
    for index, ((raw_x, raw_y), point) in enumerate(zip(raw_points, points)):
        outside = raw_x < 0 or raw_x > 1 or raw_y < 0 or raw_y > 1
        color = (255, 0, 255) if outside else ((0, 255, 255) if index in tips else (0, 0, 255))
        if index == 0:
            color = (255, 255, 0)
        cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas, str(index), (point[0] + radius, point[1] - radius),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

    banner_height = 132
    output = cv2.copyMakeBorder(
        canvas, banner_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)
    )
    lines = [
        f"{row['sample_group']} #{int(row['sample_order']):02d} | label={row['label']} | {row['handedness']}",
        f"status={row['final_status']} | reason={row.get('reason', '')}",
        "geo=" + safe_number(row.get("geometry_score"))
        + f" bad={bool(row.get('bad_geometry', False))}"
        + " | image=" + safe_number(row.get("image_quality_score"))
        + f" bad={bool(row.get('bad_image_quality', False))}",
        "stability=" + safe_number(row.get("stability_score"))
        + f" bad={bool(row.get('bad_stability', False))}"
        + " | failures=" + safe_number(row.get("stability_failure_count"), 0)
        + "/5 | mean/max drift=" + safe_number(row.get("mean_drift"), 4)
        + "/" + safe_number(row.get("max_drift"), 4),
        str(row["file"]),
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            output, text[:150], (10, 22 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA,
        )
    return output


def safe_stem(file_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(file_name).stem)[:70]


def make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    tile_width, tile_height = 400, 310
    tiles: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        y = (tile_height - resized.shape[0]) // 2
        x = (tile_width - resized.shape[1]) // 2
        tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        tiles.append(tile)
    rows = []
    for start in range(0, len(tiles), 4):
        row = tiles[start:start + 4]
        while len(row) < 4:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(output_path), np.vstack(rows))


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output audit đã tồn tại; không ghi đè dữ liệu audit cũ: {OUTPUT_DIR}"
        )
    report = pd.read_csv(REPORT_FILE, low_memory=False)
    raw = pd.read_csv(RAW_FILE)
    raw_by_file = raw.set_index("file", drop=False)
    samples = make_samples(report)
    OUTPUT_DIR.mkdir(parents=True)

    written_by_group: dict[str, list[Path]] = {}
    preview_paths: list[str] = []
    for row in samples.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        image_path = PROJECT_DIR / Path(row.file)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Không đọc được ảnh audit: {image_path}")
        if row.file not in raw_by_file.index:
            raise ValueError(f"Không có raw landmark: {row.file}")
        group_dir = OUTPUT_DIR / row.sample_group.lower()
        group_dir.mkdir(exist_ok=True)
        output_name = (
            f"{int(row.sample_order):02d}_{row.label}_"
            f"risk_{safe_number(row.audit_risk_score)}_{safe_stem(row.file)}.jpg"
        )
        output_path = group_dir / output_name
        preview = draw_preview(image, raw_by_file.loc[row.file], row_series)
        if not cv2.imwrite(str(output_path), preview):
            raise OSError(f"Không ghi được preview: {output_path}")
        written_by_group.setdefault(row.sample_group, []).append(output_path)
        preview_paths.append(str(output_path.relative_to(PROJECT_DIR)))

    samples["preview_file"] = preview_paths
    manifest_columns = [
        "sample_group", "sample_order", "file", "preview_file", "label",
        "handedness", "final_status", "reason", "evidence", "audit_risk_score",
        "geometry_score", "bad_geometry", "very_close_pairs", "tiny_bones",
        "min_pair", "min_bone", "image_quality_score", "bad_image_quality",
        "stability_score", "bad_stability", "stability_failure_count",
        "failure_rate", "mean_drift", "max_drift",
    ]
    samples[manifest_columns].to_csv(OUTPUT_DIR / "audit_manifest.csv", index=False)
    for group, paths in written_by_group.items():
        make_contact_sheet(paths, OUTPUT_DIR / f"contact_sheet_{group.lower()}.jpg")

    print(f"Đã tạo {len(samples)} preview tại {OUTPUT_DIR}")
    print(samples.groupby("sample_group").size().to_string())
    print("High-risk REVIEW pool:", int((samples["sample_group"] == "HIGH_RISK_REVIEW_RANDOM").sum()))


if __name__ == "__main__":
    main()
