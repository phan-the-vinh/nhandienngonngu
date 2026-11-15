"""Inspect saved J/Z trajectory features without decoding video or rerunning extraction.

The SVG shows displacement relative to each landmark's first selected frame,
in units of the sequence's fixed pixel hand scale. It does not reconstruct
absolute image coordinates or motion between the 16 sampled frames.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl"
OUTPUT_DIR = ROOT / "data/experiments/dynamic_jz_v1/review/extracted_16_trajectory_v1"
TARGETS = {
    "J/311.avi": ("J", 20),
    "J/241.avi": ("J", 20),
    "Z/278.avi": ("Z", 8),
    "Z/189.avi": ("Z", 8),
}
TRAJECTORY_COLUMNS = {0: (63, 64), 8: (65, 66), 20: (67, 68)}


def load_records(path: Path = DATASET) -> dict[str, dict]:
    """Read only the four required JSONL records; reject missing/duplicate records."""
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not any(f"/{target}" in line.replace("\\", "/") for target in TARGETS):
                continue
            row = json.loads(line)
            source = str(row.get("source_path", "")).replace("\\", "/")
            matches = [target for target in TARGETS if source.endswith("/" + target)]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous source path on JSONL line {line_number}")
            target = matches[0]
            if target in records:
                raise ValueError(f"Duplicate JSONL record: {target}")
            validate_record(row, target)
            records[target] = row
    missing = set(TARGETS) - set(records)
    if missing:
        raise ValueError(f"Missing JSONL records: {', '.join(sorted(missing))}")
    return records


def validate_record(row: dict, target: str) -> None:
    label, _ = TARGETS[target]
    if row.get("label") != label or row.get("extraction_status") != "SUCCESS":
        raise ValueError(f"Invalid label/status: {target}")
    if row.get("sequence_length") != 16 or row.get("feature_dim") != 69:
        raise ValueError(f"Invalid sequence schema: {target}")
    indices = row.get("selected_original_frame_indices")
    if (not isinstance(indices, list) or len(indices) != 16
            or any(type(value) is not int or value < 0 for value in indices)
            or any(right <= left for left, right in zip(indices, indices[1:]))):
        raise ValueError(f"Invalid selected frame indices: {target}")
    scale = row.get("hand_scale_px")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid fixed hand scale: {target}")
    frames = row.get("features")
    if (not isinstance(frames, list) or len(frames) != 16
            or any(not isinstance(frame, list) or len(frame) != 69 for frame in frames)):
        raise ValueError(f"Invalid feature matrix: {target}")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value)
           for frame in frames for value in frame):
        raise ValueError(f"Non-finite features: {target}")
    if any(abs(frames[0][column]) > 1e-8 for column in range(63, 69)):
        raise ValueError(f"Trajectory origin is not zero: {target}")


def trajectory(row: dict, landmark: int) -> list[tuple[float, float]]:
    x_column, y_column = TRAJECTORY_COLUMNS[landmark]
    return [(float(frame[x_column]), float(frame[y_column])) for frame in row["features"]]


def _panel(title: str, values: list[tuple[float, float]], indices: list[int],
           x0: int, color: str) -> str:
    """Render one equal-axis XY panel, with source-frame labels at each point."""
    size, top = 500, 115
    xs = [x for x, _ in values] + [0.0]
    ys = [y for _, y in values] + [0.0]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.5)
    padding = max(0.18 * span, 0.25)
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    full_span = span + 2 * padding

    def pos(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (x0 + size * (x - cx + full_span / 2) / full_span,
                top + size * (y - cy + full_span / 2) / full_span)

    points = [pos(value) for value in values]
    origin = pos((0.0, 0.0))
    parts = [
        f'<text x="{x0}" y="95" fill="white" font-size="26">{escape(title)}</text>',
        f'<rect x="{x0}" y="{top}" width="{size}" height="{size}" '
        'fill="#111827" stroke="#64748b"/>',
        f'<line x1="{x0}" y1="{origin[1]:.2f}" x2="{x0+size}" '
        f'y2="{origin[1]:.2f}" stroke="#475569" stroke-dasharray="5,5"/>',
        f'<line x1="{origin[0]:.2f}" y1="{top}" x2="{origin[0]:.2f}" '
        f'y2="{top+size}" stroke="#475569" stroke-dasharray="5,5"/>',
        f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}" '
        f'fill="none" stroke="{color}" stroke-width="3"/>',
    ]
    for ordinal, ((x, y), frame_index) in enumerate(zip(points, indices)):
        marker = "#22c55e" if ordinal == 0 else "#ef4444" if ordinal == 15 else color
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{marker}"/>')
        offset_y = -10 if ordinal % 2 == 0 else 19
        parts.append(f'<text x="{x+7:.2f}" y="{y+offset_y:.2f}" '
                     f'fill="white" font-size="12">{frame_index}</text>')
    parts.append(f'<text x="{x0}" y="645" fill="#cbd5e1" font-size="15">'
                 'X → right; Y ↓ down; units: fixed palm scale</text>')
    return "\n".join(parts)


def render_svg(row: dict, target: str) -> str:
    label, tip = TARGETS[target]
    indices = row["selected_original_frame_indices"]
    tip_values = trajectory(row, tip)
    wrist_values = trajectory(row, 0)
    rows = []
    for index in range(16):
        x = 30 + (index // 8) * 580
        y = 765 + (index % 8) * 24
        tx, ty = tip_values[index]
        wx, wy = wrist_values[index]
        rows.append(f'<text x="{x}" y="{y}" fill="#e2e8f0" font-size="15">'
                    f'{index+1:02d}  frame {indices[index]:02d}   tip ({tx:+.3f}, {ty:+.3f})'
                    f'   wrist ({wx:+.3f}, {wy:+.3f})</text>')
    title = f"{target} — extracted 16 / landmark {tip}"
    return "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="995" '
        'viewBox="0 0 1200 995">',
        '<rect width="1200" height="995" fill="#0b1220"/>',
        f'<text x="30" y="45" fill="white" font-size="30">{escape(title)}</text>',
        _panel(f"{label}: tip {tip} displacement", tip_values, indices, 30, "#38bdf8"),
        _panel("Wrist 0 displacement", wrist_values, indices, 670, "#fbbf24"),
        '<text x="30" y="690" fill="#cbd5e1" font-size="17">'
        'Green = first; red = last; numbers = original 0-based video frames.</text>',
        '<text x="30" y="718" fill="#cbd5e1" font-size="17">'
        'Relative paths only: no absolute image position or unsampled motion is stored.</text>',
        '<text x="30" y="742" fill="white" font-size="18">Saved coordinates</text>',
        *rows,
        '</svg>',
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate four saved sequences; write nothing")
    mode.add_argument("--execute", action="store_true", help="Write four SVGs from saved features only")
    args = parser.parse_args()

    records = load_records()
    paths = {target: OUTPUT_DIR / f"{target[0]}_{Path(target).stem}_trajectory.svg"
             for target in TARGETS}
    if args.check_only:
        print(json.dumps({"status": "READY", "source": str(DATASET),
                          "videos": [{"video": target, "frames": row["selected_original_frame_indices"],
                                      "tip_landmark": TARGETS[target][1],
                                      "output": str(paths[target])}
                                     for target, row in records.items()]}, indent=2))
        return

    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for target, row in records.items():
        with paths[target].open("x", encoding="utf-8") as handle:
            handle.write(render_svg(row, target))
    print(json.dumps({"status": "CREATED", "paths": [str(path) for path in paths.values()]}, indent=2))


if __name__ == "__main__":
    main()
