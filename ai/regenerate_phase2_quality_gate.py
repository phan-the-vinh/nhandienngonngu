"""Regenerate Phase-2 promotion outputs with the required quality gate.

This utility is intentionally inference-free: it reuses the existing REVIEW
scores, validated policies, and source features.  It never fits an MLP/kNN and
never runs MediaPipe.  Outputs are staged in a new directory and atomically
renamed only after all integrity checks pass.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_review_promotion import (
    FEATURE_COLUMNS,
    META_COLUMNS,
    QUALITY_FLAG_COLUMNS,
    apply_quality_gate_to_decisions,
    normalize_required_bool,
    normalize_quality_flags,
)


def parse_args() -> argparse.Namespace:
    ai_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase2-run",
        type=Path,
        default=ai_dir / "phase2_outputs" / "run_20260917_111456_626441",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=ai_dir / "dataset" / "landmark_quality_report.csv",
    )
    parser.add_argument(
        "--all-features",
        type=Path,
        default=ai_dir / "dataset" / "dataset_chucai_all.csv",
    )
    parser.add_argument(
        "--clean-features",
        type=Path,
        default=ai_dir / "dataset" / "dataset_chucai_clean.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ai_dir / "dataset" / "phase2_quality_gate_v2",
    )
    return parser.parse_args()


def _assert_unique_file(frame: pd.DataFrame, name: str) -> None:
    duplicates = frame["file"].duplicated(keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, "file"].head(5).tolist()
        raise ValueError(f"{name} has duplicate file keys: {examples}")


def _join_quality(
    decisions: pd.DataFrame, quality_path: Path
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    quality_columns = META_COLUMNS + [
        "final_status",
        "extraction_rejected",
        *QUALITY_FLAG_COLUMNS,
    ]
    quality = pd.read_csv(quality_path, usecols=quality_columns, low_memory=False)
    _assert_unique_file(decisions, "review decisions")
    _assert_unique_file(quality, "landmark quality report")

    joined = decisions.merge(
        quality,
        on="file",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_quality"),
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError(
            f"{int((joined['_merge'] != 'both').sum())} decisions lack quality rows"
        )
    for column in ["source", "label", "handedness"]:
        quality_column = f"{column}_quality"
        mismatch = joined[column].astype(str) != joined[quality_column].astype(str)
        if mismatch.any():
            raise ValueError(
                f"{int(mismatch.sum())} decisions disagree on {column} after file join"
            )
    joined = joined.drop(
        columns=["_merge", "source_quality", "label_quality", "handedness_quality"]
    )
    joined, counts = normalize_quality_flags(joined)
    joined["extraction_rejected"] = normalize_required_bool(
        joined["extraction_rejected"], "extraction_rejected"
    )
    if joined["extraction_rejected"].any():
        raise ValueError("An extraction-rejected row reached REVIEW decisions")
    if not joined["final_status"].eq("REVIEW").all():
        raise ValueError("A non-REVIEW quality row reached REVIEW decisions")
    return joined, counts


def _quality_block_reason(row: pd.Series) -> str:
    return "|".join(column for column in QUALITY_FLAG_COLUMNS if bool(row[column]))


def promotion_membership_changed(
    previous: pd.Series, current: pd.Series
) -> pd.Series:
    """Return true only when PROMOTED membership changed between versions."""

    return previous.eq("PROMOTED") != current.eq("PROMOTED")


def _verify_model_policy(
    promoted: pd.DataFrame, policy_path: Path
) -> dict[str, int]:
    policies = pd.read_csv(policy_path)
    enabled = policies.loc[
        policies["final_policy_fitted"].eq(True), "label"
    ].astype(str)
    enabled_set = set(enabled)
    policy_by_label = policies.set_index("label")
    violations = {
        "bad_geometry": int(promoted["bad_geometry"].sum()),
        "bad_stability": int(promoted["bad_stability"].sum()),
        "bad_image_quality": int(promoted["bad_image_quality"].sum()),
        "class_enabled": int((~promoted["label"].isin(enabled_set)).sum()),
        "consensus": int(
            (~promoted["model_consensus"].eq(True)
             | promoted["consensus_pred"].ne(promoted["label"])).sum()
        ),
        "confidence": 0,
        "margin": 0,
        "knn": 0,
        "distance": 0,
    }
    for label, rows in promoted.groupby("label"):
        policy = policy_by_label.loc[label]
        violations["confidence"] += int(
            (rows["mlp_confidence"] < policy["mlp_confidence_min"]).sum()
        )
        violations["margin"] += int(
            (rows["mlp_margin"] < policy["mlp_margin_min"]).sum()
        )
        violations["knn"] += int(
            (rows["knn_vote"] < policy["knn_vote_min"]).sum()
        )
        violations["distance"] += int(
            (rows["knn_mean_distance"] > policy["knn_mean_distance_max"]).sum()
        )
    if any(violations.values()):
        raise RuntimeError(f"Post-fix invariant violation: {violations}")
    return violations


def _write_outputs(
    output_dir: Path,
    decisions: pd.DataFrame,
    promoted_features: pd.DataFrame,
    remaining_features: pd.DataFrame,
    clean_reviewed: pd.DataFrame,
    summary: dict,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.in_progress")
    if staging.exists():
        raise FileExistsError(f"Incomplete staging directory already exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    decisions.to_csv(staging / "review_promotion_decisions_v2.csv", index=False)
    promoted_features.to_csv(staging / "review_promoted_to_keep_v2.csv", index=False)
    remaining_features.to_csv(staging / "review_remaining_v2.csv", index=False)
    clean_reviewed.to_csv(
        staging / "dataset_chucai_clean_reviewed_v2.csv", index=False
    )
    (staging / "phase2_quality_gate_summary_v2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(staging, output_dir)


def main() -> int:
    args = parse_args()
    decision_path = args.phase2_run / "review_promotion_decisions.csv"
    policy_path = args.phase2_run / "final_policy_thresholds.csv"
    decisions = pd.read_csv(decision_path, low_memory=False)
    previous = decisions["promotion_decision"].copy()
    decisions, quality_counts = _join_quality(decisions, args.quality_report)
    decisions["promotion_decision_v1"] = previous
    decisions = apply_quality_gate_to_decisions(decisions)
    decisions["quality_block_reason"] = decisions.apply(
        _quality_block_reason, axis=1
    )
    # REVIEW -> KEEP_REVIEW is only a clearer name, not a semantic decision
    # change.  A change means that promotion membership itself changed.
    decisions["decision_changed"] = promotion_membership_changed(
        decisions["promotion_decision_v1"], decisions["promotion_decision"]
    )

    promoted = decisions.loc[decisions["promotion_decision"] == "PROMOTED"].copy()
    remaining = decisions.loc[
        decisions["promotion_decision"] == "KEEP_REVIEW"
    ].copy()
    if len(promoted) + len(remaining) != len(decisions):
        raise RuntimeError("V2 decisions do not partition the REVIEW input")

    # Features are payload, not computation inputs in this utility.  Keep their
    # original decimal text so writing V2 cannot introduce float reserialization
    # drift, even at ~1e-21 scale.
    feature_dtypes = {column: "string" for column in FEATURE_COLUMNS}
    features = pd.read_csv(
        args.all_features,
        usecols=META_COLUMNS + FEATURE_COLUMNS,
        dtype=feature_dtypes,
    )
    clean = pd.read_csv(
        args.clean_features,
        usecols=META_COLUMNS + FEATURE_COLUMNS,
        dtype=feature_dtypes,
    )
    _assert_unique_file(features, "all features")
    _assert_unique_file(clean, "clean features")

    def attach_features(rows: pd.DataFrame, status: str) -> pd.DataFrame:
        attached = rows[META_COLUMNS + QUALITY_FLAG_COLUMNS].merge(
            features,
            on=META_COLUMNS,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not (attached["_merge"] == "both").all():
            raise ValueError("A V2 decision lacks source f1..f63")
        attached = attached.drop(columns="_merge")
        attached.insert(4, "phase2_status", status)
        return attached

    promoted_features = attach_features(promoted, "PROMOTE_TO_KEEP")
    remaining_features = attach_features(remaining, "KEEP_REVIEW")
    clean_reviewed = pd.concat(
        [
            clean[META_COLUMNS + FEATURE_COLUMNS],
            promoted_features[META_COLUMNS + FEATURE_COLUMNS],
        ],
        ignore_index=True,
    )
    _assert_unique_file(clean_reviewed, "clean reviewed V2")
    if clean_reviewed[META_COLUMNS].duplicated().any():
        raise RuntimeError("clean reviewed V2 has duplicate metadata keys")
    if clean_reviewed[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("clean reviewed V2 contains missing feature values")

    violations = _verify_model_policy(promoted, policy_path)
    old_promoted = int(previous.eq("PROMOTED").sum())
    changed = decisions.loc[decisions["decision_changed"], "file"].tolist()
    expected_clean_count = len(clean) + len(promoted)
    if len(clean_reviewed) != expected_clean_count:
        raise RuntimeError("Unexpected clean reviewed V2 row count")

    summary = {
        "status": "SAFE_AFTER_QUALITY_GATE_FIX",
        "implementation": "minimal_safe_fix_only",
        "model_retrained": False,
        "mediapipe_rerun": False,
        "source_phase2_run": str(args.phase2_run.resolve()),
        "quality_flag_counts_before_decision": quality_counts,
        "counts": {
            "initial_keep": len(clean),
            "review": len(decisions),
            "promoted_v1": old_promoted,
            "promoted_v2": len(promoted),
            "keep_review_v2": len(remaining),
            "downgraded": len(changed),
            "clean_reviewed_v2": len(clean_reviewed),
        },
        "changed_files": changed,
        "post_fix_violations": violations,
        "strict_spec_gaps_not_implemented": [
            "5-fold MLP ensemble inference on REVIEW",
            "entropy/uncertainty decision threshold",
        ],
    }
    _write_outputs(
        args.output_dir,
        decisions,
        promoted_features,
        remaining_features,
        clean_reviewed,
        summary,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
