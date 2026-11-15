"""PHASE 2: conservative, leakage-safe REVIEW auto-promotion.

The script never edits an existing CSV.  Every invocation creates a new run
directory below ``AI/phase2_outputs`` (or below ``--output-root``).

Calibration protocol
--------------------
1. Produce five-fold OOF predictions on PHASE-1 KEEP rows with
   StratifiedGroupKFold.  Both the MLP and kNN are fitted only on the training
   part of each fold.
2. Rotate the five OOF folds: use four to choose a precision-first policy and
   the untouched fifth fold to evaluate it.
3. Enable a label only when every calibration rotation found an eligible
   policy and the pooled, held-out Wilson 95% lower bound is at least 0.99.
4. Only after validation, fit on all KEEP rows and apply the validated policy
   to REVIEW.  REVIEW is never used to choose k or any threshold.

The MLP input is exactly f1..f63.  Handedness is carried only as audit
metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


FEATURE_COLUMNS = [f"f{i}" for i in range(1, 64)]
META_COLUMNS = ["source", "file", "label", "handedness"]
QUALITY_FLAG_COLUMNS = ["bad_geometry", "bad_stability", "bad_image_quality"]
N_SPLITS = 5
WILSON_Z_95 = 1.959963984540054
REQUIRED_WILSON_LOWER = 0.99

# These are floors, not targets to relax.  Search may only make them stricter.
MLP_CONFIDENCE_CANDIDATES = (0.99, 0.995, 0.999)
MLP_MARGIN_CANDIDATES = (0.98, 0.99, 0.995)
KNN_VOTE_CANDIDATES = (1.0,)
# Distance candidates are label-specific calibration quantiles.  q=0.25 is
# the least strict allowed cutoff; the search is not permitted to use q>0.25.
KNN_DISTANCE_QUANTILES = (0.25, 0.10, 0.05)

# A low-quality base classifier invalidates automatic promotion globally.
MIN_MLP_OOF_ACCURACY = 0.90
MIN_MLP_OOF_MACRO_F1 = 0.90
MIN_KNN_OOF_ACCURACY = 0.85

_TERMINAL_VARIANT_RE = re.compile(
    r"(?:\s*\([2-9]\)|\s*-\s*copy)\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Policy:
    label: str
    mlp_confidence_min: float
    mlp_margin_min: float
    knn_vote_min: float
    knn_mean_distance_max: float
    distance_quantile: float
    selected: int
    correct: int
    observed_precision: float
    wilson_lower_bound: float


def normalize_filename_stem(filename: str) -> str:
    """Normalize a stem without conflating unrelated basenames.

    Only recognized *terminal* variant suffixes are removed.  Repeating the
    substitution handles names such as ``sample (2) - Copy.jpg`` while every
    removal is still anchored to the end of the stem.  ``(1)``, ``(10)``, and
    occurrences in the middle remain untouched.
    """

    stem = PureWindowsPath(str(filename).replace("/", "\\")).stem
    normalized = unicodedata.normalize("NFC", stem).strip()
    while True:
        stripped = _TERMINAL_VARIANT_RE.sub("", normalized).strip()
        if stripped == normalized:
            break
        normalized = stripped
    return normalized.casefold()


def _normalize_namespace(value: str) -> str:
    normalized = unicodedata.normalize("NFC", str(value)).strip()
    return normalized.replace("/", "\\").casefold()


def make_group_key(label: str, source: str, filename: str) -> str:
    """Return a collision-safe label + normalized-stem group key.

    Source and parent path are namespaces only.  They prevent independent
    files with the same basename in different locations from being merged.
    """

    windows_path = PureWindowsPath(str(filename).replace("/", "\\"))
    parts = (
        unicodedata.normalize("NFC", str(label)).strip().casefold(),
        _normalize_namespace(source),
        _normalize_namespace(str(windows_path.parent)),
        normalize_filename_stem(str(windows_path)),
    )
    return "\x1f".join(parts)


def wilson_lower_bound(correct: int, selected: int, z: float = WILSON_Z_95) -> float:
    if selected <= 0:
        return 0.0
    proportion = correct / selected
    z2 = z * z
    denominator = 1.0 + z2 / selected
    centre = proportion + z2 / (2.0 * selected)
    adjustment = z * math.sqrt(
        proportion * (1.0 - proportion) / selected + z2 / (4.0 * selected * selected)
    )
    return (centre - adjustment) / denominator


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def normalize_required_bool(series: pd.Series, column: str) -> pd.Series:
    """Parse a required boolean column without Python/string truthiness.

    Missing or unexpected values are schema errors.  This is deliberately
    stricter than ``astype(bool)`` because ``bool("False")`` is ``True``.
    """

    if series.isna().any():
        raise ValueError(
            f"{column} contains {int(series.isna().sum())} missing boolean values"
        )
    if is_bool_dtype(series.dtype):
        return series.astype(bool)

    def parse(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
            return bool(int(value))
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
        raise ValueError(f"{column} contains invalid boolean value: {value!r}")

    return series.map(parse).astype(bool)


def normalize_quality_flags(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Validate/normalize required quality flags and return pre-decision counts."""

    missing = [column for column in QUALITY_FLAG_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing required quality flags: {missing}")
    normalized = frame.copy()
    counts: dict[str, dict[str, int]] = {}
    for column in QUALITY_FLAG_COLUMNS:
        raw = normalized[column]
        counts[column] = {
            "true": int(raw.map(lambda value: value is True or isinstance(value, np.bool_) and bool(value)).sum()),
            "false": int(raw.map(lambda value: value is False or isinstance(value, np.bool_) and not bool(value)).sum()),
            "nan": int(raw.isna().sum()),
        }
        normalized[column] = normalize_required_bool(raw, column)
        counts[column]["true"] = int(normalized[column].sum())
        counts[column]["false"] = int((~normalized[column]).sum())
    return normalized, counts


def apply_quality_gate_to_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Apply the minimal safe promotion rule to already-scored REVIEW rows."""

    required = ["validated_policy_pass", "model_label_agreement", *QUALITY_FLAG_COLUMNS]
    missing = [column for column in required if column not in decisions]
    if missing:
        raise ValueError(f"Decision table is missing required columns: {missing}")
    result, _ = normalize_quality_flags(decisions)
    result["validated_policy_pass"] = normalize_required_bool(
        result["validated_policy_pass"], "validated_policy_pass"
    )
    result["model_label_agreement"] = normalize_required_bool(
        result["model_label_agreement"], "model_label_agreement"
    )
    result["quality_gate_pass"] = ~result[QUALITY_FLAG_COLUMNS].any(axis=1)
    result["promotion_decision"] = np.where(
        result["quality_gate_pass"]
        & result["validated_policy_pass"]
        & result["model_label_agreement"],
        "PROMOTED",
        "KEEP_REVIEW",
    )
    return result


def load_phase1_data(dataset_path: Path, quality_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(dataset_path, usecols=META_COLUMNS + FEATURE_COLUMNS)
    quality = pd.read_csv(
        quality_path,
        usecols=META_COLUMNS
        + ["final_status", "extraction_rejected"]
        + QUALITY_FLAG_COLUMNS,
    )

    if features["file"].duplicated().any():
        raise ValueError("dataset_chucai_all contains duplicate file keys")
    if quality["file"].duplicated().any():
        raise ValueError("landmark_quality_report contains duplicate file keys")

    quality["extraction_rejected"] = normalize_required_bool(
        quality["extraction_rejected"], "extraction_rejected"
    )

    valid_quality = quality.loc[
        ~quality["extraction_rejected"]
    ].copy()
    valid_quality, quality_counts = normalize_quality_flags(valid_quality)
    print("[QUALITY FLAGS before decision] " + json.dumps(quality_counts, sort_keys=True))
    merged = features.merge(
        valid_quality,
        on="file",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_quality"),
    )
    if not (merged["_merge"] == "both").all():
        missing = int((merged["_merge"] != "both").sum())
        raise ValueError(f"{missing} feature rows have no PHASE-1 quality status")
    for column in ["source", "label", "handedness"]:
        quality_column = f"{column}_quality"
        mismatch = merged[column].astype(str) != merged[quality_column].astype(str)
        if mismatch.any():
            raise ValueError(
                f"{int(mismatch.sum())} rows disagree on {column} after file join"
            )
    merged = merged.drop(
        columns=["_merge", "source_quality", "label_quality", "handedness_quality"]
    )

    keep = merged.loc[merged["final_status"] == "KEEP"].copy().reset_index(drop=True)
    review = merged.loc[merged["final_status"] == "REVIEW"].copy().reset_index(drop=True)
    if keep.empty or review.empty:
        raise ValueError("Both KEEP and REVIEW must be non-empty")
    if keep[FEATURE_COLUMNS].isna().any().any() or review[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("NaN found in f1..f63")

    for frame in (keep, review):
        frame["group_key"] = [
            make_group_key(label, source, filename)
            for label, source, filename in zip(
                frame["label"], frame["source"], frame["file"], strict=True
            )
        ]
    return keep, review


def build_splits(keep: pd.DataFrame, seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    dummy_x = np.zeros((len(keep), 1), dtype=np.uint8)
    y = keep["label"].to_numpy()
    groups = keep["group_key"].to_numpy()
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    audits: list[dict] = []
    validation_hits = np.zeros(len(keep), dtype=np.int8)

    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(dummy_x, y, groups), start=0
    ):
        train_groups = set(groups[train_idx])
        validation_groups = set(groups[validation_idx])
        overlap = train_groups.intersection(validation_groups)
        if overlap:
            raise RuntimeError(
                f"Fold {fold}: train_groups intersection validation_groups is not empty"
            )
        validation_hits[validation_idx] += 1
        splits.append((train_idx, validation_idx))
        audits.append(
            {
                "fold": fold,
                "train_samples": len(train_idx),
                "validation_samples": len(validation_idx),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_intersection_count": 0,
                "group_intersection_empty": True,
                "train_label_counts": json.dumps(
                    pd.Series(y[train_idx]).value_counts().sort_index().to_dict(),
                    ensure_ascii=False,
                ),
                "validation_label_counts": json.dumps(
                    pd.Series(y[validation_idx]).value_counts().sort_index().to_dict(),
                    ensure_ascii=False,
                ),
            }
        )

    if not np.all(validation_hits == 1):
        raise RuntimeError("Each KEEP row must occur in exactly one OOF validation fold")
    return splits, pd.DataFrame(audits)


def _fit_mlp(x_train: np.ndarray, y_train: np.ndarray, seed: int, max_iter: int):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=15,
        random_state=seed,
    )
    mlp.fit(scaled_train, y_train)
    return scaler, mlp, scaled_train


def _predict_mlp(scaler, mlp, x: np.ndarray) -> dict[str, np.ndarray]:
    probabilities = mlp.predict_proba(scaler.transform(x))
    order = np.argsort(probabilities, axis=1)
    top = order[:, -1]
    second = order[:, -2]
    rows = np.arange(len(x))
    return {
        "mlp_pred": mlp.classes_[top],
        "mlp_confidence": probabilities[rows, top],
        "mlp_margin": probabilities[rows, top] - probabilities[rows, second],
    }


def _fit_knn(scaled_train: np.ndarray, y_train: np.ndarray, k: int):
    if k < 1 or k % 2 == 0:
        raise ValueError("--knn-k must be a positive odd integer")
    if k > len(scaled_train):
        raise ValueError("--knn-k exceeds the training fold size")
    encoder = LabelEncoder().fit(y_train)
    encoded_y = encoder.transform(y_train)
    neighbors = NearestNeighbors(
        n_neighbors=k,
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    )
    neighbors.fit(scaled_train)
    return neighbors, encoder, encoded_y


def _predict_knn(
    scaler,
    neighbors: NearestNeighbors,
    encoder: LabelEncoder,
    encoded_y: np.ndarray,
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    distances, indices = neighbors.kneighbors(scaler.transform(x), return_distance=True)
    neighbor_labels = encoded_y[indices]
    counts = np.stack(
        [(neighbor_labels == class_id).sum(axis=1) for class_id in range(len(encoder.classes_))],
        axis=1,
    )
    predicted_encoded = counts.argmax(axis=1)
    rows = np.arange(len(x))
    return {
        "knn_pred": encoder.inverse_transform(predicted_encoded),
        "knn_vote": counts[rows, predicted_encoded] / neighbor_labels.shape[1],
        "knn_nearest_distance": distances[:, 0],
        "knn_mean_distance": distances.mean(axis=1),
    }


def produce_oof_predictions(
    keep: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    max_iter: int,
    knn_k: int,
) -> pd.DataFrame:
    x_all = keep[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    y_all = keep["label"].astype(str).to_numpy()
    pieces: list[pd.DataFrame] = []

    for fold, (train_idx, validation_idx) in enumerate(splits):
        print(f"[OOF] fold {fold + 1}/{N_SPLITS}: train={len(train_idx)}, validation={len(validation_idx)}")
        scaler, mlp, scaled_train = _fit_mlp(
            x_all[train_idx], y_all[train_idx], seed + fold, max_iter
        )
        mlp_output = _predict_mlp(scaler, mlp, x_all[validation_idx])

        neighbors, encoder, encoded_y = _fit_knn(
            scaled_train, y_all[train_idx], knn_k
        )
        knn_output = _predict_knn(
            scaler, neighbors, encoder, encoded_y, x_all[validation_idx]
        )

        piece = keep.loc[
            validation_idx, META_COLUMNS + ["group_key"]
        ].copy()
        piece["row_index"] = validation_idx
        piece["oof_fold"] = fold
        for name, values in {**mlp_output, **knn_output}.items():
            piece[name] = values
        piece["model_consensus"] = piece["mlp_pred"] == piece["knn_pred"]
        piece["consensus_pred"] = np.where(
            piece["model_consensus"], piece["mlp_pred"], pd.NA
        )
        pieces.append(piece)

        # Release the fold's large estimators before the next fit.
        del scaler, mlp, scaled_train, neighbors, encoder, encoded_y

    oof = pd.concat(pieces, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    if len(oof) != len(keep) or oof["row_index"].nunique() != len(keep):
        raise RuntimeError("OOF prediction coverage is not exactly one row per KEEP sample")
    return oof


def policy_mask(frame: pd.DataFrame, policy: Policy) -> pd.Series:
    return (
        frame["model_consensus"].astype(bool)
        & (frame["consensus_pred"] == policy.label)
        & (frame["mlp_confidence"] >= policy.mlp_confidence_min)
        & (frame["mlp_margin"] >= policy.mlp_margin_min)
        & (frame["knn_vote"] >= policy.knn_vote_min)
        & (frame["knn_mean_distance"] <= policy.knn_mean_distance_max)
    )


def choose_policy(calibration: pd.DataFrame, label: str) -> Policy | None:
    base = calibration.loc[
        calibration["model_consensus"].astype(bool)
        & (calibration["consensus_pred"] == label)
    ]
    if base.empty:
        return None

    distance_cutoffs = {
        quantile: float(base["knn_mean_distance"].quantile(quantile))
        for quantile in KNN_DISTANCE_QUANTILES
    }
    eligible: list[Policy] = []
    for confidence in MLP_CONFIDENCE_CANDIDATES:
        for margin in MLP_MARGIN_CANDIDATES:
            for vote in KNN_VOTE_CANDIDATES:
                for distance_quantile, distance_cutoff in distance_cutoffs.items():
                    candidate = Policy(
                        label=label,
                        mlp_confidence_min=confidence,
                        mlp_margin_min=margin,
                        knn_vote_min=vote,
                        knn_mean_distance_max=distance_cutoff,
                        distance_quantile=distance_quantile,
                        selected=0,
                        correct=0,
                        observed_precision=0.0,
                        wilson_lower_bound=0.0,
                    )
                    selected_mask = policy_mask(calibration, candidate)
                    selected = int(selected_mask.sum())
                    correct = int((calibration.loc[selected_mask, "label"] == label).sum())
                    precision = correct / selected if selected else 0.0
                    lower = wilson_lower_bound(correct, selected)
                    if lower >= REQUIRED_WILSON_LOWER:
                        eligible.append(
                            Policy(
                                label=label,
                                mlp_confidence_min=confidence,
                                mlp_margin_min=margin,
                                knn_vote_min=vote,
                                knn_mean_distance_max=distance_cutoff,
                                distance_quantile=distance_quantile,
                                selected=selected,
                                correct=correct,
                                observed_precision=precision,
                                wilson_lower_bound=lower,
                            )
                        )

    if not eligible:
        # Deliberately do not lower a floor to manufacture statistical support.
        return None

    # Evidence and precision come before coverage.  Remaining keys prefer the
    # stricter policy; selected count is not an optimization objective.
    return max(
        eligible,
        key=lambda p: (
            p.wilson_lower_bound,
            p.observed_precision,
            p.mlp_confidence_min,
            p.mlp_margin_min,
            p.knn_vote_min,
            -p.knn_mean_distance_max,
        ),
    )


def crossfit_validate_policy(
    oof: pd.DataFrame, labels: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    rotation_rows: list[dict] = []
    evaluation_rows: list[pd.DataFrame] = []
    calibration_passes: dict[str, int] = {label: 0 for label in labels}

    for evaluation_fold in range(N_SPLITS):
        calibration = oof.loc[oof["oof_fold"] != evaluation_fold]
        evaluation = oof.loc[oof["oof_fold"] == evaluation_fold]
        for label in labels:
            policy = choose_policy(calibration, label)
            if policy is None:
                rotation_rows.append(
                    {
                        "evaluation_fold": evaluation_fold,
                        "label": label,
                        "calibration_policy_found": False,
                        "calibration_selected": 0,
                        "calibration_precision": None,
                        "calibration_wilson_lower": 0.0,
                        "evaluation_selected": 0,
                        "evaluation_correct": 0,
                        "evaluation_precision": None,
                        "evaluation_wilson_lower": 0.0,
                    }
                )
                continue

            calibration_passes[label] += 1
            selected_mask = policy_mask(evaluation, policy)
            selected = evaluation.loc[selected_mask].copy()
            selected["policy_label"] = label
            selected["policy_evaluation_fold"] = evaluation_fold
            selected["policy_correct"] = selected["label"] == label
            evaluation_rows.append(selected)
            correct = int(selected["policy_correct"].sum())
            count = len(selected)
            rotation_rows.append(
                {
                    "evaluation_fold": evaluation_fold,
                    "label": label,
                    "calibration_policy_found": True,
                    "calibration_selected": policy.selected,
                    "calibration_precision": policy.observed_precision,
                    "calibration_wilson_lower": policy.wilson_lower_bound,
                    "evaluation_selected": count,
                    "evaluation_correct": correct,
                    "evaluation_precision": correct / count if count else None,
                    "evaluation_wilson_lower": wilson_lower_bound(correct, count),
                    "mlp_confidence_min": policy.mlp_confidence_min,
                    "mlp_margin_min": policy.mlp_margin_min,
                    "knn_vote_min": policy.knn_vote_min,
                    "knn_mean_distance_max": policy.knn_mean_distance_max,
                    "distance_quantile": policy.distance_quantile,
                }
            )

    rotations = pd.DataFrame(rotation_rows)
    evaluated = (
        pd.concat(evaluation_rows, ignore_index=True)
        if evaluation_rows
        else pd.DataFrame(columns=list(oof.columns) + ["policy_label", "policy_correct"])
    )

    summaries: list[dict] = []
    enabled: dict[str, bool] = {}
    for label in labels:
        rows = evaluated.loc[evaluated.get("policy_label") == label]
        selected = len(rows)
        correct = int(rows["policy_correct"].sum()) if selected else 0
        precision = correct / selected if selected else None
        lower = wilson_lower_bound(correct, selected)
        all_rotations_calibrated = calibration_passes[label] == N_SPLITS
        is_enabled = all_rotations_calibrated and lower >= REQUIRED_WILSON_LOWER
        enabled[label] = is_enabled
        summaries.append(
            {
                "label": label,
                "number_selected_during_policy_validation": selected,
                "correct_during_policy_validation": correct,
                "observed_selective_precision": precision,
                "wilson_95_lower_bound": lower,
                "calibration_rotations_passed": calibration_passes[label],
                "auto_promotion_enabled": is_enabled,
                "policy_status": (
                    "ENABLED"
                    if is_enabled
                    else "AUTO_PROMOTION_DISABLED_FOR_CLASS"
                ),
            }
        )
    return rotations, pd.DataFrame(summaries), enabled


def oof_quality(oof: pd.DataFrame) -> dict:
    mlp_accuracy = accuracy_score(oof["label"], oof["mlp_pred"])
    mlp_macro_f1 = f1_score(oof["label"], oof["mlp_pred"], average="macro")
    knn_accuracy = accuracy_score(oof["label"], oof["knn_pred"])
    safe = (
        mlp_accuracy >= MIN_MLP_OOF_ACCURACY
        and mlp_macro_f1 >= MIN_MLP_OOF_MACRO_F1
        and knn_accuracy >= MIN_KNN_OOF_ACCURACY
    )
    return {
        "mlp_oof_accuracy": mlp_accuracy,
        "mlp_oof_macro_f1": mlp_macro_f1,
        "knn_oof_accuracy": knn_accuracy,
        "minimum_mlp_oof_accuracy": MIN_MLP_OOF_ACCURACY,
        "minimum_mlp_oof_macro_f1": MIN_MLP_OOF_MACRO_F1,
        "minimum_knn_oof_accuracy": MIN_KNN_OOF_ACCURACY,
        "quality_gate_passed": safe,
    }


def fit_final_policies(
    oof: pd.DataFrame, labels: Iterable[str], enabled: dict[str, bool]
) -> tuple[dict[str, Policy], pd.DataFrame]:
    policies: dict[str, Policy] = {}
    rows: list[dict] = []
    for label in labels:
        policy = choose_policy(oof, label) if enabled[label] else None
        if policy is None:
            rows.append(
                {
                    "label": label,
                    "final_policy_fitted": False,
                    "policy_status": "AUTO_PROMOTION_DISABLED_FOR_CLASS",
                }
            )
            continue
        policies[label] = policy
        rows.append(
            {
                **asdict(policy),
                "final_policy_fitted": True,
                "policy_status": "ENABLED",
            }
        )
    return policies, pd.DataFrame(rows)


def fit_and_score_review(
    keep: pd.DataFrame,
    review: pd.DataFrame,
    policies: dict[str, Policy],
    seed: int,
    max_iter: int,
    knn_k: int,
) -> tuple[pd.DataFrame, dict]:
    x_keep = keep[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    y_keep = keep["label"].astype(str).to_numpy()
    x_review = review[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    scaler, mlp, scaled_keep = _fit_mlp(x_keep, y_keep, seed + 10_000, max_iter)
    mlp_output = _predict_mlp(scaler, mlp, x_review)
    neighbors, encoder, encoded_y = _fit_knn(scaled_keep, y_keep, knn_k)
    knn_output = _predict_knn(scaler, neighbors, encoder, encoded_y, x_review)

    decisions = review[META_COLUMNS + ["group_key"] + QUALITY_FLAG_COLUMNS].copy()
    for name, values in {**mlp_output, **knn_output}.items():
        decisions[name] = values
    decisions["model_consensus"] = decisions["mlp_pred"] == decisions["knn_pred"]
    decisions["consensus_pred"] = np.where(
        decisions["model_consensus"], decisions["mlp_pred"], pd.NA
    )
    decisions["validated_policy_pass"] = False
    for label, policy in policies.items():
        decisions.loc[policy_mask(decisions, policy), "validated_policy_pass"] = True
    decisions["model_label_agreement"] = decisions["consensus_pred"] == decisions["label"]
    decisions = apply_quality_gate_to_decisions(decisions)
    decisions["model_disagree_high_risk"] = (
        decisions["validated_policy_pass"] & ~decisions["model_label_agreement"]
    )
    decisions["audit_status"] = np.select(
        [
            decisions["promotion_decision"] == "PROMOTED",
            decisions["model_disagree_high_risk"],
        ],
        ["PROMOTED", "MODEL_DISAGREE_HIGH_RISK"],
        default="KEEP_REVIEW",
    )

    artifact = {
        "scaler": scaler,
        "mlp": mlp,
        "knn": neighbors,
        "knn_label_encoder": encoder,
        "knn_encoded_training_labels": encoded_y,
        "feature_columns": FEATURE_COLUMNS,
        "knn_k": knn_k,
        "policies": {label: asdict(policy) for label, policy in policies.items()},
    }
    return decisions, artifact


def build_per_label_report(
    keep: pd.DataFrame,
    review: pd.DataFrame,
    validation: pd.DataFrame,
    decisions: pd.DataFrame | None,
    globally_safe: bool,
    enabled: dict[str, bool],
) -> pd.DataFrame:
    rows: list[dict] = []
    labels = sorted(keep["label"].astype(str).unique())
    validation_by_label = validation.set_index("label")
    for label in labels:
        metric = validation_by_label.loc[label]
        review_count = int((review["label"] == label).sum())
        if decisions is None:
            promoted = 0
            high_risk = 0
        else:
            label_decisions = decisions.loc[decisions["label"] == label]
            promoted = int((label_decisions["promotion_decision"] == "PROMOTED").sum())
            high_risk = int(label_decisions["model_disagree_high_risk"].sum())
        label_enabled = bool(enabled[label]) and globally_safe
        rows.append(
            {
                "label": label,
                "KEEP_count": int((keep["label"] == label).sum()),
                "REVIEW_count": review_count,
                "number_selected_during_policy_validation": int(
                    metric["number_selected_during_policy_validation"]
                ),
                "observed_selective_precision": metric["observed_selective_precision"],
                "wilson_95_lower_bound": metric["wilson_95_lower_bound"],
                "auto_promotion_enabled": label_enabled,
                "policy_status": (
                    "ENABLED"
                    if label_enabled
                    else "AUTO_PROMOTION_DISABLED_FOR_CLASS"
                ),
                "PROMOTED_count": promoted,
                "MODEL_DISAGREE_HIGH_RISK_count": high_risk,
                "REVIEW_remaining": review_count - promoted,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    ai_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ai_dir / "dataset" / "dataset_chucai_all.csv",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=ai_dir / "dataset" / "landmark_quality_report.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ai_dir / "phase2_outputs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--knn-k", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    print("Loading PHASE-1 KEEP and REVIEW rows...")
    keep, review = load_phase1_data(args.dataset, args.quality_report)
    labels = sorted(keep["label"].astype(str).unique())
    print(f"KEEP={len(keep):,}; REVIEW={len(review):,}; labels={len(labels)}")

    splits, fold_audit = build_splits(keep, args.seed)
    for row in fold_audit.itertuples(index=False):
        print(
            f"[GROUP AUDIT] fold={row.fold}: "
            "train_groups intersection validation_groups = empty"
        )
    fold_audit.to_csv(output_dir / "fold_group_audit.csv", index=False)

    oof = produce_oof_predictions(keep, splits, args.seed, args.max_iter, args.knn_k)
    oof.to_csv(output_dir / "oof_keep_predictions.csv", index=False)
    quality = oof_quality(oof)
    rotations, validation, enabled = crossfit_validate_policy(oof, labels)
    rotations.to_csv(output_dir / "crossfit_rotation_metrics.csv", index=False)
    validation.to_csv(output_dir / "crossfit_policy_validation.csv", index=False)

    globally_safe = bool(quality["quality_gate_passed"])
    if not any(enabled.values()):
        globally_safe = False

    decisions: pd.DataFrame | None = None
    final_policies: dict[str, Policy] = {}
    final_policy_table = pd.DataFrame()
    if globally_safe:
        final_policies, final_policy_table = fit_final_policies(oof, labels, enabled)
        # A class can still be disabled if fitting the final all-OOF threshold
        # cannot prove the same Wilson policy.
        for label in labels:
            enabled[label] = label in final_policies
        if not final_policies:
            globally_safe = False

    final_policy_table.to_csv(output_dir / "final_policy_thresholds.csv", index=False)

    if globally_safe:
        print("Cross-fit policy passed. Fitting final KEEP-only models for REVIEW...")
        decisions, artifact = fit_and_score_review(
            keep, review, final_policies, args.seed, args.max_iter, args.knn_k
        )
        decisions.to_csv(output_dir / "review_promotion_decisions.csv", index=False)
        joblib.dump(artifact, output_dir / "review_promotion_model.joblib")

        promoted_keys = decisions["promotion_decision"] == "PROMOTED"
        promoted = review.loc[promoted_keys.to_numpy(), META_COLUMNS + FEATURE_COLUMNS].copy()
        promoted.insert(4, "phase2_status", "PROMOTED")
        promoted.to_csv(output_dir / "promoted_review_features.csv", index=False)
        run_status = "POLICY_VALIDATED"
    else:
        run_status = "AUTO_REVIEW_NOT_SAFE"
        # Preserve an auditable, header-only decision file while making it
        # impossible to mistake this run for a promotion run.
        pd.DataFrame(
            columns=META_COLUMNS
            + [
                "promotion_decision",
                "model_disagree_high_risk",
                "audit_status",
            ]
        ).to_csv(output_dir / "review_promotion_decisions.csv", index=False)
        pd.DataFrame(columns=META_COLUMNS + ["phase2_status"] + FEATURE_COLUMNS).to_csv(
            output_dir / "promoted_review_features.csv", index=False
        )

    per_label = build_per_label_report(
        keep, review, validation, decisions, globally_safe, enabled
    )
    per_label.to_csv(output_dir / "per_label_report.csv", index=False)

    summary = {
        "phase": 2,
        "run_status": run_status,
        "precision_first_policy": {
            "wilson_confidence": 0.95,
            "required_lower_bound": REQUIRED_WILSON_LOWER,
            "threshold_floors": {
                "mlp_confidence": min(MLP_CONFIDENCE_CANDIDATES),
                "mlp_margin": min(MLP_MARGIN_CANDIDATES),
                "knn_vote": min(KNN_VOTE_CANDIDATES),
                "knn_distance_max_quantile": max(KNN_DISTANCE_QUANTILES),
            },
            "thresholds_never_relaxed_for_support": True,
        },
        "crossfit": {
            "folds": N_SPLITS,
            "calibration_folds_per_rotation": 4,
            "evaluation_folds_per_rotation": 1,
            "rotations": 5,
            "group_splitter": "StratifiedGroupKFold",
            "all_fold_group_intersections_empty": bool(
                fold_audit["group_intersection_empty"].all()
            ),
        },
        "knn": {
            "k": args.knn_k,
            "k_selected_using_review": False,
            "oof_fit_scope": "train_fold_only",
            "review_fit_scope": "all_KEEP_after_calibration",
        },
        "mlp_features": FEATURE_COLUMNS,
        "handedness_usage": "metadata/audit only",
        "model_disagree_high_risk_definition": (
            "both models agree on the same alternative label and that prediction "
            "passes the alternative label's validated policy; original REVIEW label "
            "is retained"
        ),
        "counts": {
            "KEEP": len(keep),
            "REVIEW": len(review),
            "PROMOTED": (
                int((decisions["promotion_decision"] == "PROMOTED").sum())
                if decisions is not None
                else 0
            ),
            "MODEL_DISAGREE_HIGH_RISK": (
                int(decisions["model_disagree_high_risk"].sum())
                if decisions is not None
                else 0
            ),
            "REVIEW_remaining": (
                int((decisions["promotion_decision"] != "PROMOTED").sum())
                if decisions is not None
                else len(review)
            ),
        },
        "oof_quality": quality,
        "enabled_labels": [label for label in labels if enabled[label] and globally_safe],
        "disabled_labels": [label for label in labels if not (enabled[label] and globally_safe)],
        "source_csvs_modified": False,
        "output_directory": str(output_dir.resolve()),
    }
    (output_dir / "phase2_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== PHASE 2 STATUS ===")
    print(run_status)
    print(pd.DataFrame(per_label).to_string(index=False))
    print(f"Outputs: {output_dir.resolve()}")
    return 0 if globally_safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
