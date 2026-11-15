"""Template Store and k-Medoids Selection for Dynamic J/Z Gestures.

Provides:
- DTWTemplate dataclass with full provenance metadata.
- TemplateStore container with JSON serialization.
- Strict Train-only k-Medoids template selection under VALID-only and VALID+SUSPICIOUS policies.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from app.inference.dynamic_jz.dtw_engine import dtw_multivariate
from app.inference.dynamic_jz.feature_builder import build_dtw_features
from app.inference.dynamic_jz.finger_tracker import FingerTrackingResult, FEATURE_VERSION


@dataclass(frozen=True)
class DTWTemplate:
    """A representative gesture template with strict provenance tracking."""
    template_id: str
    label: str                               # 'J' or 'Z'
    source_sample_id: str                    # e.g., JSONL source_path
    source_video_id: str                     # e.g., '1.avi'
    source_split: str                        # Strictly 'train'
    tracking_quality: str                    # 'VALID' or 'SUSPICIOUS'
    feature_schema_version: str              # 'finger_tracking_v1'
    feature_config: str                      # 'CONFIG_A', etc.
    selection_method: str                    # 'k-medoids-dtw'
    creation_timestamp: str                  # ISO 8601 UTC
    feature_sequence: list[list[float]]      # (T, D) float sequence

    def get_array(self) -> np.ndarray:
        return np.asarray(self.feature_sequence, dtype=np.float64)


class TemplateStore:
    """Manages and serializes collections of DTW templates."""

    def __init__(self, templates: list[DTWTemplate] | None = None) -> None:
        self.templates: list[DTWTemplate] = list(templates) if templates else []

    def add(self, template: DTWTemplate) -> None:
        self.templates.append(template)

    def get_by_label(self, label: str) -> list[DTWTemplate]:
        lbl = str(label).strip().upper()
        return [t for t in self.templates if t.label == lbl]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "dtw_template_store_v1",
            "template_count": len(self.templates),
            "templates": [asdict(t) for t in self.templates],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateStore:
        templates = [DTWTemplate(**item) for item in data.get("templates", [])]
        return cls(templates)

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path | str) -> TemplateStore:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def select_k_medoids_for_class(
    samples: list[tuple[str, FingerTrackingResult]],
    label: str,
    k: int,
    config_name: str = "CONFIG_A",
    policy: str = "VALID_ONLY",
    radius: int = 3,
    max_iter: int = 100,
    random_state: int = 42,
) -> list[DTWTemplate]:
    """Select K medoid templates for one class from Train split using k-Medoids on DTW distance.

    Parameters
    ----------
    samples : list of (source_path, FingerTrackingResult)
        Candidates from Train split only.
    label : str
        'J' or 'Z'.
    k : int
        Number of medoids to select (1, 2, 4, 8).
    config_name : str
        Feature configuration name.
    policy : str
        'VALID_ONLY' (filter to validity_status == 'VALID') or
        'VALID_AND_SUSPICIOUS' (allow both VALID and SUSPICIOUS).
    radius : int
        Sakoe-Chiba window constraint for DTW distance computation.

    Returns
    -------
    list[DTWTemplate]
        The K chosen representative medoid templates with full provenance metadata.
    """
    lbl = str(label).strip().upper()
    pol = str(policy).strip().upper()

    # 1. Filter candidates by class and policy
    candidates = []
    for src_path, res in samples:
        if res.letter_target != lbl:
            continue
        if pol == "VALID_ONLY":
            if res.validity_status == "VALID":
                candidates.append((src_path, res))
        elif pol == "VALID_AND_SUSPICIOUS":
            if res.validity_status in ("VALID", "SUSPICIOUS"):
                candidates.append((src_path, res))
        else:
            raise ValueError(f"Unknown policy '{policy}'. Must be 'VALID_ONLY' or 'VALID_AND_SUSPICIOUS'.")

    N = len(candidates)
    if N == 0:
        raise ValueError(f"No candidates found for label '{label}' under policy '{policy}'.")
    if k > N:
        raise ValueError(f"Cannot select K={k} medoids from only {N} candidates.")

    # 2. Extract feature sequences for all candidates
    seqs = []
    for _, res in candidates:
        feat_seq, weights = build_dtw_features(res, config_name=config_name)
        seqs.append(feat_seq)

    # 3. Compute pairwise DTW distance matrix (symmetric, N x N)
    dist_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d, _, _ = dtw_multivariate(seqs[i], seqs[j], weights=weights, radius=radius)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # 4. k-Medoids algorithm
    rng = np.random.RandomState(random_state)
    if k == 1:
        # For K=1: The medoid is simply the single candidate that minimizes total distance to all others
        total_costs = np.sum(dist_matrix, axis=1)
        best_medoid_indices = [int(np.argmin(total_costs))]
    else:
        # For K > 1: Initialize with k-medoids++ (greedy distance maximization)
        medoids = [rng.randint(0, N)]
        for _ in range(1, k):
            min_dist_to_medoids = np.min(dist_matrix[:, medoids], axis=1)
            # Pick candidate with probability proportional to squared distance
            probs = min_dist_to_medoids ** 2
            prob_sum = np.sum(probs)
            if prob_sum > 1e-12:
                probs /= prob_sum
                next_medoid = rng.choice(N, p=probs)
            else:
                next_medoid = rng.randint(0, N)
            medoids.append(next_medoid)

        medoids = sorted(set(medoids))
        while len(medoids) < k:
            extra = rng.randint(0, N)
            if extra not in medoids:
                medoids.append(extra)

        # Iterative medoid refinement (PAM swap step)
        medoid_arr = np.array(medoids, dtype=int)
        for _ in range(max_iter):
            # Assign each point to closest medoid
            assignments = np.argmin(dist_matrix[:, medoid_arr], axis=1)
            improved = False
            for cluster_idx in range(k):
                cluster_members = np.flatnonzero(assignments == cluster_idx)
                if len(cluster_members) == 0:
                    continue
                # Find new medoid in cluster that minimizes sum of distances to cluster members
                sub_dists = dist_matrix[np.ix_(cluster_members, cluster_members)]
                best_sub_idx = np.argmin(np.sum(sub_dists, axis=1))
                new_medoid = cluster_members[best_sub_idx]
                if new_medoid != medoid_arr[cluster_idx]:
                    medoid_arr[cluster_idx] = new_medoid
                    improved = True
            if not improved:
                break

        best_medoid_indices = sorted(medoid_arr.tolist())

    # 5. Build DTWTemplate objects
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    selected_templates = []
    for rank, idx in enumerate(best_medoid_indices, 1):
        src_path, res = candidates[idx]
        video_id = Path(src_path).name
        template_id = f"{lbl}_{pol.lower()}_k{k}_{rank:02d}_{video_id.replace('.', '_')}"
        t = DTWTemplate(
            template_id=template_id,
            label=lbl,
            source_sample_id=src_path,
            source_video_id=video_id,
            source_split="train",
            tracking_quality=res.validity_status,
            feature_schema_version=FEATURE_VERSION,
            feature_config=config_name,
            selection_method=f"k-medoids-dtw-R{radius}",
            creation_timestamp=now_iso,
            feature_sequence=seqs[idx].tolist(),
        )
        selected_templates.append(t)

    return selected_templates
