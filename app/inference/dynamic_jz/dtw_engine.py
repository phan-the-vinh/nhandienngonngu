"""Multivariate Dynamic Time Warping (DTW) Engine.

Provides core constrained multivariate DTW with:
- Weighted local distance metrics.
- Sakoe-Chiba band constraint.
- Alignment path length normalization.
- Robust handling of variable sequence lengths, empty/invalid inputs, and non-finite values.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


class DTWError(ValueError):
    """Raised when DTW inputs violate mathematical or dimensional constraints."""


def dtw_multivariate(
    seq1: np.ndarray,
    seq2: np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
    radius: int | None = 3,
) -> tuple[float, int, list[tuple[int, int]]]:
    """Compute constrained multivariate Dynamic Time Warping distance between two sequences.

    Parameters
    ----------
    seq1 : np.ndarray of shape (N, D)
        First feature sequence.
    seq2 : np.ndarray of shape (M, D)
        Second feature sequence.
    weights : array_like of shape (D,), optional
        Positive weights applied per feature dimension. Defaults to uniform weights (all 1.0).
    radius : int, optional
        Sakoe-Chiba band half-width: |i - j| <= radius.
        If None, computes unconstrained DTW. Defaults to 3.

    Returns
    -------
    normalized_distance : float
        Cumulative warping distance divided by the optimal alignment path length.
    alignment_path_length : int
        Number of step pairs in the optimal alignment path.
    warping_path : list[tuple[int, int]]
        List of aligned index pairs [(0, 0), ..., (N-1, M-1)].
    """
    # 1. Input validations
    s1 = np.asarray(seq1, dtype=np.float64)
    s2 = np.asarray(seq2, dtype=np.float64)

    if s1.ndim != 2 or s2.ndim != 2:
        raise DTWError(
            f"Expected 2D arrays, got seq1 shape {s1.shape} and seq2 shape {s2.shape}"
        )
    N, D1 = s1.shape
    M, D2 = s2.shape

    if D1 != D2:
        raise DTWError(
            f"Feature dimension mismatch: seq1 has {D1} features, seq2 has {D2} features."
        )
    if N == 0 or M == 0:
        raise DTWError(f"Cannot align empty sequences: seq1 has length {N}, seq2 has length {M}.")
    if not (np.isfinite(s1).all() and np.isfinite(s2).all()):
        raise DTWError("Sequences contain NaN or Inf values.")

    # 2. Weight validation
    if weights is None:
        w = np.ones(D1, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (D1,):
            raise DTWError(f"Weights shape {w.shape} does not match feature dimension {D1}.")
        if np.any(w < 0.0) or not np.isfinite(w).all():
            raise DTWError("Weights must be non-negative finite values.")

    # 3. Pairwise local distance matrix (Euclidean weighted distance)
    # diff shape: (N, M, D)
    diff = s1[:, None, :] - s2[None, :, :]
    dist_matrix = np.sqrt(np.sum(w * (diff ** 2), axis=-1))

    # 4. Sakoe-Chiba constraint window setup
    use_window = radius is not None
    r = int(radius) if use_window else max(N, M)
    if r < 0:
        raise DTWError(f"Sakoe-Chiba radius must be non-negative, got {radius}.")

    # Cumulative cost matrix initialization with infinity
    cost = np.full((N, M), np.inf, dtype=np.float64)

    # Base cell (0, 0)
    cost[0, 0] = dist_matrix[0, 0]

    # Initialize first column
    for i in range(1, min(N, 1 + r if use_window else N)):
        cost[i, 0] = cost[i - 1, 0] + dist_matrix[i, 0]

    # Initialize first row
    for j in range(1, min(M, 1 + r if use_window else M)):
        cost[0, j] = cost[0, j - 1] + dist_matrix[0, j]

    # 5. Dynamic Programming DP recursion
    for i in range(1, N):
        j_start = max(1, i - r) if use_window else 1
        j_end = min(M, i + r + 1) if use_window else M
        for j in range(j_start, j_end):
            min_prev = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            cost[i, j] = dist_matrix[i, j] + min_prev

    total_cost = cost[N - 1, M - 1]
    if not np.isfinite(total_cost):
        # Window was too narrow to reach end cell
        raise DTWError(
            f"Sakoe-Chiba window R={radius} is too restrictive to connect ({N},{M})."
        )

    # 6. Backtrack optimal warping path
    i, j = N - 1, M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (cost[i - 1, j - 1], (i - 1, j - 1)),  # diagonal (match)
                (cost[i - 1, j], (i - 1, j)),          # vertical (insertion/deletion)
                (cost[i, j - 1], (i, j - 1)),          # horizontal (insertion/deletion)
            ]
            candidates.sort(key=lambda item: item[0])
            next_step = candidates[0][1]
            i, j = next_step
        path.append((i, j))

    path.reverse()
    path_len = len(path)
    normalized_distance = float(total_cost / path_len)

    return normalized_distance, path_len, path
