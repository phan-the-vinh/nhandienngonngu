"""Experimental in-plane hand alignment for existing wrist-normalized 63 features.

This is a separate sklearn transformer. It does not change the canonical
normalize_landmarks() function or any existing production model.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HandAlignmentError(ValueError):
    """Input cannot be safely aligned by the experimental transformer."""


class DegenerateHandAxisError(HandAlignmentError):
    """The palm axes cannot be determined reliably for one or more rows."""


class HandCentric2DAligner(TransformerMixin, BaseEstimator):
    """Apply a proper 2D rotation to 21 wrist-relative landmarks; preserve z.

    The palm-longitudinal axis points from landmark 0 to the mean of 5, 9, 17.
    The 17-to-5 vector supplies the transverse-axis degeneracy check. The
    transverse basis is sign-corrected to have determinant +1, so this never
    reflects or mirrors a hand to force its index side to a chosen direction.
    """

    def __init__(self, axis_epsilon=1e-6):
        self.axis_epsilon = axis_epsilon

    def _points(self, X):
        if not np.isfinite(self.axis_epsilon) or self.axis_epsilon <= 0:
            raise ValueError("axis_epsilon must be finite and positive")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 63 or values.shape[0] == 0:
            raise HandAlignmentError("Expected a nonempty matrix of 63 features per row")
        if not np.isfinite(values).all():
            raise HandAlignmentError("Hand features contain NaN or Inf")
        return values.reshape(-1, 21, 3)

    def fit(self, X, y=None):
        self._points(X)
        self.n_features_in_ = 63
        return self

    def transform(self, X):
        points = self._points(X)
        relative = points - points[:, 0:1, :]
        xy = relative[:, :, :2]

        vertical_raw = xy[:, [5, 9, 17], :].mean(axis=1)
        vertical_norm = np.linalg.norm(vertical_raw, axis=1)
        if np.any(vertical_norm <= self.axis_epsilon):
            raise DegenerateHandAxisError("Palm longitudinal axis is degenerate")
        vertical = vertical_raw / vertical_norm[:, None]

        transverse_raw = xy[:, 5, :] - xy[:, 17, :]
        transverse_orthogonal = transverse_raw - (
            np.sum(transverse_raw * vertical, axis=1)[:, None] * vertical
        )
        transverse_norm = np.linalg.norm(transverse_orthogonal, axis=1)
        if np.any(transverse_norm <= self.axis_epsilon):
            raise DegenerateHandAxisError("Palm transverse axis is degenerate")
        horizontal = transverse_orthogonal / transverse_norm[:, None]

        # A pure in-plane rotation must have determinant +1. Do not reflect
        # left/right hands merely to force landmark 5 onto the same side.
        determinant = horizontal[:, 0] * vertical[:, 1] - horizontal[:, 1] * vertical[:, 0]
        horizontal = np.where(determinant[:, None] < 0, -horizontal, horizontal)

        aligned = np.empty_like(relative)
        aligned[:, :, 0] = np.sum(xy * horizontal[:, None, :], axis=2)
        aligned[:, :, 1] = np.sum(xy * vertical[:, None, :], axis=2)
        aligned[:, :, 2] = relative[:, :, 2]
        return aligned.reshape(-1, 63)
