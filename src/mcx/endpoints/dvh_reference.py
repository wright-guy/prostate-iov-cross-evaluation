"""A second, independent DVH estimator, for cross-validation only.

``mcx.endpoints.dvh`` works by sorting the dose values and indexing by rank. This one
builds a differential histogram and reads the cumulative curve. The two share no code
and fail in different ways: the sort-based version is exact but sensitive to rank
conventions at the ends of the distribution, while this one is approximate but bounded
by the bin width and insensitive to ranking.

Agreement between them does not prove either is right, so both are additionally pinned
against closed-form answers in tests/test_dvh_reference.py, where a linear dose ramp
over a cuboid gives every endpoint analytically.

Never use this in the pipeline proper. It exists to disagree with the real one.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BIN_GY = 0.01


def cumulative_dvh(
    values: np.ndarray, *, bin_gy: float = DEFAULT_BIN_GY
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dose_edges, volume_fraction_at_or_above)``."""
    if values.size == 0:
        return np.zeros(0), np.zeros(0)
    top = float(np.nanmax(values))
    edges = np.arange(0.0, top + 2 * bin_gy, bin_gy)
    counts, _ = np.histogram(values, bins=edges)
    # Fraction of the volume receiving at least the dose at each left bin edge.
    above = np.concatenate([[values.size], values.size - np.cumsum(counts)])
    return edges, above / values.size


def d_at_volume_pct(
    values: np.ndarray, pct: float, *, bin_gy: float = DEFAULT_BIN_GY
) -> float:
    """D_x% read off the cumulative curve rather than by rank."""
    edges, frac = cumulative_dvh(values, bin_gy=bin_gy)
    if edges.size == 0:
        return float("nan")
    target = pct / 100.0
    # frac is monotonically decreasing; find the last edge where frac >= target.
    idx = np.nonzero(frac >= target)[0]
    if idx.size == 0:
        return float(edges[0])
    return float(edges[idx[-1]])


def d_at_volume_cc(
    values: np.ndarray, cc: float, voxel_cc: float, *, bin_gy: float = DEFAULT_BIN_GY
) -> float:
    total_cc = values.size * voxel_cc
    if values.size == 0 or total_cc < cc:
        return float("nan")
    return d_at_volume_pct(values, 100.0 * cc / total_cc, bin_gy=bin_gy)


def v_at_dose_pct(values: np.ndarray, dose_gy: float, *, bin_gy: float = DEFAULT_BIN_GY) -> float:
    edges, frac = cumulative_dvh(values, bin_gy=bin_gy)
    if edges.size == 0:
        return float("nan")
    idx = int(np.searchsorted(edges, dose_gy, side="left"))
    idx = min(idx, frac.size - 1)
    return float(100.0 * frac[idx])


def mean_dose(values: np.ndarray) -> float:
    """Mean from the histogram's first moment, not from the raw values."""
    edges, frac = cumulative_dvh(values)
    if edges.size < 2:
        return float("nan")
    differential = -np.diff(frac)
    centres = edges[:-1] + 0.5 * (edges[1] - edges[0])
    return float(np.sum(differential * centres))
