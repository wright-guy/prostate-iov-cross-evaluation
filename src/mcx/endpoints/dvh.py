"""Dose-volume statistics on a mask.

Kept deliberately small and dependency-free so it can be cross-validated against an
independent implementation in Tier 3 without shared code paths.
"""

from __future__ import annotations

import numpy as np


def dose_in_mask(dose: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Descending-sorted dose values inside a mask."""
    if dose.shape != mask.shape:
        raise ValueError(f"shape mismatch: dose {dose.shape} vs mask {mask.shape}")
    values = dose[mask]
    return np.sort(values)[::-1]


def d_at_volume_pct(sorted_desc: np.ndarray, pct: float) -> float:
    """D_x%: the minimum dose received by the hottest x% of the volume.

    Uses the standard cumulative-DVH convention. With n voxels, D_x% is the dose at
    rank ceil(x/100 * n), 1-indexed.
    """
    n = sorted_desc.size
    if n == 0:
        return float("nan")
    k = int(np.ceil(pct / 100.0 * n))
    return float(sorted_desc[min(max(k, 1) - 1, n - 1)])


def d_at_volume_cc(sorted_desc: np.ndarray, cc: float, voxel_cc: float) -> float:
    """D_Ncc: the minimum dose received by the hottest N cm3.

    Returns NaN when the structure is smaller than N cm3 rather than silently
    reporting its minimum dose, which would be a different quantity.
    """
    n = sorted_desc.size
    if n == 0 or n * voxel_cc < cc:
        return float("nan")
    k = int(np.ceil(cc / voxel_cc))
    return float(sorted_desc[min(max(k, 1) - 1, n - 1)])


def v_at_dose_pct(sorted_desc: np.ndarray, dose_gy: float) -> float:
    """V_D as a percentage of the structure volume."""
    if sorted_desc.size == 0:
        return float("nan")
    return float(100.0 * np.count_nonzero(sorted_desc >= dose_gy) / sorted_desc.size)


def v_at_dose_cc(sorted_desc: np.ndarray, dose_gy: float, voxel_cc: float) -> float:
    """V_D as an absolute volume in cm3."""
    if sorted_desc.size == 0:
        return float("nan")
    return float(np.count_nonzero(sorted_desc >= dose_gy) * voxel_cc)


def mean_dose(sorted_desc: np.ndarray) -> float:
    return float(sorted_desc.mean()) if sorted_desc.size else float("nan")


def summary(sorted_desc: np.ndarray, voxel_cc: float) -> dict[str, float]:
    """A compact set used by Tier 1 and the monotonicity checks."""
    return {
        "volume_cc": float(sorted_desc.size * voxel_cc),
        "D98": d_at_volume_pct(sorted_desc, 98),
        "D95": d_at_volume_pct(sorted_desc, 95),
        "D50": d_at_volume_pct(sorted_desc, 50),
        "D2": d_at_volume_pct(sorted_desc, 2),
        "Dmean": mean_dose(sorted_desc),
        "Dmax": float(sorted_desc[0]) if sorted_desc.size else float("nan"),
    }
