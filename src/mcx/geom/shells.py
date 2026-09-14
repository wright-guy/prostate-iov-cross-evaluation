"""Derived structures: expansion shells, PRVs and boolean combinations.

Every expansion here is a true 3D isotropic expansion computed from the Euclidean
distance transform with the grid's real anisotropic spacing, not a slice-wise dilation
by a fixed number of voxels. On a grid whose z spacing is 2.5 times its in-plane
spacing, the latter would produce a shell that is far too thick superoinferiorly.
"""

from __future__ import annotations

import numpy as np

from mcx.geom.grid import Grid
from mcx.geom.surface import signed_distance


def expand(mask: np.ndarray, grid: Grid, margin_mm: float) -> np.ndarray:
    """Isotropic 3D expansion (or contraction, for a negative margin)."""
    if not mask.any():
        return np.zeros_like(mask)
    if margin_mm == 0:
        return mask.copy()
    return signed_distance(mask, grid) <= margin_mm


def shell(mask: np.ndarray, grid: Grid, inner_mm: float, outer_mm: float) -> np.ndarray:
    """The rind between two expansions, ``inner_mm`` < ``outer_mm``."""
    if inner_mm >= outer_mm:
        raise ValueError("inner margin must be smaller than outer")
    d = signed_distance(mask, grid)
    return (d <= outer_mm) & (d > inner_mm)


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a & b


def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a`` minus ``b``."""
    return a & ~b


def treated_volume(dose: np.ndarray, covered: np.ndarray, isodose_gy: float) -> np.ndarray:
    """ICRU 83 treated volume: the region enclosed by a given isodose.

    Restricted to where dose is defined; a voxel outside the calculation box is not
    "below the isodose", it is unknown, and treating it as below would inflate every
    conformity index.
    """
    return covered & np.nan_to_num(dose, nan=-1.0) >= isodose_gy


def paddick_conformity(treated: np.ndarray, target: np.ndarray) -> float:
    """Paddick conformity index: TV_intersect^2 / (TV * target).

    Penalises under-coverage and over-treatment symmetrically, unlike D98, which
    rewards a small target sitting inside a large high-dose region.
    """
    nt, ng = int(treated.sum()), int(target.sum())
    if nt == 0 or ng == 0:
        return float("nan")
    overlap = float(np.count_nonzero(treated & target))
    return (overlap * overlap) / (nt * ng)


def disagreement(plan_mask: np.ndarray, truth_mask: np.ndarray) -> dict[str, np.ndarray]:
    """False-positive and false-negative regions for one cross-evaluation cell.

    ``false_positive`` is anatomy the planning contour claimed that the truth does not
    contain; ``false_negative`` is truth the planning contour missed. Keeping them
    separate rather than summing to an unsigned overlap is the whole point of H6.
    """
    return {
        "false_positive": subtract(plan_mask, truth_mask),
        "false_negative": subtract(truth_mask, plan_mask),
        "intersection": intersect(plan_mask, truth_mask),
    }
