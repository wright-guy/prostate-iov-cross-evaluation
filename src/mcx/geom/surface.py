"""Distance transforms and surface geometry.

Anisotropic spacing is passed explicitly to every transform. Forgetting it is the
classic way to get plausible-looking but wrong surface distances on a grid whose z
spacing is 2.5x its in-plane spacing, which is exactly this dataset.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from mcx.geom.grid import Grid


def signed_distance(mask: np.ndarray, grid: Grid) -> np.ndarray:
    """Signed distance to the mask surface in mm. Negative inside, positive outside."""
    sampling = grid.spacing[::-1]  # array axes are (z, y, x)
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, -np.inf, dtype=np.float32)
    outside = ndimage.distance_transform_edt(~mask, sampling=sampling)
    inside = ndimage.distance_transform_edt(mask, sampling=sampling)
    return (outside - inside).astype(np.float32)


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels: in the mask with at least one 6-neighbour outside it."""
    if not mask.any():
        return np.zeros_like(mask)
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))
    return mask & ~eroded


def expansion_distance(inner: np.ndarray, outer: np.ndarray, grid: Grid) -> dict[str, float]:
    """Distance from the ``inner`` surface out to the ``outer`` surface.

    For a genuine isotropic expansion this distribution is tight around the margin,
    which is how the PTV margin is measured back out of the data.
    """
    if not inner.any() or not outer.any():
        return {"median": float("nan"), "p05": float("nan"), "p95": float("nan"),
                "iqr": float("nan"), "contained": False}
    dist = signed_distance(inner, grid)
    shell = surface_voxels(outer)
    values = dist[shell]
    if values.size == 0:
        return {"median": float("nan"), "p05": float("nan"), "p95": float("nan"),
                "iqr": float("nan"), "contained": False}
    q25, q75 = np.percentile(values, [25, 75])
    return {
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "iqr": float(q75 - q25),
        "contained": bool((inner & ~outer).sum() == 0),
    }


def hausdorff_percentile(
    a: np.ndarray, b: np.ndarray, grid: Grid, *, percentile: float = 95.0
) -> float:
    """Symmetric percentile Hausdorff distance in mm.

    Symmetric because the one-sided version is not a metric and reports different
    values depending on which contour is called the reference.
    """
    if not a.any() or not b.any():
        return float("nan")
    da, db = signed_distance(a, grid), signed_distance(b, grid)
    sa, sb = surface_voxels(a), surface_voxels(b)
    both = np.concatenate([np.abs(db[sa]), np.abs(da[sb])])
    return float(np.percentile(both, percentile))
