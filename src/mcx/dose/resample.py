"""Resampling dose onto the common analysis grid.

Every plan has its own dose grid origin and extent, so nothing can be compared until
they all live on one geometry. Two things this module refuses to do quietly:

*   It never invents dose outside the source grid. Voxels beyond the dose grid come
    back marked uncovered, and it is the caller's job to decide what that means. Filling
    them with zero would read as "no dose here", which is a different and false claim,
    and would silently corrupt any absolute-volume endpoint (D2cc, V70cc) on a structure
    that pokes outside the calculation box.
*   It never extrapolates. Interpolation is trilinear, which is what a dose grid
    supports; higher-order schemes overshoot across the steep penumbra gradients that
    dominate this dataset.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from mcx.geom.grid import Grid
from mcx.io.rtdose import DoseGrid


def resample_to_grid(
    dose: DoseGrid,
    grid: Grid,
    *,
    shift_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ``dose`` onto ``grid``.

    ``shift_mm`` translates the dose distribution in patient coordinates, for the
    setup-uncertainty sensitivity analysis. A shift of zero is the identity.

    Returns ``(dose_gy, covered)``: dose in Gy as float32 with NaN where the analysis
    grid falls outside the dose grid, and a boolean array marking where it does not.
    """
    nz, ny, nx = grid.shape
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    covered = np.zeros(grid.shape, dtype=bool)

    shift = np.asarray(shift_mm, dtype=float)
    src_nz, src_ny, src_nx = dose.array.shape

    # The grids are axis-aligned, so the index mapping is separable per axis.
    gx = grid.origin[0] + np.arange(nx) * grid.spacing[0] - shift[0]
    gy = grid.origin[1] + np.arange(ny) * grid.spacing[1] - shift[1]
    gz = grid.z - shift[2]

    ix = (gx - dose.origin[0]) / dose.spacing[0]
    iy = (gy - dose.origin[1]) / dose.spacing[1]
    iz = (gz - dose.z[0]) / dose.spacing[2]

    ok_x = (ix >= 0) & (ix <= src_nx - 1)
    ok_y = (iy >= 0) & (iy <= src_ny - 1)
    ok_z = (iz >= 0) & (iz <= src_nz - 1)
    if not (ok_x.any() and ok_y.any() and ok_z.any()):
        return out, covered

    grid_y, grid_x = np.meshgrid(iy, ix, indexing="ij")
    plane_ok = ok_y[:, None] & ok_x[None, :]

    for k, (zi, zok) in enumerate(zip(iz, ok_z, strict=True)):
        if not zok:
            continue
        coords = np.stack(
            [np.full(grid_y.shape, zi, dtype=float), grid_y, grid_x]
        ).reshape(3, -1)
        values = ndimage.map_coordinates(
            dose.array, coords, order=1, mode="nearest", prefilter=False
        ).reshape(ny, nx)
        out[k] = np.where(plane_ok, values.astype(np.float32), np.nan)
        covered[k] = plane_ok

    return out, covered


def coverage_fraction(mask: np.ndarray, covered: np.ndarray) -> float:
    """Fraction of a structure that lies inside the dose grid.

    Anything below 1.0 invalidates absolute-volume endpoints for that structure, which
    is why this is a gate rather than a note.
    """
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float((mask & covered).sum()) / n
