"""Polygon -> binary mask.

Two conventions, both deliberate and both checked by the tests.

**Within one source slice: even-odd (XOR) fill.** It is the only rule that handles
both cases present in this dataset with one code path -- two disjoint loops on a slice
(bilateral seminal vesicles) become their union, and a loop nested inside another
(rectal gas, bladder lumen) becomes a ring. A union rule would fill the holes; a
largest-loop-wins rule would drop a vesicle.

**Across source slices: union (OR).** Loops that came from *different* original CT
slices are different anatomy, so if two of them are forced onto the same target grid
plane they must combine by union. Using XOR across source slices would make them
cancel, silently hollowing out the structure. That is a live risk here: V027's CT is
2.0 mm while every dose grid is 2.5 mm, so contours genuinely do collide when
rasterised onto a native dose grid. Collisions are counted and reported, never hidden.

**Voxel membership is centre-in-polygon, boundary-inclusive.** For contours that
happen to align exactly with voxel centres this over-counts by one voxel per boundary
row; for real contours it is unbiased. The residual is bounded by the grid-convergence
sweep in stage 03 and by the independent shoelace volume in ``RasterReport``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from skimage.draw import polygon as _polygon

from mcx.geom.grid import Grid
from mcx.io.rtstruct import ROI

# A contour is assigned to the nearest grid plane only if it lies within this fraction
# of the slice spacing. Anything further is off-grid: reported, never snapped.
# Must be below 0.5, or no contour on a regular grid could ever be off-grid and the
# check would be unreachable.
Z_SNAP_FRACTION = 0.25

# Loops with fewer than this many vertices cannot bound an area.
MIN_VERTICES = 3


@dataclass(frozen=True)
class RasterReport:
    """What happened during rasterisation. Every field feeds a Tier 2 check."""

    n_loops: int
    n_loops_off_grid_z: int
    n_loops_degenerate: int
    n_loops_zero_area: int
    area_dropped_mm2: float
    n_loops_clipped_xy: int
    n_slices_multiloop: int
    max_loops_per_slice: int
    n_slices_z_collapsed: int
    z_gap_slices: int
    mask_volume_cc: float
    polygon_volume_cc: float

    @property
    def volume_agreement(self) -> float:
        """Relative difference between the two independent volume estimates."""
        if self.polygon_volume_cc <= 0:
            return float("nan")
        return abs(self.mask_volume_cc - self.polygon_volume_cc) / self.polygon_volume_cc

    @property
    def clean(self) -> bool:
        return (
            self.n_loops_off_grid_z == 0
            and self.n_loops_degenerate == 0
            and self.n_loops_zero_area == 0
            and self.n_loops_clipped_xy == 0
            and self.n_slices_z_collapsed == 0
            and self.z_gap_slices == 0
        )


def rasterise(roi: ROI, grid: Grid) -> tuple[np.ndarray, RasterReport]:
    """Rasterise an ROI onto ``grid``, returning ``(mask, report)``."""
    nz, ny, nx = grid.shape
    mask = np.zeros(grid.shape, dtype=bool)

    off_grid = degenerate = zero_area = clipped = 0
    dropped_mm2 = 0.0
    per_source_z: dict[float, int] = defaultdict(int)
    # target grid slice -> the set of source z values that landed on it
    landed: dict[int, set[float]] = defaultdict(set)
    # accumulate each source slice separately so XOR stays within one source slice
    planes: dict[tuple[int, float], np.ndarray] = {}

    for loop in roi.slices:
        per_source_z[loop.z] += 1

        if loop.n_points < MIN_VERTICES:
            # A point or a line cannot bound an area, so nothing is lost by dropping
            # it -- but record that it happened and how much area went with it.
            degenerate += 1
            dropped_mm2 += abs(loop.area_mm2())
            continue

        k = grid.nearest_slice(loop.z)
        if abs(grid.z[k] - loop.z) > Z_SNAP_FRACTION * grid.spacing[2]:
            off_grid += 1
            dropped_mm2 += abs(loop.area_mm2())
            continue

        col = (loop.points[:, 0] - grid.origin[0]) / grid.spacing[0]
        row = (loop.points[:, 1] - grid.origin[1]) / grid.spacing[1]
        if col.min() < -0.5 or col.max() > nx - 0.5 or row.min() < -0.5 or row.max() > ny - 0.5:
            clipped += 1

        rr, cc = _polygon(row, col, shape=(ny, nx))
        if rr.size == 0:
            # A sliver narrower than one voxel. Distinct from a degenerate loop:
            # the polygon is valid, it is just below the grid resolution.
            zero_area += 1
            dropped_mm2 += abs(loop.area_mm2())
            continue

        key = (k, loop.z)
        plane = planes.get(key)
        if plane is None:
            plane = np.zeros((ny, nx), dtype=bool)
            planes[key] = plane
        loop_plane = np.zeros((ny, nx), dtype=bool)
        loop_plane[rr, cc] = True
        # Even-odd within a source slice: disjoint loops union, nested loops subtract.
        plane ^= loop_plane
        landed[k].add(loop.z)

    # Union across source slices, so colliding source slices never cancel.
    for (k, _z), plane in planes.items():
        mask[k] |= plane

    collapsed = sum(1 for zs in landed.values() if len(zs) > 1)
    multiloop = {z: n for z, n in per_source_z.items() if n > 1}
    gaps = 0
    if landed:
        span = range(min(landed), max(landed) + 1)
        gaps = len([k for k in span if k not in landed])

    report = RasterReport(
        n_loops=len(roi.slices),
        n_loops_off_grid_z=off_grid,
        n_loops_degenerate=degenerate,
        n_loops_zero_area=zero_area,
        area_dropped_mm2=dropped_mm2,
        n_loops_clipped_xy=clipped,
        n_slices_multiloop=len(multiloop),
        max_loops_per_slice=max(per_source_z.values(), default=0),
        n_slices_z_collapsed=collapsed,
        z_gap_slices=gaps,
        mask_volume_cc=float(mask.sum()) * grid.voxel_volume_cc,
        polygon_volume_cc=roi.polygon_volume_cc(),
    )
    return mask, report


def grid_from_dose(dose) -> Grid:  # noqa: ANN001 - avoids a circular import
    """The native grid of an RTDOSE, for evaluating on the dose grid directly."""
    return Grid(
        origin=np.array([dose.origin[0], dose.origin[1], float(dose.z[0])]),
        spacing=dose.spacing,
        shape=dose.array.shape,
    )
