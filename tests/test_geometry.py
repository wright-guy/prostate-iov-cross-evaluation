"""Unit tests for grid, rasterisation and DVH arithmetic.

These use synthetic geometry with analytically known answers. The point is that a
refactor cannot silently move a number: every value here is derived by hand, not
copied from a previous run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.endpoints.dvh import (  # noqa: E402
    d_at_volume_cc,
    d_at_volume_pct,
    dose_in_mask,
    v_at_dose_cc,
    v_at_dose_pct,
)
from mcx.geom.grid import Grid  # noqa: E402
from mcx.geom.rasterise import rasterise  # noqa: E402
from mcx.io.rtstruct import ROI, ContourSlice  # noqa: E402


def square_loop(z: float, cx: float, cy: float, half: float) -> ContourSlice:
    pts = np.array(
        [
            [cx - half, cy - half, z],
            [cx + half, cy - half, z],
            [cx + half, cy + half, z],
            [cx - half, cy + half, z],
        ]
    )
    return ContourSlice(z=z, points=pts, geometric_type="CLOSED_PLANAR")


def unit_grid(n: int = 40, dz: float = 1.0) -> Grid:
    return Grid(origin=np.zeros(3), spacing=np.array([1.0, 1.0, dz]), shape=(10, n, n))


# ------------------------------------------------------------------------ Grid


def test_world_voxel_roundtrip():
    g = Grid(origin=np.array([-10.0, 5.0, -3.0]), spacing=np.array([2.0, 2.0, 2.5]),
             shape=(8, 16, 16))
    pts = np.array([[-10.0, 5.0, -3.0], [0.0, 11.0, 4.5]])
    assert np.allclose(g.voxel_to_world(g.world_to_voxel(pts)), pts)


def test_nearest_slice_clamps():
    g = unit_grid()
    assert g.nearest_slice(-100.0) == 0
    assert g.nearest_slice(1000.0) == g.shape[0] - 1
    assert g.nearest_slice(3.4) == 3


def test_from_bounds_covers_request():
    g = Grid.from_bounds(np.array([0.0, 0.0, 0.0]), np.array([9.0, 9.0, 9.0]), 1.0)
    lo, hi = g.bounds()
    assert np.all(lo <= 0.0) and np.all(hi >= 9.0)


# ----------------------------------------------------------------- rasterise


def test_single_square_volume_is_exact():
    """A 10x10 mm square on 3 slices of 1 mm = 300 mm3 = 0.3 cc.

    The square is offset by half a voxel so that no voxel centre lies exactly on its
    boundary; see test_grid_aligned_boundary_is_inclusive for the degenerate case.
    """
    g = unit_grid()
    roi = ROI("sq", 1, tuple(square_loop(z, 20.5, 20.5, 5.0) for z in (2.0, 3.0, 4.0)))
    mask, rep = rasterise(roi, g)
    assert rep.n_loops == 3
    assert rep.clean
    assert mask.sum(axis=(1, 2)).tolist()[2:5] == [100, 100, 100]
    assert rep.mask_volume_cc == pytest.approx(0.3, rel=1e-9)


def test_grid_aligned_boundary_is_inclusive():
    """Documents the convention: voxel centres exactly on the boundary count in.

    A 10 mm square whose edges fall exactly on voxel centres spans 11 centres, not 10.
    Real contours do not align this way, so the bias is a degenerate case rather than a
    systematic error -- but it is the convention, and it is pinned here on purpose.
    """
    g = unit_grid()
    roi = ROI("aligned", 1, (square_loop(2.0, 20.0, 20.0, 5.0),))
    mask, _ = rasterise(roi, g)
    assert mask.sum() == 121


def test_two_disjoint_loops_union_not_cancel():
    """Bilateral seminal vesicles: XOR of disjoint regions must be their union."""
    g = unit_grid()
    roi = ROI(
        "sv", 1,
        (square_loop(2.0, 10.5, 20.5, 3.0), square_loop(2.0, 30.5, 20.5, 3.0)),
    )
    mask, rep = rasterise(roi, g)
    assert rep.max_loops_per_slice == 2
    assert mask.sum() == 2 * 36
    assert rep.volume_agreement < 0.05


def test_nested_loop_becomes_ring():
    """Rectal gas: an inner loop must subtract, not fill."""
    g = unit_grid()
    roi = ROI(
        "rect", 1,
        (square_loop(2.0, 20.5, 20.5, 8.0), square_loop(2.0, 20.5, 20.5, 3.0)),
    )
    mask, _ = rasterise(roi, g)
    assert mask.sum() == 16 * 16 - 6 * 6
    assert not mask[2, 20, 20]  # centre is hollow


def test_degenerate_loop_is_reported_not_filled():
    g = unit_grid()
    bad = ContourSlice(z=2.0, points=np.array([[1.0, 1.0, 2.0], [2.0, 2.0, 2.0]]),
                       geometric_type="CLOSED_PLANAR")
    roi = ROI("bad", 1, (square_loop(2.0, 20.5, 20.5, 5.0), bad))
    mask, rep = rasterise(roi, g)
    assert rep.n_loops_degenerate == 1
    assert not rep.clean
    assert mask.sum() == 100


def test_subvoxel_sliver_is_distinguished_from_a_degenerate_loop():
    """A valid polygon below grid resolution is a different defect from a stray point.

    Both are dropped, but they are waived for different reasons, so the report has to
    tell them apart -- and record how much area went with each.
    """
    g = unit_grid()
    sliver = ContourSlice(
        z=2.0,
        # Offset off the integer lattice so the triangle encloses no voxel centre.
        points=np.array([[5.2, 5.2, 2.0], [5.8, 5.2, 2.0], [5.8, 5.5, 2.0]]),
        geometric_type="CLOSED_PLANAR",
    )
    roi = ROI("sliver", 1, (square_loop(2.0, 20.5, 20.5, 5.0), sliver))
    mask, rep = rasterise(roi, g)
    assert rep.n_loops_zero_area == 1
    assert rep.n_loops_degenerate == 0
    assert rep.area_dropped_mm2 == pytest.approx(0.09, abs=1e-6)
    assert not rep.clean
    assert mask.sum() == 100


def test_dropped_area_is_zero_for_a_clean_roi():
    g = unit_grid()
    roi = ROI("sq", 1, (square_loop(2.0, 20.5, 20.5, 5.0),))
    _, rep = rasterise(roi, g)
    assert rep.area_dropped_mm2 == 0.0
    assert rep.clean


def test_off_grid_z_is_reported_not_snapped():
    """A contour further than Z_SNAP_FRACTION from a grid plane must not be moved."""
    g = unit_grid(dz=1.0)
    roi = ROI("off", 1, (square_loop(2.4, 20.5, 20.5, 5.0),))
    mask, rep = rasterise(roi, g)
    assert rep.n_loops_off_grid_z == 1
    assert mask.sum() == 0


def test_snap_fraction_is_reachable():
    """A fraction of 0.5 or more would make the off-grid check dead code."""
    from mcx.geom.rasterise import Z_SNAP_FRACTION

    assert Z_SNAP_FRACTION < 0.5


def test_colliding_source_slices_union_and_are_reported():
    """Two source slices forced onto one grid plane must not cancel.

    This is live for V027: 2.0 mm CT contours rasterised onto a 2.5 mm dose grid.
    XOR across source slices would hollow the structure out silently.
    """
    g = unit_grid(dz=2.0)
    roi = ROI(
        "collide", 1,
        (square_loop(2.0, 20.5, 20.5, 5.0), square_loop(2.4, 20.5, 20.5, 5.0)),
    )
    mask, rep = rasterise(roi, g)
    assert rep.n_slices_z_collapsed == 1
    assert not rep.clean
    assert mask.sum() == 100  # union, not zero


def test_internal_z_gap_is_detected():
    g = unit_grid()
    roi = ROI("gap", 1, tuple(square_loop(z, 20.5, 20.5, 5.0) for z in (2.0, 4.0)))
    _, rep = rasterise(roi, g)
    assert rep.z_gap_slices == 1


def test_clipped_in_plane_is_flagged():
    g = unit_grid(n=20)
    roi = ROI("big", 1, (square_loop(2.0, 10.5, 10.5, 30.0),))
    _, rep = rasterise(roi, g)
    assert rep.n_loops_clipped_xy == 1


def test_polygon_volume_independent_of_rasteriser():
    """The round-trip check must compare two genuinely different computations."""
    roi = ROI("sq", 1, tuple(square_loop(z, 20.5, 20.5, 5.0) for z in (0.0, 1.0, 2.0)))
    assert roi.polygon_volume_cc() == pytest.approx(0.3, rel=1e-9)


# ----------------------------------------------------------------------- DVH


def test_d_at_volume_pct_conventions():
    """100 voxels, dose 1..100 Gy. D50 is the 50th hottest = 51 Gy."""
    d = dose_in_mask(np.arange(1, 101, dtype=float).reshape(1, 10, 10),
                     np.ones((1, 10, 10), bool))
    assert d[0] == 100.0
    assert d_at_volume_pct(d, 100) == 1.0
    assert d_at_volume_pct(d, 50) == 51.0
    assert d_at_volume_pct(d, 2) == 99.0


def test_v_at_dose_is_percentage_and_volume():
    d = dose_in_mask(np.arange(1, 101, dtype=float).reshape(1, 10, 10),
                     np.ones((1, 10, 10), bool))
    assert v_at_dose_pct(d, 51.0) == pytest.approx(50.0)
    assert v_at_dose_cc(d, 51.0, voxel_cc=0.01) == pytest.approx(0.5)


def test_d_at_volume_cc_returns_nan_for_small_structure():
    """D2cc of a 1 cc structure is undefined and must not silently become Dmin."""
    d = np.array([10.0, 9.0, 8.0])
    assert np.isnan(d_at_volume_cc(d, cc=2.0, voxel_cc=0.1))
    assert d_at_volume_cc(d, cc=0.1, voxel_cc=0.1) == 10.0


def test_dose_in_mask_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        dose_in_mask(np.zeros((2, 2, 2)), np.ones((3, 3, 3), bool))
