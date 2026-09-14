"""Tests for contour-comparison metrics, against geometry with known answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.geom.grid import Grid  # noqa: E402
from mcx.metrics.core import (  # noqa: E402
    compare,
    dice,
    jaccard,
    joint_crop,
    region_volumes,
    signed_volume_difference,
    surface_metrics,
)


def grid(spacing=(1.0, 1.0, 1.0), shape=(60, 60, 60)) -> Grid:
    return Grid(origin=np.zeros(3), spacing=np.array(spacing), shape=shape)


def cube(shape, lo, size) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    z, y, x = lo
    dz, dy, dx = size if isinstance(size, tuple) else (size, size, size)
    m[z:z + dz, y:y + dy, x:x + dx] = True
    return m


# ------------------------------------------------------------------------ overlap


def test_dice_of_identical_masks_is_one():
    g = grid()
    a = cube(g.shape, (20, 20, 20), 10)
    assert dice(a, a) == 1.0
    assert jaccard(a, a) == 1.0


def test_dice_of_disjoint_masks_is_zero():
    g = grid()
    a = cube(g.shape, (10, 10, 10), 5)
    b = cube(g.shape, (40, 40, 40), 5)
    assert dice(a, b) == 0.0


def test_dice_matches_the_closed_form_for_a_known_overlap():
    """Two 10-cubes offset by 5 in x: overlap 5x10x10, so DSC = 2*500/2000 = 0.5."""
    g = grid()
    a = cube(g.shape, (20, 20, 20), 10)
    b = cube(g.shape, (20, 20, 25), 10)
    assert dice(a, b) == pytest.approx(0.5)
    assert jaccard(a, b) == pytest.approx(500 / 1500)


def test_both_empty_is_nan_not_one():
    """Two absent contours do not agree perfectly; the comparison is undefined."""
    g = grid()
    empty = np.zeros(g.shape, dtype=bool)
    assert np.isnan(dice(empty, empty))


# ------------------------------------------------------------------------- signed


def test_signed_volume_difference_sign_convention():
    g = grid()
    small = cube(g.shape, (20, 20, 20), 10)     # 1000
    large = cube(g.shape, (20, 20, 20), 20)     # 8000
    assert signed_volume_difference(large, small) == pytest.approx(7.0)
    assert signed_volume_difference(small, large) == pytest.approx(-0.875)


def test_fp_and_fn_are_reported_separately():
    """An unsigned metric would sum these; H6 says the direction is the information."""
    g = grid()
    plan = cube(g.shape, (20, 20, 20), 10)
    truth = cube(g.shape, (20, 20, 25), 10)
    r = region_volumes(plan, truth, voxel_cc=0.001)
    assert r["fp_volume_cc"] == pytest.approx(0.5)
    assert r["fn_volume_cc"] == pytest.approx(0.5)
    assert r["intersection_cc"] == pytest.approx(0.5)


def test_signed_msd_is_negative_when_the_plan_is_inside_the_truth():
    g = grid()
    inner = cube(g.shape, (25, 25, 25), 10)
    outer = cube(g.shape, (20, 20, 20), 20)
    m = surface_metrics(inner, outer, g)
    assert m["signed_msd_mm"] < 0
    assert surface_metrics(outer, inner, g)["signed_msd_mm"] > 0


# ------------------------------------------------------------------------ surface


def test_surface_distances_of_identical_masks_are_zero():
    g = grid()
    a = cube(g.shape, (20, 20, 20), 12)
    m = surface_metrics(a, a, g)
    assert m["msd_mm"] == pytest.approx(0.0)
    assert m["hd95_mm"] == pytest.approx(0.0)
    assert m["signed_msd_mm"] == pytest.approx(0.0)
    assert m["surface_dice"] == pytest.approx(1.0)


def test_hausdorff_of_a_translated_cube_is_the_translation():
    """A cube shifted 5 mm along x has a maximum surface separation of 5 mm."""
    g = grid()
    a = cube(g.shape, (20, 20, 20), 12)
    b = cube(g.shape, (20, 20, 25), 12)
    m = surface_metrics(a, b, g)
    assert m["hd_max_mm"] == pytest.approx(5.0, abs=1e-6)


def test_unsigned_surface_metrics_are_symmetric():
    """The one-sided form is not a metric and would depend on argument order."""
    g = grid()
    a = cube(g.shape, (20, 20, 20), 12)
    b = cube(g.shape, (22, 21, 25), 16)
    fwd, rev = surface_metrics(a, b, g), surface_metrics(b, a, g)
    for key in ("hd95_mm", "hd_max_mm", "msd_mm", "surface_dice"):
        assert fwd[key] == pytest.approx(rev[key]), key


def test_signed_msd_is_not_symmetric():
    """It must carry direction, which is the whole reason H6 distinguishes the classes.

    Note the sign does NOT simply flip for partially overlapping contours: where
    neither contains the other, both surfaces lie partly outside the other and both
    means can be positive. The sign-flip property holds for nested contours, which
    test_signed_msd_is_negative_when_the_plan_is_inside_the_truth covers.
    """
    g = grid()
    a = cube(g.shape, (20, 20, 20), 12)
    b = cube(g.shape, (22, 21, 25), 16)
    fwd, rev = surface_metrics(a, b, g), surface_metrics(b, a, g)
    assert fwd["signed_msd_mm"] != pytest.approx(rev["signed_msd_mm"])
    # ...while the unsigned mean of the same pair is identical either way.
    assert fwd["msd_mm"] == pytest.approx(rev["msd_mm"])


def test_anisotropic_spacing_is_respected():
    """A 1-voxel shift in z on a 2.5 mm grid is 2.5 mm, not 1 mm.

    Getting this wrong is the classic way to produce plausible but wrong surface
    distances on a grid whose z spacing differs from its in-plane spacing.
    """
    g = grid(spacing=(1.0, 1.0, 2.5))
    a = cube(g.shape, (20, 20, 20), 12)
    b = cube(g.shape, (21, 20, 20), 12)   # shifted one slice in z
    assert surface_metrics(a, b, g)["hd_max_mm"] == pytest.approx(2.5, abs=1e-6)


def test_cropping_does_not_change_the_answer():
    """The crop is an optimisation; it must not alter a single metric."""
    g_small = grid(shape=(40, 40, 40))
    g_big = grid(shape=(120, 120, 120))
    a_s = cube(g_small.shape, (10, 10, 10), 12)
    b_s = cube(g_small.shape, (12, 11, 14), 12)
    a_b = cube(g_big.shape, (10, 10, 10), 12)
    b_b = cube(g_big.shape, (12, 11, 14), 12)
    m_s, m_b = surface_metrics(a_s, b_s, g_small), surface_metrics(a_b, b_b, g_big)
    for key in m_s:
        assert m_s[key] == pytest.approx(m_b[key], rel=1e-9), key


def test_crop_margin_exceeds_the_distances_read_back():
    from mcx.metrics.core import CROP_MARGIN_MM

    g = grid()
    a = cube(g.shape, (20, 20, 20), 10)
    b = cube(g.shape, (25, 25, 25), 10)
    crop = joint_crop(a, b, g)
    assert crop is not None
    assert surface_metrics(a, b, g)["hd_max_mm"] < CROP_MARGIN_MM


def test_compare_returns_every_declared_metric():
    g = grid()
    a = cube(g.shape, (20, 20, 20), 12)
    b = cube(g.shape, (22, 21, 24), 12)
    out = compare(a, b, g)
    for key in ("dsc", "hd95_mm", "msd_mm", "signed_msd_mm",
                "signed_volume_difference", "fp_volume_cc", "fn_volume_cc"):
        assert key in out and not np.isnan(out[key]), key
