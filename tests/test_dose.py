"""Tests for dose resampling and the two DVH estimators.

The DVH tests use a linear dose ramp over a cuboid, where every endpoint has a
closed-form answer. Agreement between the two implementations would not prove either
is correct; agreement with the analytic value does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.dose.resample import coverage_fraction, resample_to_grid  # noqa: E402
from mcx.endpoints import dvh, dvh_reference  # noqa: E402
from mcx.geom.grid import Grid  # noqa: E402
from mcx.io.rtdose import DoseGrid  # noqa: E402


def make_dose(array: np.ndarray, origin=(0.0, 0.0, 0.0), spacing=(2.0, 2.0, 2.0)) -> DoseGrid:
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    return DoseGrid(
        array=array.astype(np.float32),
        origin=origin,
        spacing=spacing,
        z=origin[2] + np.arange(array.shape[0]) * spacing[2],
        path=Path("synthetic.dcm"),
        sop_instance_uid="1.2.3",
        referenced_rtplan_uid="1.2.4",
        summation_type="PLAN",
        dose_type="PHYSICAL",
        units="GY",
        frame_of_reference_uid="1.2.5",
    )


# ------------------------------------------------------------------ resampling


def test_resampling_onto_the_source_grid_is_the_identity():
    rng = np.random.default_rng(0)
    array = rng.uniform(0, 80, size=(6, 9, 11)).astype(np.float32)
    dose = make_dose(array, origin=(-5.0, 3.0, -7.0), spacing=(2.0, 2.0, 2.5))
    grid = Grid(origin=dose.origin.copy(), spacing=dose.spacing.copy(), shape=array.shape)

    out, covered = resample_to_grid(dose, grid)
    assert covered.all()
    assert np.allclose(out, array, atol=1e-4)


def test_zero_shift_is_the_identity():
    rng = np.random.default_rng(1)
    dose = make_dose(rng.uniform(0, 80, size=(5, 7, 7)).astype(np.float32))
    grid = Grid(origin=dose.origin.copy(), spacing=dose.spacing.copy(), shape=(5, 7, 7))
    a, _ = resample_to_grid(dose, grid)
    b, _ = resample_to_grid(dose, grid, shift_mm=(0.0, 0.0, 0.0))
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_shift_moves_the_distribution_the_right_way():
    """A +4 mm shift in x must move the pattern +4 mm in patient coordinates."""
    array = np.zeros((3, 5, 9), dtype=np.float32)
    array[1, 2, 4] = 100.0
    dose = make_dose(array, spacing=(2.0, 2.0, 2.0))
    grid = Grid(origin=np.zeros(3), spacing=np.array([2.0, 2.0, 2.0]), shape=(3, 5, 9))

    plain, _ = resample_to_grid(dose, grid)
    shifted, _ = resample_to_grid(dose, grid, shift_mm=(4.0, 0.0, 0.0))
    assert np.nanargmax(plain) != np.nanargmax(shifted)
    assert np.unravel_index(int(np.nanargmax(shifted)), grid.shape) == (1, 2, 6)


def test_trilinear_interpolation_is_exact_on_a_linear_field():
    """Trilinear interpolation reproduces a linear ramp exactly."""
    nz, ny, nx = 4, 6, 8
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    array = (2.0 * xx + 3.0 * yy + 5.0 * zz).astype(np.float32)
    dose = make_dose(array, spacing=(2.0, 2.0, 2.0))

    grid = Grid(origin=np.array([1.0, 1.0, 1.0]), spacing=np.array([1.0, 1.0, 1.0]),
                shape=(3, 5, 7))
    out, covered = resample_to_grid(dose, grid)
    gz, gy, gx = np.meshgrid(grid.z, np.arange(5) * 1.0 + 1.0, np.arange(7) * 1.0 + 1.0,
                             indexing="ij")
    expected = 2.0 * (gx / 2.0) + 3.0 * (gy / 2.0) + 5.0 * (gz / 2.0)
    assert covered.all()
    assert np.allclose(out, expected, atol=1e-4)


def test_outside_the_dose_grid_is_nan_not_zero():
    """Filling with zero would read as 'no dose here', which is a different claim."""
    dose = make_dose(np.full((3, 4, 4), 50.0, dtype=np.float32))
    grid = Grid(origin=np.array([-20.0, -20.0, -20.0]), spacing=np.array([2.0, 2.0, 2.0]),
                shape=(10, 20, 20))
    out, covered = resample_to_grid(dose, grid)
    assert np.isnan(out[~covered]).all()
    assert not covered.all()
    assert np.isfinite(out[covered]).all()


def test_coverage_fraction_counts_only_the_structure():
    covered = np.zeros((4, 4, 4), dtype=bool)
    covered[:2] = True
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3] = True
    assert coverage_fraction(mask, covered) == pytest.approx(0.5)


# ------------------------------------------------------- DVH, analytic ground truth


def ramp_values(n: int = 100_000, lo: float = 0.0, hi: float = 80.0) -> np.ndarray:
    """A uniform dose distribution on [lo, hi] — every endpoint is closed form."""
    return np.linspace(lo, hi, n)


def test_sorted_dvh_matches_the_analytic_ramp():
    """For dose uniform on [0, 80], D_x% = 80 * (1 - x/100)."""
    d = np.sort(ramp_values())[::-1]
    for pct in (2.0, 50.0, 95.0, 98.0):
        assert dvh.d_at_volume_pct(d, pct) == pytest.approx(80.0 * (1 - pct / 100), abs=0.01)
    assert dvh.mean_dose(d) == pytest.approx(40.0, abs=0.01)
    assert dvh.v_at_dose_pct(d, 40.0) == pytest.approx(50.0, abs=0.01)


def test_reference_dvh_matches_the_analytic_ramp():
    v = ramp_values()
    for pct in (2.0, 50.0, 95.0, 98.0):
        assert dvh_reference.d_at_volume_pct(v, pct) == pytest.approx(
            80.0 * (1 - pct / 100), abs=0.05
        )
    assert dvh_reference.mean_dose(v) == pytest.approx(40.0, abs=0.05)
    assert dvh_reference.v_at_dose_pct(v, 40.0) == pytest.approx(50.0, abs=0.05)


def test_the_two_estimators_agree_on_a_realistic_distribution():
    """Bimodal, like an OAR partly in the high-dose region."""
    rng = np.random.default_rng(7)
    v = np.concatenate([rng.normal(15, 6, 40_000), rng.normal(72, 3, 12_000)])
    v = np.clip(v, 0, None)
    d = np.sort(v)[::-1]
    for pct in (2.0, 35.0, 50.0, 95.0):
        a = dvh.d_at_volume_pct(d, pct)
        b = dvh_reference.d_at_volume_pct(v, pct)
        assert abs(a - b) < 0.05, f"D{pct}%: sorted {a:.4f} vs histogram {b:.4f}"
    assert abs(dvh.mean_dose(d) - dvh_reference.mean_dose(v)) < 0.05


def test_absolute_volume_endpoints_agree_between_estimators():
    rng = np.random.default_rng(8)
    v = rng.uniform(0, 80, 50_000)
    d = np.sort(v)[::-1]
    voxel_cc = 0.001
    for cc in (1.0, 2.0, 10.0):
        a = dvh.d_at_volume_cc(d, cc, voxel_cc)
        b = dvh_reference.d_at_volume_cc(v, cc, voxel_cc)
        assert abs(a - b) < 0.05, f"D{cc}cc: sorted {a:.4f} vs histogram {b:.4f}"
