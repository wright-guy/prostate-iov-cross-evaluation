"""Tests for EQD2, LKB NTCP and Poisson TCP.

Every assertion is either closed form or a published value. None is copied from a
previous run of this code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.endpoints.biological import (  # noqa: E402
    calibrate_clonogen_density,
    eqd2,
    geud,
    lkb_ntcp,
    poisson_tcp,
)


# ------------------------------------------------------------------------ EQD2


def test_eqd2_is_the_identity_at_two_gray_per_fraction():
    """78 Gy in 39 fractions is 2 Gy/fraction, so EQD2 must return 78 Gy exactly."""
    d = np.array([78.0])
    assert eqd2(d, n_fractions=39, alpha_beta_gy=3.0)[0] == pytest.approx(78.0)
    assert eqd2(d, n_fractions=39, alpha_beta_gy=1.5)[0] == pytest.approx(78.0)


def test_eqd2_matches_the_closed_form():
    """EQD2 = D (d + ab) / (2 + ab), with d = D/n."""
    D, n, ab = 60.0, 20.0, 3.0          # 3 Gy per fraction
    expected = D * (D / n + ab) / (2.0 + ab)
    assert eqd2(np.array([D]), n_fractions=int(n), alpha_beta_gy=ab)[0] == pytest.approx(
        expected
    )
    assert expected > D  # hypofractionation is biologically hotter for late tissue


def test_eqd2_uses_each_voxels_own_fraction_size():
    """A hot voxel gets a bigger fraction size, so it must scale up more."""
    d = np.array([40.0, 80.0])
    out = eqd2(d, n_fractions=39, alpha_beta_gy=3.0)
    assert out[1] / d[1] > out[0] / d[0]


# ------------------------------------------------------------------------ gEUD


def test_geud_of_a_uniform_dose_is_that_dose():
    for n in (0.09, 0.5, 1.0):
        assert geud(np.full(500, 70.0), n) == pytest.approx(70.0)


def test_geud_at_n_equals_one_is_the_mean_dose():
    d = np.array([10.0, 20.0, 30.0, 80.0])
    assert geud(d, 1.0) == pytest.approx(d.mean())


def test_small_n_approaches_the_maximum_dose():
    """A serial organ (small n) is dominated by its hot spot."""
    d = np.array([10.0, 10.0, 10.0, 80.0])
    assert geud(d, 0.02) > 70.0
    assert geud(d, 0.02) < 80.0


def test_geud_is_monotonic_in_n():
    d = np.concatenate([np.full(90, 20.0), np.full(10, 80.0)])
    values = [geud(d, n) for n in (0.05, 0.1, 0.3, 0.7, 1.0)]
    assert values == sorted(values, reverse=True)


# ------------------------------------------------------------------------ NTCP


# A very large alpha/beta makes EQD2 the identity, which isolates the LKB core from
# the EQD2 conversion. The composition of the two is tested separately below.
NO_EQD2 = 1.0e9


def test_ntcp_is_one_half_at_td50():
    """A uniform gEUD equal to TD50 must give exactly 50% by construction."""
    out = lkb_ntcp(
        np.full(1000, 76.9), n=0.09, m=0.13, td50_gy=76.9,
        n_fractions=39, alpha_beta_gy=NO_EQD2, voxel_volume_cc=0.001,
    )
    assert out["gEUD"] == pytest.approx(76.9, rel=1e-6)
    assert out["NTCP"] == pytest.approx(0.5, abs=1e-6)


def test_ntcp_one_sigma_above_td50():
    """At gEUD = TD50(1 + m), t = 1, so NTCP is the normal CDF at 1."""
    td50, m = 76.9, 0.13
    out = lkb_ntcp(
        np.full(1000, td50 * (1 + m)), n=0.09, m=m, td50_gy=td50,
        n_fractions=39, alpha_beta_gy=NO_EQD2, voxel_volume_cc=0.001,
    )
    assert out["t"] == pytest.approx(1.0, rel=1e-6)
    assert out["NTCP"] == pytest.approx(0.8413447, abs=1e-6)


def test_ntcp_composes_eqd2_then_lkb():
    """78 Gy in 39 fractions is exactly 2 Gy/fraction, so EQD2 leaves it unchanged.

    At 76.9 Gy the fraction size is 1.972 Gy, so EQD2 pulls the dose DOWN and NTCP must
    fall below 0.5. An earlier version of this test wrongly assumed the composition was
    the identity at TD50.
    """
    kw = dict(n=0.09, m=0.13, td50_gy=76.9, n_fractions=39,
              alpha_beta_gy=3.0, voxel_volume_cc=0.001)
    at_td50_physical = lkb_ntcp(np.full(1000, 76.9), **kw)
    assert at_td50_physical["gEUD"] < 76.9
    assert at_td50_physical["NTCP"] < 0.5

    # The physical dose whose EQD2 is exactly TD50 restores NTCP = 0.5.
    out = lkb_ntcp(np.full(1000, 77.212011), **kw)
    assert out["gEUD"] == pytest.approx(76.9, abs=1e-3)
    assert out["NTCP"] == pytest.approx(0.5, abs=1e-4)


def test_ntcp_increases_with_dose():
    kw = dict(n=0.09, m=0.13, td50_gy=76.9, n_fractions=39,
              alpha_beta_gy=3.0, voxel_volume_cc=0.001)
    low = lkb_ntcp(np.full(1000, 40.0), **kw)["NTCP"]
    high = lkb_ntcp(np.full(1000, 78.0), **kw)["NTCP"]
    assert low < high


def test_relative_normalisation_is_blind_to_contour_size():
    """This is the point of reporting both normalisations."""
    kw = dict(n=0.09, m=0.13, td50_gy=76.9, n_fractions=39,
              alpha_beta_gy=3.0, voxel_volume_cc=0.001)
    small = lkb_ntcp(np.full(500, 70.0), **kw)
    large = lkb_ntcp(np.full(5000, 70.0), **kw)
    assert small["NTCP"] == pytest.approx(large["NTCP"])


def test_absolute_normalisation_responds_to_contour_size():
    """With a fixed denominator, doubling the contour must raise gEUD and NTCP."""
    kw = dict(n=0.09, m=0.13, td50_gy=76.9, n_fractions=39,
              alpha_beta_gy=3.0, voxel_volume_cc=0.01, reference_volume_cc=50.0)
    small = lkb_ntcp(np.full(2500, 70.0), **kw)   # 25 cc
    large = lkb_ntcp(np.full(5000, 70.0), **kw)   # 50 cc
    assert large["gEUD"] > small["gEUD"]
    assert large["NTCP"] > small["NTCP"]


# ------------------------------------------------------------------------- TCP


def _tcp(dose, volume_cc=50.0, **over):
    n_vox = 20_000
    kw = dict(
        alpha_mean=0.15, sigma_alpha=0.04, alpha_beta_gy=1.5,
        clonogen_density_per_cc=1.0e7, n_fractions=39,
        voxel_volume_cc=volume_cc / n_vox,
    )
    kw.update(over)
    return poisson_tcp(np.full(n_vox, dose), **kw)


def test_tcp_is_zero_at_zero_dose():
    assert _tcp(0.0)["TCP"] == pytest.approx(0.0, abs=1e-12)


def test_tcp_is_monotonic_in_dose():
    values = [_tcp(d)["TCP"] for d in (40.0, 60.0, 70.0, 78.0, 90.0)]
    assert values == sorted(values)
    assert values[-1] > 0.99


def test_tcp_matches_the_single_alpha_closed_form():
    """With sigma_alpha = 0, TCP = exp(-N * SF) exactly."""
    dose, volume, n_fx = 70.0, 50.0, 39
    alpha, ab, rho = 0.15, 1.5, 1.0e7
    beta = alpha / ab
    sf = np.exp(-alpha * dose - beta * dose * (dose / n_fx))
    expected = np.exp(-rho * volume * sf)
    out = _tcp(dose, volume_cc=volume, sigma_alpha=0.0)
    assert out["TCP"] == pytest.approx(expected, rel=1e-9)


def test_heterogeneity_flattens_the_dose_response():
    """A single-alpha model is far too steep; that is why sigma_alpha is used.

    The claim is about the MAXIMUM gradient of the curve, not about any particular dose
    window. Heterogeneity also shifts the curve, so a fixed window can end up comparing
    different parts of the two responses and give the wrong answer -- which is what an
    earlier version of this test did.
    """
    doses = np.arange(50.0, 95.0, 1.0)
    steep = np.diff([_tcp(d, sigma_alpha=0.0)["TCP"] for d in doses]).max()
    flat = np.diff([_tcp(d, sigma_alpha=0.04)["TCP"] for d in doses]).max()
    assert steep > 2 * flat, f"max slope {steep:.3f}/Gy against {flat:.3f}/Gy"


def test_fixed_density_penalises_a_larger_contour_at_identical_dose():
    """The artefact section 5 asks to be quantified: same dose, bigger contour, lower TCP."""
    small = _tcp(74.0, volume_cc=30.0)
    large = _tcp(74.0, volume_cc=90.0)
    assert large["n_clonogens"] > small["n_clonogens"]
    assert large["TCP"] < small["TCP"]


def test_fixed_number_removes_that_artefact():
    """Holding the clonogen count fixed, contour size alone must not move TCP."""
    small = _tcp(74.0, volume_cc=30.0, reference_volume_cc=50.0)
    large = _tcp(74.0, volume_cc=90.0, reference_volume_cc=50.0)
    assert small["n_clonogens"] == pytest.approx(large["n_clonogens"])
    assert small["TCP"] == pytest.approx(large["TCP"], rel=1e-9)


def test_published_density_saturates_at_the_prescription():
    """Documents WHY the density is calibrated rather than used as published.

    At 1e7 per cm3 the model gives 97.5% control at 78 Gy, on the flat top of the curve
    where every difference between contour sets is compressed toward zero. That would
    be a finding about the model's operating point, not about contouring.
    """
    tcp = _tcp(78.0, volume_cc=50.6, clonogen_density_per_cc=1.0e7)["TCP"]
    assert tcp > 0.95
    slope = (
        _tcp(79.0, volume_cc=50.6, clonogen_density_per_cc=1.0e7)["TCP"]
        - _tcp(77.0, volume_cc=50.6, clonogen_density_per_cc=1.0e7)["TCP"]
    ) / 2.0
    assert slope < 0.02, "under 2% per Gy: too flat to resolve contour effects"


def test_calibration_hits_its_target():
    rho = calibrate_clonogen_density(
        target_tcp=0.85, uniform_dose_gy=78.0, volume_cc=50.6,
        alpha_mean=0.15, sigma_alpha=0.04, alpha_beta_gy=1.5, n_fractions=39,
    )
    assert 1e8 < rho < 1e9
    assert _tcp(78.0, volume_cc=50.6, clonogen_density_per_cc=rho)["TCP"] == pytest.approx(
        0.85, abs=1e-3
    )


def test_calibrated_model_sits_on_a_usable_slope():
    """The point of calibrating: a slope that can resolve contour differences."""
    rho = calibrate_clonogen_density(
        target_tcp=0.85, uniform_dose_gy=78.0, volume_cc=50.6,
        alpha_mean=0.15, sigma_alpha=0.04, alpha_beta_gy=1.5, n_fractions=39,
    )
    slope = (
        _tcp(79.0, volume_cc=50.6, clonogen_density_per_cc=rho)["TCP"]
        - _tcp(77.0, volume_cc=50.6, clonogen_density_per_cc=rho)["TCP"]
    ) / 2.0
    assert slope > 0.03, "over 3% per Gy"
