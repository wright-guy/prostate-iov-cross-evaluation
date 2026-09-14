"""EQD2, LKB NTCP and Poisson TCP.

Everything here works on a voxel dose array plus a voxel volume, not on a binned DVH,
so no binning convention enters the biological result. All three functions are pinned
to closed-form or published values in tests/test_biological.py.

The study reports DIFFERENCES across contour sets at fixed delivered dose. Absolute
values depend on calibration; the local slope is what the design measures robustly.
"""

from __future__ import annotations

import numpy as np
from scipy import special


def eqd2(dose_gy: np.ndarray, n_fractions: int, alpha_beta_gy: float) -> np.ndarray:
    """Convert physical dose to the equivalent dose in 2 Gy fractions.

    Uses each voxel's own dose per fraction, ``d = D / n``, which is the correct form
    for a heterogeneous distribution: a hot voxel receives a larger fraction size and
    is therefore biologically more effective than scaling by the prescription would
    suggest.
    """
    d = np.asarray(dose_gy, dtype=float) / n_fractions
    return np.asarray(dose_gy, dtype=float) * (d + alpha_beta_gy) / (2.0 + alpha_beta_gy)


def geud(dose_gy: np.ndarray, n: float, *, weights: np.ndarray | None = None) -> float:
    """Generalised equivalent uniform dose with volume parameter ``n``.

    ``weights`` are fractional volumes summing to 1 for relative normalisation, or to
    V_structure / V_reference for absolute normalisation -- which is exactly how the
    dilution artefact is exposed.
    """
    d = np.asarray(dose_gy, dtype=float)
    if d.size == 0:
        return float("nan")
    w = np.full(d.size, 1.0 / d.size) if weights is None else np.asarray(weights, float)
    if n <= 0:
        raise ValueError("n must be positive")
    # a = 1/n; gEUD = (sum w_i D_i^a)^(1/a).
    #
    # For a serial organ n is small, so a is large: the QUANTEC rectum parameter n = 0.09
    # gives a = 11.1, and 80^11.1 is about 1e21. Summed over tens of thousands of voxels
    # that stays inside float64, but it is close enough to the edge to be worth removing.
    # Factoring out the maximum dose first makes every base at most 1, so the sum cannot
    # overflow whatever n is.
    a = 1.0 / n
    positive = np.clip(d, 0.0, None)
    dmax = float(positive.max())
    if dmax <= 0:
        return 0.0
    total = float(np.sum(w * np.power(positive / dmax, a)))
    return float(dmax * total ** (1.0 / a)) if total > 0 else 0.0


def lkb_ntcp(
    dose_gy: np.ndarray,
    *,
    n: float,
    m: float,
    td50_gy: float,
    n_fractions: int,
    alpha_beta_gy: float,
    voxel_volume_cc: float,
    reference_volume_cc: float | None = None,
) -> dict[str, float]:
    """Lyman-Kutcher-Burman NTCP from a voxel dose array.

    ``reference_volume_cc`` selects the normalisation. ``None`` gives relative
    normalisation, where the weights sum to 1 regardless of contour size. A value gives
    absolute normalisation, where the weights sum to V_structure / V_reference, so a
    larger contour raises gEUD rather than diluting it. Reporting both is what separates
    a genuine dose effect from mechanical dilution.
    """
    d = np.asarray(dose_gy, dtype=float)
    if d.size == 0:
        return {"gEUD": float("nan"), "NTCP": float("nan"), "t": float("nan")}

    d2 = eqd2(d, n_fractions, alpha_beta_gy)
    volume_cc = d.size * voxel_volume_cc
    if reference_volume_cc is None:
        weights = np.full(d.size, 1.0 / d.size)
    else:
        weights = np.full(d.size, voxel_volume_cc / reference_volume_cc)

    g = geud(d2, n, weights=weights)
    t = (g - td50_gy) / (m * td50_gy)
    return {
        "gEUD": g,
        "t": float(t),
        "NTCP": float(0.5 * (1.0 + special.erf(t / np.sqrt(2.0)))),
        "volume_cc": float(volume_cc),
    }


def poisson_tcp(
    dose_gy: np.ndarray,
    *,
    alpha_mean: float,
    sigma_alpha: float,
    alpha_beta_gy: float,
    clonogen_density_per_cc: float,
    n_fractions: int,
    voxel_volume_cc: float,
    reference_volume_cc: float | None = None,
    n_alpha_nodes: int = 61,
) -> dict[str, float]:
    """Poisson TCP with Gaussian inter-patient heterogeneity in alpha.

    ``reference_volume_cc`` selects the clonogen convention. ``None`` is fixed density,
    where the clonogen count scales with the contour, so a larger contour reports a
    lower TCP at identical dose -- an artefact of the contour, not the dose. A value
    gives fixed number, where the total clonogen count is held constant and only the
    dose distribution matters. Section 5 requires both.
    """
    d = np.asarray(dose_gy, dtype=float)
    if d.size == 0:
        return {"TCP": float("nan"), "n_clonogens": float("nan")}

    volume_cc = d.size * voxel_volume_cc
    target_cc = volume_cc if reference_volume_cc is None else reference_volume_cc
    # Clonogens per voxel, so that the total is rho * target_cc either way.
    per_voxel = clonogen_density_per_cc * target_cc / d.size

    beta = alpha_mean / alpha_beta_gy
    dose_per_fraction = d / n_fractions
    quadratic = beta * d * dose_per_fraction  # beta * D * d

    if sigma_alpha <= 0:
        nodes = np.array([alpha_mean])
        weights = np.array([1.0])
    else:
        lo, hi = alpha_mean - 4.0 * sigma_alpha, alpha_mean + 4.0 * sigma_alpha
        nodes = np.linspace(max(lo, 1e-6), hi, n_alpha_nodes)
        density = np.exp(-0.5 * ((nodes - alpha_mean) / sigma_alpha) ** 2)
        weights = density / density.sum()

    # TCP(alpha) = exp(-sum_i N_i * SF_i(alpha)), then average over the alpha population.
    linear = np.outer(nodes, d)                      # (n_alpha, n_voxel)
    surviving = np.exp(-linear - quadratic[None, :])
    expected = per_voxel * surviving.sum(axis=1)
    tcp_by_alpha = np.exp(-expected)

    return {
        "TCP": float(np.sum(weights * tcp_by_alpha)),
        "n_clonogens": float(clonogen_density_per_cc * target_cc),
        "volume_cc": float(volume_cc),
    }


def calibrate_clonogen_density(
    *,
    target_tcp: float,
    uniform_dose_gy: float,
    volume_cc: float,
    alpha_mean: float,
    sigma_alpha: float,
    alpha_beta_gy: float,
    n_fractions: int,
    n_voxels: int = 20_000,
    bracket: tuple[float, float] = (1e5, 1e13),
) -> float:
    """Solve for the clonogen density giving ``target_tcp`` at a uniform prescription.

    Why this exists. With the published density of 1e7 per cm3 the model returns
    TCP = 0.975 at 78 Gy to a 50.6 cm3 target -- near saturation, where the local slope
    is under 2% per Gy and every difference between contour sets is compressed toward
    zero. That would not be a finding about contouring; it would be a finding about
    where the model happens to sit on its own curve.

    Calibrating the density to a stated clinical control probability puts the model on
    the steep part of the curve, where the local slope is clinically representative.
    It is a standard step, but it IS our choice rather than the source publication's,
    so the uncalibrated value is retained as a sensitivity analysis and both are
    reported.
    """
    from scipy.optimize import brentq

    def tcp_at(log10_rho: float) -> float:
        return poisson_tcp(
            np.full(n_voxels, uniform_dose_gy),
            alpha_mean=alpha_mean,
            sigma_alpha=sigma_alpha,
            alpha_beta_gy=alpha_beta_gy,
            clonogen_density_per_cc=10.0**log10_rho,
            n_fractions=n_fractions,
            voxel_volume_cc=volume_cc / n_voxels,
        )["TCP"]

    lo, hi = np.log10(bracket[0]), np.log10(bracket[1])
    return float(10.0 ** brentq(lambda r: tcp_at(r) - target_tcp, lo, hi, xtol=1e-6))
