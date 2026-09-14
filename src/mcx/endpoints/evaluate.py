"""Evaluate the declared endpoint list against a dose distribution and a mask.

Every endpoint in ``config/endpoints.yaml`` is dispatched from here, so the pipeline
computes exactly what the pre-registered protocol declares and nothing else. Adding an
endpoint means editing the config, not the code.

Dose values are taken only where the dose is defined. A voxel outside the calculation
box is not zero dose, it is unknown, and Tier 3 established that this never happens for
a scoring structure in this dataset -- but the guard stays, because a silent zero would
be indistinguishable from a real cold spot.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mcx.endpoints import dvh


def structure_dose(dose: np.ndarray, covered: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Descending-sorted dose values inside a structure, restricted to defined dose."""
    values = dose[mask & covered]
    values = values[np.isfinite(values)]
    return np.sort(values)[::-1]


def evaluate(spec: dict[str, Any], desc: np.ndarray, voxel_cc: float) -> float:
    """One endpoint from its declaration and a sorted dose array."""
    kind = spec["kind"]
    if desc.size == 0:
        return float("nan")
    if kind == "d_at_volume_pct":
        return dvh.d_at_volume_pct(desc, float(spec["pct"]))
    if kind == "d_at_volume_cc":
        return dvh.d_at_volume_cc(desc, float(spec["cc"]), voxel_cc)
    if kind == "v_at_dose_pct":
        return dvh.v_at_dose_pct(desc, float(spec["dose_gy"]))
    if kind == "v_at_dose_cc":
        return dvh.v_at_dose_cc(desc, float(spec["dose_gy"]), voxel_cc)
    if kind == "mean_dose":
        return dvh.mean_dose(desc)
    if kind == "d_max":
        return float(desc[0])
    if kind == "d_min":
        return float(desc[-1])
    if kind == "homogeneity_index":
        d2 = dvh.d_at_volume_pct(desc, 2)
        d98 = dvh.d_at_volume_pct(desc, 98)
        d50 = dvh.d_at_volume_pct(desc, 50)
        return float((d2 - d98) / d50) if d50 > 0 else float("nan")
    if kind == "volume_cc":
        return float(desc.size) * voxel_cc
    raise ValueError(f"unsupported endpoint kind {kind!r}")


def target_specs(cfg) -> list[dict[str, Any]]:  # noqa: ANN001
    return list(cfg.endpoints_raw["target"]["endpoints"])


def oar_specs(cfg) -> dict[str, list[dict[str, Any]]]:  # noqa: ANN001
    out: dict[str, list[dict[str, Any]]] = {}
    for oar in cfg.endpoints_raw["oars"]:
        out[oar["structure"]] = list(oar["relative"]) + list(oar["absolute"])
    return out


def ladder_specs(cfg) -> tuple[list[float], list[dict[str, Any]]]:  # noqa: ANN001
    ladder = cfg.endpoints_raw["coverage_ladder"]
    return [float(m) for m in ladder["expansions_mm"]], list(ladder["endpoints"])


def boundary_gradient(d98_by_margin: dict[float, float], prescription_gy: float) -> float:
    """Slope of D98 against expansion distance, in percent of prescription per mm.

    This is §5's quantitative statement of the CTV-optimisation limitation: how fast
    coverage falls away outside the optimised target. A steep gradient means the
    coverage results are strongly dependent on where the target boundary was drawn.
    """
    margins = sorted(m for m, v in d98_by_margin.items() if np.isfinite(v))
    if len(margins) < 2 or prescription_gy <= 0:
        return float("nan")
    x = np.array(margins, dtype=float)
    y = np.array([100.0 * d98_by_margin[m] / prescription_gy for m in margins])
    return float(np.polyfit(x, y, 1)[0])
