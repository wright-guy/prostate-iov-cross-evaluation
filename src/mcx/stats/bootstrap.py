"""Cluster bootstrap over patients.

Cells within a patient share anatomy, a dose grid and a plan library, so they are not
independent. Resampling cells would understate every interval substantially; resampling
patients is the honest unit. With five patients the intervals are wide, and that width
is the real information content of the design.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 20260904


def cluster_bootstrap(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    cluster: str = "patient",
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> dict[str, float]:
    """Percentile interval for ``statistic`` under resampling of whole clusters.

    Clusters are drawn with replacement, so a patient can appear more than once in a
    resample; all of that patient's rows travel together.

    Resampling is done by index rather than by rebuilding a frame. The obvious
    implementation, ``pd.concat`` of the drawn clusters on every draw, costs on the
    order of a millisecond, which is invisible once and half an hour across an endpoint
    list at ten thousand resamples. Concatenating integer positions and taking them in
    one call is the same operation two orders of magnitude cheaper.
    """
    clusters = df[cluster].unique()
    if len(clusters) < 2:
        point = statistic(df)
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_clusters": len(clusters), "n_valid": 0}

    positions = {c: np.flatnonzero((df[cluster] == c).to_numpy()) for c in clusters}
    rng = np.random.default_rng(seed)
    values = np.full(n_resamples, np.nan)

    for k in range(n_resamples):
        picked = rng.choice(len(clusters), size=len(clusters), replace=True)
        idx = np.concatenate([positions[clusters[p]] for p in picked])
        try:
            values[k] = statistic(df.take(idx))
        except (ValueError, ZeroDivisionError, KeyError):
            values[k] = np.nan

    finite = values[np.isfinite(values)]
    alpha = (1.0 - level) / 2.0
    return {
        "point": statistic(df),
        "lo": float(np.percentile(finite, 100 * alpha)) if finite.size else float("nan"),
        "hi": float(np.percentile(finite, 100 * (1 - alpha))) if finite.size else float("nan"),
        "n_clusters": int(len(clusters)),
        "n_valid": int(finite.size),
    }


def bootstrap_ratio_of_medians(
    df: pd.DataFrame,
    *,
    value: str,
    group: str,
    numerator: str,
    denominator: str,
    cluster: str = "patient",
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> dict[str, float]:
    """Cluster bootstrap of median(numerator group) / median(denominator group).

    A pure-numpy specialisation of the above for the compression factor, which is the
    statistic computed most often here. Each cluster's numerator and denominator values
    are extracted once as arrays, so a resample is a concatenate and two medians.
    """
    clusters = df[cluster].unique()
    num_by, den_by = {}, {}
    for c in clusters:
        sub = df[df[cluster] == c]
        num_by[c] = sub.loc[sub[group] == numerator, value].dropna().to_numpy(dtype=float)
        den_by[c] = sub.loc[sub[group] == denominator, value].dropna().to_numpy(dtype=float)

    def ratio(nums: list[np.ndarray], dens: list[np.ndarray]) -> float:
        n = np.concatenate(nums) if nums else np.empty(0)
        d = np.concatenate(dens) if dens else np.empty(0)
        if n.size == 0 or d.size == 0:
            return np.nan
        med_d = float(np.median(d))
        return float(np.median(n)) / med_d if med_d > 0 else np.nan

    point = ratio([num_by[c] for c in clusters], [den_by[c] for c in clusters])
    if len(clusters) < 2:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_clusters": len(clusters), "n_valid": 0}

    rng = np.random.default_rng(seed)
    values = np.full(n_resamples, np.nan)
    for k in range(n_resamples):
        picked = rng.choice(len(clusters), size=len(clusters), replace=True)
        chosen = [clusters[p] for p in picked]
        values[k] = ratio([num_by[c] for c in chosen], [den_by[c] for c in chosen])

    finite = values[np.isfinite(values)]
    alpha = (1.0 - level) / 2.0
    return {
        "point": point,
        "lo": float(np.percentile(finite, 100 * alpha)) if finite.size else float("nan"),
        "hi": float(np.percentile(finite, 100 * (1 - alpha))) if finite.size else float("nan"),
        "n_clusters": int(len(clusters)),
        "n_valid": int(finite.size),
    }


def spread(values: np.ndarray, measure: str) -> float:
    """Spread measures used for the estimands.

    SD is available but is never the primary: dose response is asymmetric and
    saturating, so a symmetric second moment misrepresents it.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    if measure == "iqr":
        q25, q75 = np.percentile(v, [25, 75])
        return float(q75 - q25)
    if measure == "p5_p95_range":
        p5, p95 = np.percentile(v, [5, 95])
        return float(p95 - p5)
    if measure == "range":
        return float(v.max() - v.min())
    if measure == "sd":
        return float(v.std(ddof=1))
    raise ValueError(f"unknown spread measure {measure!r}")
