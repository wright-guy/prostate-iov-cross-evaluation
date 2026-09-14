"""Variance decomposition of a cross-evaluation matrix.

What §7.3 wants is the proportion of variance attributable to the planning contour
(index i), the evaluation contour (index j), and their interaction. H5 asks specifically
whether the evaluation side contributes as much as or more than the planning side for
OAR endpoints.

**Why this is a two-way decomposition within patient rather than a crossed mixed model.**
The pre-registered plan names R/lme4 as the primary engine with statsmodels as a
cross-check. Neither R nor a trustworthy crossed random-effects implementation is
available in this environment: statsmodels fits crossed effects only through variance-
component formulas that are fragile on unbalanced data, and the design is unbalanced
(M030 has nine observers, and the diagonal is excluded from Arm S by design).

Within a patient, though, the Arm S cells form a near-complete two-way layout: every
plan against every truth except itself. A classical two-way decomposition of that layout
is exact, transparent, needs no optimiser, and answers H5 directly -- it is the ratio of
the plan-effect to the truth-effect sum of squares. Patients are then combined by the
cluster bootstrap used everywhere else.

The cost is that between-patient variance is not partitioned alongside the rest; it is
reported separately. For this question that is not a loss, since the comparison H5 makes
is between two effects measured within the same patient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def two_way_within_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose one endpoint's Arm S matrix per patient.

    Expects columns patient, plan_set, truth_set, value. Returns one row per patient
    with the sum of squares attributable to the plan effect (rows), the truth effect
    (columns) and the residual interaction, each as a share of the total.

    Missing cells -- the excluded diagonal -- are handled by using the unweighted mean
    of the cells that are present for each row and column effect. With nine of ten cells
    present in every row that approximation is slight, and it avoids inventing values
    for combinations the design deliberately excludes.
    """
    rows = []
    for patient, grp in df.groupby("patient"):
        wide = grp.pivot_table(index="plan_set", columns="truth_set", values="value")
        m = wide.to_numpy(dtype=float)
        if m.size == 0 or np.all(np.isnan(m)):
            continue
        grand = np.nanmean(m)
        row_mean = np.nanmean(m, axis=1)
        col_mean = np.nanmean(m, axis=0)
        present = ~np.isnan(m)
        n = int(present.sum())
        if n < 4:
            continue

        # Sums of squares over the cells that exist.
        ss_plan = float(np.nansum(present * (row_mean[:, None] - grand) ** 2))
        ss_truth = float(np.nansum(present * (col_mean[None, :] - grand) ** 2))
        fitted = row_mean[:, None] + col_mean[None, :] - grand
        ss_resid = float(np.nansum((m - fitted) ** 2))
        ss_total = float(np.nansum((m - grand) ** 2))
        if ss_total <= 0:
            continue

        rows.append({
            "patient": patient, "n_cells": n,
            "ss_plan": ss_plan, "ss_truth": ss_truth, "ss_interaction": ss_resid,
            "ss_total": ss_total,
            "pct_plan": 100.0 * ss_plan / ss_total,
            "pct_truth": 100.0 * ss_truth / ss_total,
            "pct_interaction": 100.0 * ss_resid / ss_total,
            "truth_over_plan": ss_truth / ss_plan if ss_plan > 0 else np.inf,
            "sd_total": float(np.sqrt(ss_total / max(n - 1, 1))),
        })
    return pd.DataFrame(rows)


def between_patient_share(df: pd.DataFrame) -> float:
    """Share of total variance sitting between patients rather than within them.

    Reported alongside, not inside, the within-patient decomposition. In a five-patient
    cohort this is usually the largest single component and would otherwise swamp the
    comparison H5 is about.
    """
    v = df["value"].dropna()
    if v.empty:
        return float("nan")
    grand = float(v.mean())
    means = df.groupby("patient")["value"].mean()
    counts = df.groupby("patient")["value"].count()
    ss_between = float((counts * (means - grand) ** 2).sum())
    ss_total = float(((v - grand) ** 2).sum())
    return 100.0 * ss_between / ss_total if ss_total > 0 else float("nan")


def mixed_model_crosscheck(df: pd.DataFrame) -> dict[str, float]:
    """Optional cross-check with a crossed variance-component model.

    Returns NaNs rather than raising if the fit does not converge, which on unbalanced
    crossed data it often will not. A disagreement with the two-way decomposition is
    reported, not silently resolved in favour of either.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"plan_var": np.nan, "truth_var": np.nan, "resid_var": np.nan,
                "converged": 0.0}

    data = df.dropna(subset=["value"]).copy()
    if data["patient"].nunique() < 2 or len(data) < 30:
        return {"plan_var": np.nan, "truth_var": np.nan, "resid_var": np.nan,
                "converged": 0.0}
    data["grp"] = 1
    try:
        model = smf.mixedlm(
            "value ~ 1", data, groups=data["patient"],
            vc_formula={"plan": "0 + C(plan_set)", "truth": "0 + C(truth_set)"},
        )
        fit = model.fit(reml=True, method="lbfgs", maxiter=200)
        vc = dict(fit.vcomp) if hasattr(fit, "vcomp") else {}
        keys = list(getattr(fit, "model", model).exog_vc.names) if hasattr(
            model, "exog_vc") else []
        values = list(fit.vcomp) if hasattr(fit, "vcomp") else []
        out = dict(zip(keys, values, strict=False)) if keys else vc
        return {
            "plan_var": float(out.get("plan", np.nan)),
            "truth_var": float(out.get("truth", np.nan)),
            "resid_var": float(fit.scale),
            "converged": float(bool(fit.converged)),
        }
    except Exception:  # noqa: BLE001 - a failed cross-check is reported, not fatal
        return {"plan_var": np.nan, "truth_var": np.nan, "resid_var": np.nan,
                "converged": 0.0}
