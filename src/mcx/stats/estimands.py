"""The three estimands, and the compression factor built from them.

"Variability" is not one quantity, and conflating the three is the objection this
analysis exists to pre-empt.

E1  Delivery variability   spread across planning contours i, at a FIXED truth j.
                           Identically zero in Arm C, because there is one plan. That is
                           a definitional artefact, so it is surfaced here rather than
                           left for a reviewer to find.

E2  Residual truth         spread across plausible truths j, for a FIXED plan i.
    uncertainty            Non-zero in both arms. This is what a consensus contour
                           cannot remove, and what the compression factor compares.

E3  Expected systematic    E[endpoint(i, j)] - E[endpoint(j, j)]: the average penalty
    loss                   for having planned on the wrong anatomy, measured against the
                           perfect-information diagonal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mcx.stats.bootstrap import spread


def e1_delivery(df: pd.DataFrame, measure: str = "iqr") -> pd.DataFrame:
    """Spread across plans, at fixed truth. One row per (patient, truth, arm)."""
    rows = []
    for (patient, truth_set, arm), grp in df.groupby(["patient", "truth_set", "arm"]):
        if arm not in ("S", "C"):
            continue
        rows.append({
            "patient": patient, "truth_set": truth_set, "arm": arm,
            "n_plans": int(grp["plan_set"].nunique()),
            "e1": spread(grp["value"].to_numpy(), measure),
            "median": float(grp["value"].median()),
        })
    return pd.DataFrame(rows)


def e2_truth_uncertainty(df: pd.DataFrame, measure: str = "iqr") -> pd.DataFrame:
    """Spread across truths, at fixed plan. One row per (patient, plan, arm)."""
    rows = []
    for (patient, plan_set, arm), grp in df.groupby(["patient", "plan_set", "arm"]):
        if arm not in ("S", "C"):
            continue
        rows.append({
            "patient": patient, "plan_set": plan_set, "arm": arm,
            "n_truths": int(grp["truth_set"].nunique()),
            "e2": spread(grp["value"].to_numpy(), measure),
            "median": float(grp["value"].median()),
        })
    return pd.DataFrame(rows)


def e3_systematic_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Mean penalty against the perfect-information diagonal, per patient and arm."""
    diagonal = (df[df["arm"] == "diagonal"]
                .set_index(["patient", "truth_set"])["value"])
    rows = []
    for (patient, arm), grp in df.groupby(["patient", "arm"]):
        if arm not in ("S", "C"):
            continue
        ref = np.array([diagonal.get((patient, j), np.nan) for j in grp["truth_set"]])
        delta = grp["value"].to_numpy() - ref
        finite = delta[np.isfinite(delta)]
        if finite.size == 0:
            continue
        rows.append({
            "patient": patient, "arm": arm, "n_cells": int(finite.size),
            "e3_mean": float(finite.mean()), "e3_median": float(np.median(finite)),
            "e3_p05": float(np.percentile(finite, 5)),
            "e3_p95": float(np.percentile(finite, 95)),
            "e3_worst": float(finite[np.argmax(np.abs(finite))]),
        })
    return pd.DataFrame(rows)


def planning_error(df: pd.DataFrame) -> pd.DataFrame:
    """endpoint(i, j) - endpoint(j, j): the error from having planned on the wrong anatomy.

    This is the quantity structurally analogous to the geometric benchmark. The 1.34 and
    1.49 figures come from a variance identity on the DISTANCE between a contour and its
    reference -- SD(x_i - x_j) against SD(x_i - x_bar) -- so the dosimetric analogue is
    the spread of the per-cell error against the perfect-information diagonal, not the
    spread of the endpoint itself.

    Comparing the endpoint's own spread across truths (E2) between arms answers a
    different and also useful question, but it is dominated by how much the ten truths
    differ from each other, which barely depends on which plan is being scored. Both are
    reported; this one is the like-for-like comparison with the geometric factor.
    """
    diagonal = (df[df["arm"] == "diagonal"]
                .set_index(["patient", "truth_set"])["value"])
    sub = df[df["arm"].isin(["S", "C"])].copy()
    ref = [diagonal.get((p, j), np.nan)
           for p, j in zip(sub["patient"], sub["truth_set"], strict=True)]
    sub["error"] = sub["value"].to_numpy() - np.asarray(ref, dtype=float)
    return sub.dropna(subset=["error"])


def error_spread(df: pd.DataFrame, measure: str = "iqr") -> pd.DataFrame:
    """Spread of the planning error, PER PLAN, then comparable across arms.

    Grouping by plan rather than pooling the whole arm matters. Pooling all of Arm S
    mixes two sources of variation: how much the error moves as the assumed truth
    changes, and how much plan quality differs between observers. Arm C has one plan, so
    it carries only the first. Pooling would therefore inflate Arm S and manufacture a
    compression factor out of plan quality -- something the geometric benchmark has no
    analogue for, since contours have no equivalent of a good or bad plan.

    Taking one spread per plan and comparing medians puts a typical single-observer plan
    against the consensus plan, which is the comparison the study is actually about.
    """
    err = planning_error(df)
    rows = []
    for (patient, arm, plan_set), grp in err.groupby(["patient", "arm", "plan_set"]):
        rows.append({"patient": patient, "arm": arm, "plan_set": plan_set,
                     "n_cells": len(grp),
                     "error_spread": spread(grp["error"].to_numpy(), measure),
                     "error_median": float(grp["error"].median())})
    return pd.DataFrame(rows)


def compression_factor(df: pd.DataFrame, measure: str = "iqr") -> float:
    """spread(Arm S) / spread(Arm C), on E2-comparable terms.

    E2 is the only estimand on which the two arms are comparable: E1 is zero in Arm C by
    construction, so a ratio built on it would be infinite and meaningless.

    Arm S contributes one E2 per observer plan; those are summarised by their median
    before the ratio, so a single unusually variable plan cannot drive the result.
    """
    return compression_from_e2(e2_truth_uncertainty(df, measure))


def compression_from_e2(e2: pd.DataFrame) -> float:
    """The same ratio, from an already-computed E2 table.

    Separated out because the cluster bootstrap resamples patients ten thousand times
    per endpoint. Recomputing E2 inside the loop means a pandas groupby over the cell
    table on every resample -- about a million of them across the endpoint list, which
    is minutes per endpoint. E2 does not depend on which patients are drawn, only on
    which of its own rows are selected, so it is computed once and the bootstrap
    resamples the summary. The result is identical; it is three orders of magnitude
    faster.
    """
    s = e2.loc[e2["arm"] == "S", "e2"].dropna()
    c = e2.loc[e2["arm"] == "C", "e2"].dropna()
    if s.empty or c.empty:
        return float("nan")
    denominator = float(c.median())
    return float(s.median()) / denominator if denominator > 0 else float("nan")


def tail_compression(df: pd.DataFrame) -> dict[str, float]:
    """Compression in the tails, which is what H4 is about.

    Consensus regularises the centre of a distribution more readily than its extremes,
    so a factor computed on the IQR can look healthy while the tail barely moves.
    """
    return {
        f"compression_{measure}": compression_factor(df, measure)
        for measure in ("iqr", "p5_p95_range", "range", "sd")
    }
