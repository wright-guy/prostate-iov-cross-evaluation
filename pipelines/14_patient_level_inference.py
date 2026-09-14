"""Stage 14 — patient-level inference, operator-specific benchmarks, and the controls
added in response to external review (2026-09-11).

    python pipelines/14_patient_level_inference.py

Why this stage exists
---------------------
Three problems with the earlier inference, each raised in review and each confirmed:

1. Five patients give only 126 distinct cluster-bootstrap resamples, and percentile
   intervals from five clusters under-cover. Every interval is recomputed here from
   per-patient estimates: ratios on the log scale (geometric mean, t with 4 df),
   differences on the natural scale (mean, t with 4 df).

2. The 1.34 / 1.49 benchmarks assume the consensus is the MEAN of the observers. A
   strict-majority vote is an order statistic. The benchmark for each estimator is
   simulated here for the operator actually used (5 of 9 or 5 of 8 leave-one-out;
   6 of 10 or 5 of 9 full) and for the exact estimator applied to the data, including
   its small-sample behaviour with ten observers per patient.

3. Strict majority was the only voting rule examined. A 5-of-10 consensus is built
   here from the same masks as a control on the direction of the consensus bias.

Plan-to-plan noise in Section 2.3 is not modelled -- the design has one plan per
contour set, so it cannot be -- but the noise share that would be needed to reconcile
each observed R_2.3 with its benchmark is computed. It is not compared with the
varied-plan share of the decomposition table, which mixes the planning-contour effect with
plan-to-plan variation and so bounds neither.

Reads results/ and the mask cache. Writes results/inference_*.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import load_config  # noqa: E402
from mcx.geom.analysis_grid import load_grid  # noqa: E402
from mcx.geom.consensus import majority_vote  # noqa: E402
from mcx.geom.masks import MaskStore  # noqa: E402
from mcx.metrics.core import compare  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402

SEED = 20260904
NULL_REPS = 60_000
T_LEVEL = 0.95

S2_ENDPOINTS = ["ctv_d98", "ctv_d50", "paddick_ci", "rectum_v70pct", "rectum_dmean",
                "rectum_d2cc", "bladder_v70pct", "bladder_dmean"]
S23_ENDPOINTS = [*S2_ENDPOINTS, "rectum_late_bleeding"]
S1_METRICS = {"dsc": True, "hd95_mm": False, "msd_mm": False}   # True = agreement metric
STRUCTURES = ["CTV", "PTV", "Rectum", "Bladder"]


# --------------------------------------------------------------------- intervals


def t_interval_log(values) -> tuple[float, float, float]:
    """Geometric mean of per-patient ratios, with a t interval on the log scale."""
    v = np.log(np.asarray([x for x in values if np.isfinite(x) and x > 0], float))
    if v.size < 2:
        return float("nan"), float("nan"), float("nan")
    q = stats.t.ppf(0.5 + T_LEVEL / 2, v.size - 1)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
    return float(np.exp(m)), float(np.exp(m - q * se)), float(np.exp(m + q * se))


def t_interval(values) -> tuple[float, float, float]:
    """Mean of per-patient values, with a t interval."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size < 2:
        return float("nan"), float("nan"), float("nan")
    q = stats.t.ppf(0.5 + T_LEVEL / 2, v.size - 1)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
    return float(m), float(m - q * se), float(m + q * se)


# --------------------------------------------------------------------- benchmarks


def _tth_largest(x: np.ndarray, t: int) -> np.ndarray:
    return np.sort(x, axis=1)[:, -t]


def simulate_patient(n_obs: int, reps: int, rng) -> dict[str, np.ndarray]:
    """Per-patient estimators under the null: observer effects iid N(0, 1), no bias.

    A point is inside observer k's contour if r < x_k; a consensus needing T votes has
    its boundary at the T-th largest x_k. Returns log-ratios for every estimator.
    """
    x = rng.normal(size=(reps, n_obs))
    t_full, t_loo = n_obs // 2 + 1, (n_obs - 1) // 2 + 1
    full = _tth_largest(x, t_full)
    iu = np.triu_indices(n_obs, 1)
    pair = np.abs(x[:, iu[0]] - x[:, iu[1]])                     # unordered pairs
    loo = np.empty_like(x)
    a = np.empty_like(x)
    for i in range(n_obs):
        others = np.delete(x, i, axis=1)
        loo[:, i] = _tth_largest(others, t_loo)
        a[:, i] = np.abs(others - x[:, [i]]).mean(axis=1)
    b_loo, b_full = np.abs(loo - x), np.abs(full[:, None] - x)
    med = np.median
    return {
        # Per-patient estimators with ten observer-level values, as in Sections 2.2 and
        # 2.3, where each observer contributes one endpoint value. A is already a mean.
        "s2_mean_loo": np.log(a.mean(axis=1) / b_loo.mean(axis=1)),
        "s2_median_loo": np.log(med(a, axis=1) / med(b_loo, axis=1)),
        "s2_mean_full": np.log(a.mean(axis=1) / b_full.mean(axis=1)),
        "s2_median_full": np.log(med(a, axis=1) / med(b_full, axis=1)),
        # Population ratio, pooled over every replicate: the large-sample benchmark.
        # Section 1 statistics aggregate whole surfaces over 90 pairs per patient and
        # their per-patient values sit within a few percent of each other, so it is the
        # population value, not the ten-observer estimator, that applies there.
        "_pop_loo": (pair.mean(), b_loo.mean()),
        "_pop_full": (pair.mean(), b_full.mean()),
        "bias_full_sd": full,
    }


def benchmarks(cfg) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    sizes = [len([s for s in cfg.observers if cfg.is_known_absent(p, s) is None])
             for p in cfg.patients]
    sims = {n: simulate_patient(n, NULL_REPS, rng) for n in sorted(set(sizes))}
    rows = []
    for key in ("_pop_loo", "_pop_full"):
        per_n = {n: np.log(sims[n][key][0] / sims[n][key][1]) for n in sims}
        for n, v in per_n.items():
            rows.append({"estimator": f"population{key[4:]}_n{n}", "benchmark": float(np.exp(v)),
                         "null_lo": np.nan, "null_hi": np.nan})
        rows.append({"estimator": f"population{key[4:]}",
                     "benchmark": float(np.exp(np.mean([per_n[n] for n in sizes]))),
                     "null_lo": np.nan, "null_hi": np.nan})
    for key in ("s2_mean_loo", "s2_median_loo", "s2_mean_full", "s2_median_full"):
        # Geometric mean over the cohort's patients, each with its own observer count.
        cohort = np.mean([sims[n][key] for n in sizes], axis=0)
        rows.append({"estimator": key, "benchmark": float(np.exp(cohort.mean())),
                     "null_lo": float(np.exp(np.percentile(cohort, 2.5))),
                     "null_hi": float(np.exp(np.percentile(cohort, 97.5)))})
    for n in sorted(set(sizes)):
        rows.append({"estimator": f"full_vote_bias_sd_n{n}",
                     "benchmark": float(sims[n]["bias_full_sd"].mean()),
                     "null_lo": np.nan, "null_hi": np.nan})
    # The idealised mean-of-observers values the text used previously.
    for n in sorted(set(sizes)):
        rows.append({"estimator": f"mean_operator_loo_n{n}",
                     "benchmark": float(np.sqrt(2) / np.sqrt(n / (n - 1))),
                     "null_lo": np.nan, "null_hi": np.nan})
        rows.append({"estimator": f"mean_operator_full_n{n}",
                     "benchmark": float(np.sqrt(2) / np.sqrt((n - 1) / n)),
                     "null_lo": np.nan, "null_hi": np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- Section 1


def section1(cfg, geo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp_rows, bias_rows = [], []
    for structure in STRUCTURES:
        for family, label in (("obs_loo", "loo"), ("obs_vote", "full")):
            for metric, agreement in S1_METRICS.items():
                per = []
                for p in cfg.patients:
                    g = geo[(geo.patient == p) & (geo.structure == structure)]
                    base = g.loc[g.family == "obs_obs", metric].median()
                    cons = g.loc[g.family == family, metric].median()
                    per.append((1 - base) / (1 - cons) if agreement else base / cons)
                m, lo, hi = t_interval_log(per)
                comp_rows.append({"structure": structure, "reference": label,
                                  "metric": metric, "estimate": m, "lo": lo, "hi": hi,
                                  **{f"patient_{p}": v for p, v in zip(cfg.patients, per,
                                                                       strict=True)}})
            for field in ("signed_msd_mm", "signed_volume_difference"):
                per = [geo[(geo.patient == p) & (geo.structure == structure)
                           & (geo.family == family)][field].median() for p in cfg.patients]
                m, lo, hi = t_interval(per)
                bias_rows.append({"structure": structure, "reference": label,
                                  "quantity": field, "estimate": m, "lo": lo, "hi": hi})
    return pd.DataFrame(comp_rows), pd.DataFrame(bias_rows)


def threshold_control(cfg) -> pd.DataFrame:
    """Observer-to-consensus bias for 6-of-10 and 5-of-10 consensus contours."""
    store = MaskStore(cfg)
    rows = []
    for p in cfg.patients:
        observers = [s for s in cfg.observers if cfg.is_known_absent(p, s) is None]
        if len(observers) != 10:
            continue            # the control needs an even observer count
        grid = load_grid(cfg, p)
        for structure in STRUCTURES:
            masks = {s: store.get(p, s, structure) for s in observers
                     if store.has(p, s, structure)}
            for k in (5, 6):
                cons = majority_vote(list(masks.values()), threshold=k)
                for s, m in masks.items():
                    out = compare(m, cons, grid)
                    rows.append({"patient": p, "structure": structure,
                                 "threshold": f"{k}/10", "observer": s,
                                 "signed_msd_mm": out["signed_msd_mm"],
                                 "signed_volume_difference": out["signed_volume_difference"],
                                 "dsc": out["dsc"]})
        store.drop_patient(p)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- Section 2


def section2(cfg, per_plan: pd.DataFrame, per_col: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    s22, s23, signed = [], [], []
    for e in S2_ENDPOINTS:
        g = per_plan[per_plan.endpoint == e]
        for den, label in (("B_LOO", "loo"), ("B_VOTE", "full")):
            for est, fn in (("mean", np.mean), ("median", np.median)):
                per = []
                for p in cfg.patients:
                    gp = g[g.patient == p]
                    b = fn(gp[den])
                    per.append(fn(gp.A) / b if b > 0 else np.nan)
                m, lo, hi = t_interval_log(per)
                s22.append({"endpoint": e, "reference": label, "estimator": est,
                            "estimate": m, "lo": lo, "hi": hi,
                            "n_patients_finite": int(np.isfinite(per).sum())})
    for e in S23_ENDPOINTS:
        g = per_col[per_col.endpoint == e]
        for est, fn in (("mean", np.mean), ("median", np.median)):
            per = []
            for p in cfg.patients:
                gp = g[g.patient == p]
                b = fn(gp.B)
                per.append(fn(gp.A) / b if b > 0 else np.nan)
            m, lo, hi = t_interval_log(per)
            s23.append({"endpoint": e, "estimator": est, "estimate": m, "lo": lo, "hi": hi,
                        **{f"patient_{p}": v for p, v in zip(cfg.patients, per, strict=True)}})
        for col, who in (("A_signed", "observers"), ("B_signed", "consensus")):
            per = [g[g.patient == p][col].mean() for p in cfg.patients]
            m, lo, hi = t_interval(per)
            signed.append({"endpoint": e, "plans": who, "estimate": m, "lo": lo, "hi": hi,
                           "n_patients_positive": int(np.sum(np.asarray(per) > 0)),
                           **{f"patient_{p}": v for p, v in zip(cfg.patients, per,
                                                                strict=True)}})
    return pd.DataFrame(s22), pd.DataFrame(s23), pd.DataFrame(signed)


def noise_share(s23: pd.DataFrame, bench: float) -> pd.DataFrame:
    """Noise variance, as a share of the numerator, that alone would pull the benchmark
    down to the observed value: R^2 = (1 + n) / (1/B0^2 + n)."""
    rows = []
    for _, r in s23[s23.estimator == "mean"].iterrows():
        R = r.estimate
        n = (1 - R**2 / bench**2) / (R**2 - 1) if R > 1 else np.inf
        rows.append({"endpoint": r.endpoint, "observed": R, "benchmark": bench,
                     "noise_share_needed": n / (1 + n) if np.isfinite(n) and n > 0 else
                     (0.0 if R >= bench else 1.0)})
    return pd.DataFrame(rows)


def volume_test(cfg, results: Path) -> pd.DataFrame:
    """Does the consensus plan's rectal penalty follow from its smaller rectum?

    Fit the observer plans' signed penalty against (true rectum volume - planning
    rectum volume) and compare the consensus plan with what that line predicts for its
    own volume deficit.
    """
    store = MaskStore(cfg)
    vol = {}
    for p in cfg.patients:
        vox = load_grid(cfg, p).voxel_volume_cc
        for s in [*cfg.observers, "VOTE"]:
            if store.has(p, s, "Rectum"):
                vol[(p, s)] = float(store.get(p, s, "Rectum").sum()) * vox
        store.drop_patient(p)

    ep = pd.read_parquet(results / "endpoints_long.parquet")
    bio = pd.read_parquet(results / "bio_long.parquet")
    bio = bio[(bio.model == "ntcp") & (bio.variant == "relative")
              & (bio.endpoint_id == "rectum_late_bleeding")].copy()
    bio["endpoint"], bio["value"] = "rectum_late_bleeding", bio["value"] * 100
    cols = ["patient", "plan_set", "truth_set", "endpoint", "value"]
    allv = pd.concat([ep[cols], bio[cols]], ignore_index=True)

    rows = []
    for e in ("rectum_d2cc", "rectum_late_bleeding", "rectum_dmean", "rectum_v70pct"):
        d = allv[allv.endpoint == e].copy()
        diag = d[d.plan_set == d.truth_set].set_index(["patient", "truth_set"])["value"]
        d = d[(d.plan_set != d.truth_set) & d.truth_set.isin(cfg.observers)]
        d["penalty"] = d["value"].to_numpy() - diag.reindex(
            list(zip(d.patient, d.truth_set, strict=True))).to_numpy()
        d["deficit_cc"] = [vol[(p, j)] - vol[(p, i)]
                           for p, i, j in zip(d.patient, d.plan_set, d.truth_set, strict=True)]
        obs, cons = d[d.plan_set != "VOTE"], d[d.plan_set == "VOTE"]
        fit = stats.linregress(obs.deficit_cc, obs.penalty)
        predicted = fit.intercept + fit.slope * cons.deficit_cc
        rows.append({"endpoint": e, "slope_per_cc": fit.slope, "r": fit.rvalue,
                     "p": fit.pvalue, "consensus_deficit_cc": cons.deficit_cc.mean(),
                     "consensus_penalty": cons.penalty.mean(),
                     "predicted_from_volume": float(predicted.mean()),
                     "excess": float((cons.penalty - predicted).mean()),
                     "observer_residual_sd": float(np.std(
                         obs.penalty - (fit.intercept + fit.slope * obs.deficit_cc), ddof=2))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main


def main() -> int:
    cfg = load_config()
    run = RunInfo.create("14_patient_level_inference", cfg)
    r = cfg.results_root
    pd.set_option("display.width", 250)

    print("simulating operator-specific benchmarks ...")
    bench = benchmarks(cfg)
    write_table(bench, r / "inference_benchmarks.parquet", run)
    print(bench.round(3).to_string(index=False))
    b = bench.set_index("estimator")["benchmark"]

    geo = pd.read_parquet(r / "geometric_long.parquet")
    comp, bias = section1(cfg, geo)
    write_table(comp, r / "inference_s1_compression.parquet", run)
    write_table(bias, r / "inference_s1_bias.parquet", run)
    print("\n=== Section 1 compression, patient level ===")
    print(comp[["structure", "reference", "metric", "estimate", "lo", "hi"]]
          .round(3).to_string(index=False))
    print("\n=== Section 1 bias, patient level ===")
    print(bias.round(4).to_string(index=False))

    print("\nbuilding 5-of-10 and 6-of-10 consensus contours ...")
    ctrl = threshold_control(cfg)
    write_table(ctrl, r / "inference_s1_threshold_control.parquet", run)
    summary = (ctrl.groupby(["structure", "threshold", "patient"])
               [["signed_msd_mm", "signed_volume_difference"]].median().reset_index())
    rows = []
    for (structure, thr), g in summary.groupby(["structure", "threshold"]):
        for q in ("signed_msd_mm", "signed_volume_difference"):
            m, lo, hi = t_interval(g[q])
            rows.append({"structure": structure, "threshold": thr, "quantity": q,
                         "estimate": m, "lo": lo, "hi": hi})
    ctrl_sum = pd.DataFrame(rows)
    write_table(ctrl_sum, r / "inference_s1_threshold_summary.parquet", run)
    print(ctrl_sum.round(4).to_string(index=False))

    per_plan = pd.read_parquet(r / "sec22_per_plan.parquet")
    per_col = pd.read_parquet(r / "sec23_per_column.parquet")
    s22, s23, signed = section2(cfg, per_plan, per_col)
    write_table(s22, r / "inference_s22.parquet", run)
    write_table(s23, r / "inference_s23.parquet", run)
    write_table(signed, r / "inference_s23_signed.parquet", run)
    print("\n=== Section 2.2 ===")
    print(s22.round(3).to_string(index=False))
    print("\n=== Section 2.3 ===")
    print(s23[["endpoint", "estimator", "estimate", "lo", "hi"]].round(3).to_string(index=False))
    print("\n=== Section 2.3 signed penalties ===")
    print(signed[["endpoint", "plans", "estimate", "lo", "hi", "n_patients_positive"]]
          .round(3).to_string(index=False))

    ns = noise_share(s23, float(b["population_full"]))
    write_table(ns, r / "inference_s23_noise_share.parquet", run)
    print("\n=== Section 2.3 noise share needed ===")
    print(ns.round(3).to_string(index=False))

    vt = volume_test(cfg, r)
    write_table(vt, r / "inference_s23_volume_test.parquet", run)
    print("\n=== Section 2.3 volume test ===")
    print(vt.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
