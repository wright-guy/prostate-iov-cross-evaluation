"""Stage 07b — Aim 1 statistics: compression factors with intervals, H2, tendencies.

    python pipelines/07b_aim1_statistics.py

Reads results/geometric_long.parquet, so it can be re-run without recomputing any
geometry. Produces the numbers that go into the paper for H1, H2 and the per-observer
tendency analysis, each with a cluster bootstrap over patients.

A note on which metrics carry the compression factor. For an overlap metric the factor
is computed on the DISAGREEMENT, (1 - metric), because agreement itself is bounded above
by 1 and a ratio of agreements is meaningless. That transform is unstable when the
metric approaches 1: bladder surface DSC is 0.97 against consensus and 0.99 between
observers, so the disagreements are 0.03 and 0.008 and their ratio is nearly 5 -- a
number driven by the last decimal place of a saturated metric, not by anatomy. Surface
DSC is therefore reported but excluded from the headline compression comparison, and the
exclusion is stated rather than quietly applied.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import load_config  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402
from mcx.stats.bootstrap import cluster_bootstrap  # noqa: E402

HEADLINE_METRICS = ("dsc", "msd_mm", "hd95_mm")
REPORTED_NOT_HEADLINE = ("surface_dice",)
HIGHER_IS_BETTER = {"dsc", "surface_dice", "jaccard"}

BENCHMARKS = {"uncorrected": 1.49, "loo_corrected": 1.34}
FAMILY = {"uncorrected": "obs_vote", "loo_corrected": "obs_loo"}

# Saturated overlap metrics make (1 - metric) unstable. Flagged, not silently dropped.
SATURATION_GUARD = 0.05


def compression_statistic(df: pd.DataFrame, metric: str, family: str) -> float:
    """Ratio of observer-to-observer disagreement to observer-to-consensus disagreement."""
    base = df.loc[df["family"] == "obs_obs", metric]
    comp = df.loc[df["family"] == family, metric]
    if base.empty or comp.empty:
        return float("nan")
    b, c = float(base.median()), float(comp.median())
    if metric in HIGHER_IS_BETTER:
        return (1 - b) / (1 - c) if (1 - c) > 0 else float("nan")
    return b / c if c > 0 else float("nan")


def main() -> int:
    cfg = load_config()
    run = RunInfo.create("07b_aim1_statistics", cfg)
    path = cfg.results_root / "geometric_long.parquet"
    if not path.exists():
        print("results/geometric_long.parquet not found — run 07_geometric_metrics.py first")
        return 2
    df = pd.read_parquet(path)
    boot = cfg.analysis_raw["uncertainty"]
    pd.set_option("display.width", 240)

    # ------------------------------------------------------------------ H1
    rows = []
    for structure in sorted(df["structure"].unique()):
        sub = df[df["structure"] == structure]
        for metric in (*HEADLINE_METRICS, *REPORTED_NOT_HEADLINE):
            if metric not in sub.columns:
                continue
            saturated = False
            if metric in HIGHER_IS_BETTER:
                worst = 1 - float(sub.loc[sub["family"] != "obs_obs", metric].median())
                saturated = worst < SATURATION_GUARD
            for label, family in FAMILY.items():
                if not (sub["family"] == family).any():
                    continue
                res = cluster_bootstrap(
                    sub, lambda d, m=metric, f=family: compression_statistic(d, m, f),
                    n_resamples=int(boot["n_resamples"]), seed=int(boot["seed"]),
                    level=float(boot["level"]),
                )
                rows.append({
                    "structure": structure, "metric": metric, "comparison": label,
                    "compression_factor": res["point"], "ci_lo": res["lo"],
                    "ci_hi": res["hi"], "benchmark": BENCHMARKS[label],
                    "benchmark_in_ci": bool(res["lo"] <= BENCHMARKS[label] <= res["hi"])
                    if np.isfinite(res["lo"]) and np.isfinite(res["hi"]) else False,
                    "headline": metric in HEADLINE_METRICS and not saturated,
                    "saturated": saturated,
                })
    h1 = pd.DataFrame(rows)
    write_table(h1, cfg.results_root / "aim1_compression.parquet", run)

    print("=== H1: compression factor, cluster bootstrap over patients (95% CI) ===")
    for label in FAMILY:
        sub = h1[(h1["comparison"] == label)]
        print(f"\n  {label} — benchmark {BENCHMARKS[label]}")
        show = sub.assign(
            factor=lambda d: d.apply(
                lambda r: f"{r.compression_factor:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
                + ("" if r.headline else "  (not headline)"), axis=1)
        )
        print(show.pivot_table(index="structure", columns="metric", values="factor",
                               aggfunc="first").to_string())

    # ------------------------------------------------------------------ H2
    rows = []
    for structure in sorted(df["structure"].unique()):
        for label, family in FAMILY.items():
            sub = df[(df["structure"] == structure) & (df["family"] == family)]
            if sub.empty:
                continue
            for field in ("signed_volume_difference", "signed_msd_mm"):
                res = cluster_bootstrap(
                    sub, lambda d, f=field: float(d[f].median()),
                    n_resamples=int(boot["n_resamples"]), seed=int(boot["seed"]),
                    level=float(boot["level"]),
                )
                rows.append({
                    "structure": structure, "reference": label, "quantity": field,
                    "median": res["point"], "ci_lo": res["lo"], "ci_hi": res["hi"],
                    "excludes_zero": bool(res["lo"] > 0 or res["hi"] < 0),
                })
    h2 = pd.DataFrame(rows)
    write_table(h2, cfg.results_root / "aim1_consensus_bias.parquet", run)

    print("\n\n=== H2: is the consensus unbiased? "
          "(positive = the observer is larger than the consensus) ===")
    for field in ("signed_volume_difference", "signed_msd_mm"):
        sub = h2[h2["quantity"] == field]
        print(f"\n  {field}")
        show = sub.assign(v=lambda d: d.apply(
            lambda r: f"{r['median']:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
            + ("  *" if r.excludes_zero else ""), axis=1))
        print(show.pivot_table(index="structure", columns="reference", values="v",
                               aggfunc="first").to_string())
    print("\n  * interval excludes zero")

    # ------------------------------------- observer tendency, and its correlation
    loo = df[df["family"] == "obs_loo"]
    tend = (loo.groupby(["set_a", "structure"])["signed_volume_difference"]
            .median().unstack())
    tend.index.name = "observer"

    pairs = []
    structures = [s for s in ("CTV", "PTV", "Rectum", "Bladder") if s in tend.columns]
    for a, b in [(x, y) for i, x in enumerate(structures) for y in structures[i + 1:]]:
        ok = tend[[a, b]].dropna()
        if len(ok) < 4:
            continue
        r, p = stats.pearsonr(ok[a], ok[b])
        rho, prho = stats.spearmanr(ok[a], ok[b])
        pairs.append({"structure_a": a, "structure_b": b, "n_observers": len(ok),
                      "pearson_r": r, "pearson_p": p,
                      "spearman_rho": rho, "spearman_p": prho})
    corr = pd.DataFrame(pairs)
    write_table(tend.reset_index(), cfg.results_root / "aim1_observer_tendency.parquet", run)
    if not corr.empty:
        write_table(corr, cfg.results_root / "aim1_tendency_correlation.parquet", run)

    print("\n\n=== observer tendency: median signed volume difference vs LOO consensus ===")
    print(tend.round(3).to_string())
    print("\n=== does an observer who contours a large target also contour a large OAR? ===")
    print("    (section 7.2; n = 10 observers, so these are indicative not confirmatory)")
    if not corr.empty:
        print(corr.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
