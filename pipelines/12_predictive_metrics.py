"""Stage 12 — Aim 3: can a contour-comparison metric predict dosimetric consequence?

    python pipelines/12_predictive_metrics.py

Builds the feature table joining each cell's contour-comparison metrics to its
dosimetric consequence, then evaluates the eight pre-specified primary metrics against
the two pre-specified primary endpoints under leave-one-patient-out validation.

H6 says unsigned overlap metrics cannot predict a SIGNED endpoint change while signed
metrics can. H7 says dose-aware, regionally restricted metrics beat whole-structure
geometric ones for OAR endpoints.

Two prediction targets, both reported:
  replanning benefit  endpoint(j, j) - endpoint(i, j)   non-circular, primary
  estimation error    endpoint(i, j) - endpoint(i, i)   computable at decision time

Correlations are never pooled across patients: inter-patient anatomy would dominate and
produce spuriously excellent numbers. Everything is within-patient or leave-one-patient-out.

Writes results/aim3_features.parquet and results/aim3_prediction.parquet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import Config, load_config  # noqa: E402
from mcx.dose.store import DoseStore  # noqa: E402
from mcx.geom.analysis_grid import load_grid  # noqa: E402
from mcx.geom.masks import MaskStore  # noqa: E402
from mcx.metrics.core import dice  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402

# Endpoint each structure's prediction is about, fixed in config/endpoints.yaml.
TARGETS = {"CTV": "ctv_d98", "Rectum": "ntcp_rectum_relative"}

# 90% of the prescription: the region where dose can actually respond.
HIGH_DOSE_FRACTION = 0.90


def dose_aware_features(cfg: Config, store: MaskStore, doses: DoseStore) -> pd.DataFrame:
    """Per-cell dose-aware metrics, which need the plan's dose as well as two contours."""
    high_dose = HIGH_DOSE_FRACTION * cfg.prescription_gy
    rows = []
    for patient in cfg.patients:
        t0 = time.perf_counter()
        grid = load_grid(cfg, patient)
        vox = grid.voxel_volume_cc
        plans = [s for s in cfg.observers if cfg.is_known_absent(patient, s) is None]
        for plan_set in plans:
            dose, covered = doses.get(patient, plan_set)
            hot = covered & (np.nan_to_num(dose, nan=-1.0) >= high_dose)
            for truth_set in plans:
                if truth_set == plan_set:
                    continue
                for structure in TARGETS:
                    if not (store.has(patient, plan_set, structure)
                            and store.has(patient, truth_set, structure)):
                        continue
                    a = store.get(patient, plan_set, structure)
                    b = store.get(patient, truth_set, structure)
                    fp = a & ~b & covered
                    fn = b & ~a & covered
                    rows.append({
                        "patient": patient, "plan_set": plan_set,
                        "truth_set": truth_set, "structure": structure,
                        "mean_dose_in_fp": float(dose[fp].mean()) if fp.any() else np.nan,
                        "mean_dose_in_fn": float(dose[fn].mean()) if fn.any() else np.nan,
                        "integral_dose_in_fp": float(dose[fp].sum()) * vox if fp.any() else 0.0,
                        "integral_dose_in_fn": float(dose[fn].sum()) * vox if fn.any() else 0.0,
                        "dsc_in_high_dose": dice(a & hot, b & hot),
                    })
            doses.drop_patient(patient)
        store.drop_patient(patient)
        print(f"  {patient}: {time.perf_counter() - t0:.0f}s")
    return pd.DataFrame(rows)


def build_features(cfg: Config, dose_aware: pd.DataFrame) -> pd.DataFrame:
    """Join geometry, dose-aware metrics and the two prediction targets."""
    r = cfg.results_root
    geo = pd.read_parquet(r / "geometric_long.parquet")
    geo = geo[geo["family"] == "obs_obs"].rename(
        columns={"set_a": "plan_set", "set_b": "truth_set"})
    keep = ["patient", "structure", "plan_set", "truth_set", "dsc", "hd95_mm", "msd_mm",
            "signed_volume_difference", "signed_msd_mm", "fp_volume_cc", "fn_volume_cc",
            "surface_dice", "apl_mm"]
    geo = geo[[c for c in keep if c in geo.columns]]

    ep = pd.read_parquet(r / "endpoints_long.parquet")
    bio = pd.read_parquet(r / "bio_long.parquet")
    ntcp = bio[(bio["model"] == "ntcp") & (bio["variant"] == "relative")].copy()
    ntcp["endpoint"] = "ntcp_" + ntcp["structure"].str.lower() + "_relative"
    values = pd.concat([
        ep[["patient", "plan_set", "truth_set", "endpoint", "value"]],
        ntcp[["patient", "plan_set", "truth_set", "endpoint", "value"]],
    ], ignore_index=True)

    lookup = values.set_index(["patient", "plan_set", "truth_set", "endpoint"])["value"]
    frames = []
    for structure, endpoint in TARGETS.items():
        sub = geo[geo["structure"] == structure].copy()
        if sub.empty:
            continue
        idx = list(zip(sub["patient"], sub["plan_set"], sub["truth_set"], strict=True))
        sub["endpoint"] = endpoint
        sub["value_ij"] = [lookup.get((p, i, j, endpoint), np.nan) for p, i, j in idx]
        sub["value_jj"] = [lookup.get((p, j, j, endpoint), np.nan) for p, _, j in idx]
        sub["value_ii"] = [lookup.get((p, i, i, endpoint), np.nan) for p, i, _ in idx]
        # Non-circular: what a replan would recover. Not available at decision time.
        sub["replanning_benefit"] = sub["value_jj"] - sub["value_ij"]
        # Computable given the dose and both contours; a screening tool only.
        sub["estimation_error"] = sub["value_ij"] - sub["value_ii"]
        frames.append(sub)

    feats = pd.concat(frames, ignore_index=True)
    return feats.merge(dose_aware,
                       on=["patient", "plan_set", "truth_set", "structure"], how="left")


def evaluate_prediction(feats: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Within-patient Spearman, plus leave-one-patient-out predictive R-squared."""
    rows = []
    for (structure, endpoint, target), grp in feats.groupby(
        ["structure", "endpoint"], sort=False
    ) if False else [
        ((s, e, t), g)
        for (s, e), g in feats.groupby(["structure", "endpoint"], sort=False)
        for t in ("replanning_benefit", "estimation_error")
    ]:
        for metric in metrics:
            if metric not in grp.columns:
                continue
            data = grp[["patient", metric, target]].dropna()
            if data["patient"].nunique() < 3 or len(data) < 20:
                continue

            # Within-patient rank correlation, pooled by Fisher z over patients.
            zs, ns = [], []
            for _, g in data.groupby("patient"):
                if len(g) < 5 or g[metric].nunique() < 3:
                    continue
                rho, _ = stats.spearmanr(g[metric], g[target])
                if np.isfinite(rho) and abs(rho) < 1:
                    zs.append(np.arctanh(rho))
                    ns.append(len(g) - 3)
            if not zs:
                continue
            z_bar = float(np.average(zs, weights=ns))
            se = float(np.sqrt(1.0 / np.sum(ns)))
            rho_pooled = float(np.tanh(z_bar))
            p = float(2 * (1 - stats.norm.cdf(abs(z_bar) / se)))

            # Leave-one-patient-out predictive R-squared from a simple linear fit.
            preds, truths = [], []
            for held in data["patient"].unique():
                train = data[data["patient"] != held]
                test = data[data["patient"] == held]
                if len(train) < 10 or test.empty or train[metric].nunique() < 3:
                    continue
                slope, intercept = np.polyfit(train[metric], train[target], 1)
                preds.extend(intercept + slope * test[metric])
                truths.extend(test[target])
            if len(truths) > 10:
                truths_a, preds_a = np.array(truths), np.array(preds)
                ss_res = float(((truths_a - preds_a) ** 2).sum())
                ss_tot = float(((truths_a - truths_a.mean()) ** 2).sum())
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            else:
                r2 = np.nan

            rows.append({
                "structure": structure, "endpoint": endpoint, "target": target,
                "metric": metric, "n_pairs": len(data),
                "n_patients": int(data["patient"].nunique()),
                "spearman_rho": rho_pooled, "p_value": p, "lopo_r2": r2,
            })
    return pd.DataFrame(rows)


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    thresholds = q * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    cutoff = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, dtype=bool)
    out[order[:cutoff]] = True
    return out


def main() -> int:
    cfg = load_config()
    run = RunInfo.create("12_predictive_metrics", cfg)
    store, doses = MaskStore(cfg), DoseStore(cfg)

    print("computing dose-aware features")
    dose_aware = dose_aware_features(cfg, store, doses)
    feats = build_features(cfg, dose_aware)
    write_table(feats, cfg.results_root / "aim3_features.parquet", run)
    print(f"\n{len(feats)} feature rows over {feats['structure'].nunique()} structures")

    primary = [m["id"] for m in cfg.metrics_raw["primary"]]
    alias = {"msd": "msd_mm", "hd95": "hd95_mm", "signed_msd": "signed_msd_mm",
             "mean_dose_in_fn": "mean_dose_in_fn", "mean_dose_in_fp": "mean_dose_in_fp"}
    metrics = [alias.get(m, m) for m in primary]
    classes = {alias.get(m["id"], m["id"]): m["class"] for m in cfg.metrics_raw["primary"]}

    pred = evaluate_prediction(feats, metrics)
    if pred.empty:
        print("no evaluable metric/endpoint combinations")
        return 1
    pred["metric_class"] = pred["metric"].map(classes)
    q = float(cfg.analysis_raw["prediction"]["multiplicity"]["q"])
    pred["significant_fdr"] = benjamini_hochberg(pred["p_value"].to_numpy(), q)
    write_table(pred, cfg.results_root / "aim3_prediction.parquet", run)

    pd.set_option("display.width", 240)
    for target in ("replanning_benefit", "estimation_error"):
        sub = pred[pred["target"] == target]
        if sub.empty:
            continue
        print(f"\n\n=== target: {target} ===")
        for endpoint, grp in sub.groupby("endpoint"):
            print(f"\n  {endpoint}")
            show = grp.sort_values("spearman_rho", key=abs, ascending=False)
            show = show.assign(
                rho=lambda d: d.apply(
                    lambda x: f"{x.spearman_rho:+.3f}"
                    + ("*" if x.significant_fdr else " "), axis=1),
                r2=lambda d: d["lopo_r2"].map(lambda v: f"{v:+.3f}"))
            print(show[["metric", "metric_class", "rho", "r2", "n_pairs"]]
                  .to_string(index=False))

    print("\n  * survives Benjamini-Hochberg at q = %.2f across the whole grid" % q)
    print("\n=== H6: unsigned against signed, by |rho| (replanning benefit) ===")
    rb = pred[pred["target"] == "replanning_benefit"]
    print(rb.groupby(["endpoint", "metric_class"])["spearman_rho"]
          .agg(best=lambda s: s.abs().max(), median=lambda s: s.abs().median())
          .round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
