"""Section 2.3 -- fixed evaluation contour, varied dose.

Read down a column of the grid: the anatomy is assumed known, and the plan varies
according to whose contour it was optimised on. The consensus supplies the dose here,
which it never did in Section 2.2.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from mcx.config import load_config

cfg = load_config()
r = cfg.results_root
pd.set_option("display.width", 260)

ep = pd.read_parquet(r / "endpoints_long.parquet")
bio = pd.read_parquet(r / "bio_long.parquet")
bio = bio[
    ((bio.model == "ntcp") & (bio.variant == "relative"))
    | ((bio.model == "tcp") & (bio.variant == "fixed_density")
       & (bio.density_label == "calibrated"))
].copy()
bio["endpoint"] = bio["endpoint_id"]
bio["value"] = bio["value"] * 100.0
allep = pd.concat(
    [ep[["patient", "plan_set", "truth_set", "arm", "endpoint", "value"]],
     bio[["patient", "plan_set", "truth_set", "arm", "endpoint", "value"]]],
    ignore_index=True)

HEAD = ["ctv_d98", "ctv_d50", "ctv_v95pct", "paddick_ci", "rectum_v70pct",
        "rectum_dmean", "rectum_d2cc", "bladder_v70pct", "bladder_dmean",
        "rectum_late_bleeding", "tcp"]

obs = allep[allep.plan_set.isin(cfg.observers)]
vote = allep[allep.plan_set == "VOTE"].set_index(
    ["patient", "truth_set", "endpoint"])["value"]

# ---- A. raw column spread: observer plans only, i includes j.
recs = []
for (patient, j, e), g in obs.groupby(["patient", "truth_set", "endpoint"]):
    v = g["value"].to_numpy(float)
    if v.size < 5:
        continue
    self_row = g[g.plan_set == j]["value"]
    if not len(self_row):
        continue
    recs.append({"patient": patient, "truth_set": j, "endpoint": e, "n_i": v.size,
                 "median": float(np.median(v)),
                 "range": float(v.max() - v.min()),
                 "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
                 "sd": float(v.std(ddof=1)),
                 "self": float(self_row.iloc[0])})
col = pd.DataFrame(recs)
col["range_pct"] = np.where(col["median"].abs() > 1e-9,
                            100 * col["range"] / col["median"].abs(), np.nan)
col.to_parquet(r / "sec23_col_iov.parquet", index=False)

print("=== A. within-column spread across the 10 observer plans (n = 49 columns) ===")
t = col[col.endpoint.isin(HEAD)].groupby("endpoint").agg(
    n=("range", "count"), range_med=("range", "median"),
    range_q1=("range", lambda s: s.quantile(.25)),
    range_q3=("range", lambda s: s.quantile(.75)),
    range_max=("range", "max"), sd_med=("sd", "median"),
    relrange=("range_pct", "median")).reindex(HEAD)
print(t.round(3).to_string())

# ---- B. penalty relative to the correct plan, and the consensus penalty.
recs = []
for (patient, j, e), g in obs.groupby(["patient", "truth_set", "endpoint"]):
    self_row = g[g.plan_set == j]["value"]
    if not len(self_row):
        continue
    sv = float(self_row.iloc[0])
    others = g[g.plan_set != j]["value"].to_numpy(float)
    if others.size < 4:
        continue
    key = (patient, j, e)
    bv = float(vote.loc[key]) if key in vote.index else np.nan
    recs.append({"patient": patient, "truth_set": j, "endpoint": e, "self": sv,
                 "A": float(np.mean(np.abs(others - sv))),
                 "A_signed": float(np.mean(others - sv)),
                 "B": abs(bv - sv) if np.isfinite(bv) else np.nan,
                 "B_signed": (bv - sv) if np.isfinite(bv) else np.nan,
                 "E_vote": bv})
per = pd.DataFrame(recs)
per.to_parquet(r / "sec23_per_column.parquet", index=False)

print("\n=== B. penalty for planning on the wrong anatomy (n = 49 columns) ===")
u = per[per.endpoint.isin(HEAD)].groupby("endpoint").agg(
    n=("A", "count"), A_med=("A", "median"),
    A_q1=("A", lambda s: s.quantile(.25)), A_q3=("A", lambda s: s.quantile(.75)),
    A_signed=("A_signed", "median"), B_med=("B", "median"),
    B_signed=("B_signed", "median")).reindex(HEAD)
u["R_mean"] = per.groupby("endpoint").apply(
    lambda g: g.A.mean() / g.B.mean() if g.B.mean() > 0 else np.nan,
    include_groups=False).reindex(HEAD)
u["R_median"] = u.A_med / u.B_med
u["frac_B_lt_A"] = per.groupby("endpoint").apply(
    lambda g: float((g.B < g.A).mean()), include_groups=False).reindex(HEAD)
print(u.round(4).to_string())

# ---- C. cluster bootstrap over patients for R_mean.
rng = np.random.default_rng(20260904)
patients = sorted(per.patient.unique())
print("\n=== C. bootstrap CI for R_2.3 (mean-based) ===")
out = []
for e in HEAD:
    g = per[per.endpoint == e].dropna(subset=["A", "B"])
    if g.empty:
        continue
    pt = g.A.mean() / g.B.mean() if g.B.mean() > 0 else np.nan
    boots = []
    for _ in range(10000):
        pick = rng.choice(len(patients), size=len(patients), replace=True)
        sub = pd.concat([g[g.patient == patients[p]] for p in pick])
        boots.append(sub.A.mean() / sub.B.mean() if sub.B.mean() > 0 else np.nan)
    boots = np.array([b for b in boots if np.isfinite(b)])
    out.append({"endpoint": e, "R": pt,
                "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5)})
boot = pd.DataFrame(out).set_index("endpoint")
print(boot.round(3).to_string())
boot.to_parquet(r / "sec23_compression.parquet")

print("\n=== D. per-patient R_2.3 (mean-based) ===")
pp = per.groupby(["endpoint", "patient"]).apply(
    lambda g: g.A.mean() / g.B.mean() if g.B.mean() > 0 else np.nan,
    include_groups=False).unstack()
print(pp.reindex(HEAD).round(2).to_string())

print("\n=== E. worst column per endpoint (raw spread) ===")
for e in ["rectum_v70pct", "rectum_dmean", "bladder_dmean", "rectum_late_bleeding",
          "ctv_d98"]:
    g = col[col.endpoint == e]
    w = g.loc[g["range"].idxmax()]
    sub = obs[(obs.patient == w.patient) & (obs.truth_set == w.truth_set)
              & (obs.endpoint == e)]["value"]
    vv = vote.get((w.patient, w.truth_set, e), np.nan)
    print(f"  {e:22s} {w.patient}, truth {w.truth_set}: {sub.min():.2f} to "
          f"{sub.max():.2f} (correct plan {w['self']:.2f}, consensus plan {vv:.2f})")
