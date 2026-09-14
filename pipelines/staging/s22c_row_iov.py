"""Raw interobserver variation in dose, Section 2.2.

Fix a delivered plan i; the assumed true anatomy j runs over all observer contours.
The spread of E(i,j) across j is the uncertainty about what that plan delivered.
Reported before any ratio is taken.
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

# Biological endpoints, expressed as percentage points, on the same long schema.
bio = bio[
    ((bio.model == "ntcp") & (bio.variant == "relative"))
    | ((bio.model == "tcp") & (bio.variant == "fixed_density")
       & (bio.density_label == "calibrated"))
].copy()
bio["endpoint"] = bio["endpoint_id"]
bio["value"] = bio["value"] * 100.0
bio = bio[["patient", "plan_set", "truth_set", "arm", "endpoint", "value"]]

allep = pd.concat([ep[bio.columns], bio], ignore_index=True)

# Row = one delivered plan; j ranges over every observer contour, i's own included.
rows_of_grid = allep[allep.arm.isin(["S", "diagonal"])]
rows_of_grid = rows_of_grid[rows_of_grid.plan_set.isin(cfg.observers)]

HEAD = ["ctv_d98", "ctv_d50", "ctv_v95pct", "rectum_v70pct", "rectum_dmean",
        "rectum_d2cc", "bladder_v70pct", "bladder_dmean", "paddick_ci",
        "rectum_late_bleeding", "tcp"]

recs = []
for (patient, i, e), g in rows_of_grid.groupby(["patient", "plan_set", "endpoint"]):
    v = g["value"].to_numpy(float)
    if v.size < 5:
        continue
    self_row = g[g.truth_set == i]["value"]
    recs.append({
        "patient": patient, "plan_set": i, "endpoint": e, "n_j": v.size,
        "median": float(np.median(v)),
        "range": float(v.max() - v.min()),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "sd": float(v.std(ddof=1)),
        "self": float(self_row.iloc[0]) if len(self_row) else np.nan,
    })
row_iov = pd.DataFrame(recs)
row_iov["range_pct_of_median"] = np.where(
    row_iov["median"].abs() > 1e-9, 100 * row_iov["range"] / row_iov["median"].abs(), np.nan)
row_iov.to_parquet(r / "sec22_row_iov.parquet", index=False)

print("=== within-plan spread across the assumed true anatomy (n = 49 plans) ===")
print("    every row: one delivered plan, scored on all 10 observer contours\n")
t = row_iov[row_iov.endpoint.isin(HEAD)].groupby("endpoint").agg(
    n=("range", "count"),
    range_med=("range", "median"),
    range_q1=("range", lambda s: s.quantile(.25)),
    range_q3=("range", lambda s: s.quantile(.75)),
    range_max=("range", "max"),
    sd_med=("sd", "median"),
    iqr_med=("iqr", "median"),
    relrange_med=("range_pct_of_median", "median"),
).reindex(HEAD)
print(t.round(3).to_string())

print("\n=== the same, per patient: median within-plan range ===")
pp = row_iov[row_iov.endpoint.isin(HEAD)].pivot_table(
    index="endpoint", columns="patient", values="range", aggfunc="median")
print(pp.reindex(HEAD).round(2).to_string())

print("\n=== worst single plan per endpoint ===")
for e in HEAD:
    g = row_iov[row_iov.endpoint == e]
    if g.empty:
        continue
    w = g.loc[g["range"].idxmax()]
    lo = w["median"]
    print(f"  {e:22s} {w.patient}/{w.plan_set}: range {w['range']:.3f}  "
          f"(median across j {lo:.3f}, self {w['self']:.3f})")

print("\n=== absolute min and max across j for the worst plan, headline endpoints ===")
for e in ["rectum_v70pct", "rectum_dmean", "bladder_dmean", "rectum_late_bleeding", "ctv_d98"]:
    g = row_iov[row_iov.endpoint == e]
    w = g.loc[g["range"].idxmax()]
    sub = rows_of_grid[(rows_of_grid.patient == w.patient)
                       & (rows_of_grid.plan_set == w.plan_set)
                       & (rows_of_grid.endpoint == e)]
    v = sub["value"]
    print(f"  {e:22s} {w.patient}/{w.plan_set}: {v.min():.2f} to {v.max():.2f} "
          f"(self {w['self']:.2f})")
