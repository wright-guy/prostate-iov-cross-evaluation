"""Checks behind the 2026-09-14 revision. Reads results/ only.

    python analysis/revision_checks.py

- How close the CTV-optimised plans come to a PTV-sized high-dose region (treated volume
  against post-hoc PTV volume), without reporting PTV dose metrics.
- Where the plan optimised on the correct contour ranks within its column, against the
  rank ten interchangeable plans would give.
- Within-patient Spearman correlations for the geometric metrics of Section 4, summarised
  over patients; writes results/section4_per_patient_rho.parquet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcx.config import load_config

cfg = load_config()
r = cfg.results_root
pd.set_option("display.width", 250)

# ------------------------------------------------ issue 3: how close are plans to PTV coverage?
d = pd.read_parquet(r / "derived_structures.parquet")
tv = d[d.structure == "Treated_Volume"][["patient", "contour_set", "volume_cc", "isodose_gy"]]
print("treated-volume isodose (Gy):", sorted(tv.isodose_gy.dropna().unique()))
s = pd.read_parquet(r / "structures.parquet")
print("structures cols:", list(s.columns)[:12])
vcol = "volume_cc" if "volume_cc" in s.columns else [c for c in s.columns if "vol" in c][0]
ptv = s[s.structure == "PTV"][["patient", "contour_set", vcol]].rename(columns={vcol: "ptv_cc"})
ctv = s[s.structure == "CTV"][["patient", "contour_set", vcol]].rename(columns={vcol: "ctv_cc"})
m = tv.merge(ptv, on=["patient", "contour_set"]).merge(ctv, on=["patient", "contour_set"])
m = m[m.contour_set.isin(cfg.observers)]
m["tv_over_ptv"] = m.volume_cc / m.ptv_cc
m["tv_over_ctv"] = m.volume_cc / m.ctv_cc
print(m[["tv_over_ptv", "tv_over_ctv"]].describe().round(2).to_string())

c = pd.read_parquet(r / "constraints_long.parquet")
dg = c[(c.arm == "diagonal") & c.plan_set.isin(cfg.observers)]
for k in ("ptv_d95", "ctv_d99", "rectum_v70"):
    v = dg[dg.constraint == k]["value"]
    print(f"diagonal {k}: median {v.median():.2f}, range {v.min():.2f}-{v.max():.2f}, "
          f"per-protocol {100*(dg[dg.constraint==k].tier=='per_protocol').mean():.0f}%")

# ------------------------------------------------ issue 2: rank of the correct plan in its column
ep = pd.read_parquet(r / "endpoints_long.parquet")
bio = pd.read_parquet(r / "bio_long.parquet")
bio = bio[(bio.model == "ntcp") & (bio.variant == "relative")
          & (bio.endpoint_id == "rectum_late_bleeding")].copy()
bio["endpoint"], bio["value"] = "rectum_late_bleeding", bio.value * 100
cols = ["patient", "plan_set", "truth_set", "arm", "endpoint", "value"]
allv = pd.concat([ep[cols], bio[cols]])
obs = allv[allv.plan_set.isin(cfg.observers) & allv.arm.isin(["S", "diagonal"])]
print("\n=== rank of the correct plan within its column (0 = best, 1 = worst) ===")
for e, lower_better in (("rectum_v70pct", True), ("rectum_dmean", True),
                        ("rectum_late_bleeding", True), ("bladder_dmean", True),
                        ("ctv_d98", False), ("paddick_ci", False)):
    rows = []
    for (p, j), g in obs[obs.endpoint == e].groupby(["patient", "truth_set"]):
        n = len(g)
        own = float(g[g.plan_set == j].value.iloc[0])
        better = (g.value < own).sum() if lower_better else (g.value > own).sum()
        ties = (g.value == own).sum() - 1
        rank = better + 0.5 * ties          # plans doing better than the correct one
        rows.append({"patient": p, "norm_rank": rank / (n - 1), "best": int(better == 0)})
    df = pd.DataFrame(rows)
    per = df.groupby("patient").norm_rank.mean()
    t = stats.ttest_1samp(per, 0.5)
    print(f"  {e:22s} mean normalised rank {df.norm_rank.mean():.2f} "
          f"(per patient {np.round(per.values, 2)}; t-test vs 0.5 over 5 patients p = {t.pvalue:.3f}); "
          f"correct plan best in {df.best.sum()}/{len(df)} (exchangeable expectation "
          f"{sum(1/10 if len(g) == 10 else 1/9 for _, g in obs[obs.endpoint == e].groupby(['patient','truth_set'])):.1f})")

# ------------------------------------------------ issue 4: per-patient Spearman for Section 4
f = pd.read_parquet(r / "aim3_features.parquet")
p = pd.read_parquet(r / "aim3_prediction.parquet")
print("\n=== Section 4: per-patient Spearman, geometric metrics only ===")
metrics = ["dsc", "hd95_mm", "msd_mm", "signed_volume_difference", "signed_msd_mm"]
out = []
for (st, e), g in f.groupby(["structure", "endpoint"]):
    for tgt in ("estimation_error", "replanning_benefit"):
        for mtr in metrics:
            per = []
            for pt, gp in g.groupby("patient"):
                ok = gp[[mtr, tgt]].dropna()
                per.append(stats.spearmanr(ok[mtr], ok[tgt])[0])
            lopo = p[(p.structure == st) & (p.target == tgt) & (p.metric == mtr)].lopo_r2.iloc[0]
            out.append({"endpoint": e, "target": tgt, "metric": mtr,
                        "rho_median": np.median(per), "rho_min": np.min(per),
                        "rho_max": np.max(per), "lopo_r2": lopo})
o = pd.DataFrame(out)
o.to_parquet(r / "section4_per_patient_rho.parquet", index=False)
print(o.round(3).to_string(index=False))
