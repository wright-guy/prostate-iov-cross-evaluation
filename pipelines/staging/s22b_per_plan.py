import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from mcx.config import load_config
from mcx.stats.bootstrap import bootstrap_ratio_of_medians

cfg = load_config()
r = cfg.results_root
pd.set_option("display.width", 260)

ep = pd.read_parquet(r / "endpoints_long.parquet")
cons = pd.read_parquet(r / "consensus_scored_cells.parquet")

HEAD = ["ctv_d98", "ctv_d50", "ctv_v95pct", "rectum_v70pct", "rectum_dmean",
        "rectum_d2cc", "bladder_v70pct", "bladder_dmean", "paddick_ci"]

diag = ep[ep.arm == "diagonal"].set_index(["patient", "plan_set", "endpoint"])["value"]
S = ep[ep.arm == "S"]

recs = []
for (patient, i, e), g in S.groupby(["patient", "plan_set", "endpoint"]):
    key = (patient, i, e)
    if key not in diag.index:
        continue
    self_val = float(diag.loc[key])
    A = float(np.mean(np.abs(g["value"].to_numpy(float) - self_val)))
    rec = {"patient": patient, "plan_set": i, "endpoint": e,
           "self": self_val, "A": A, "n_j": len(g)}
    for kind in ("LOO", "VOTE"):
        c = cons[(cons.patient == patient) & (cons.plan_set == i)
                 & (cons.endpoint == e) & (cons.truth_kind == kind)]
        rec[f"B_{kind}"] = abs(float(c["value"].iloc[0]) - self_val) if len(c) else np.nan
        rec[f"E_{kind}"] = float(c["value"].iloc[0]) if len(c) else np.nan
    recs.append(rec)

per_plan = pd.DataFrame(recs)
per_plan.to_parquet(r / "sec22_per_plan.parquet", index=False)

print("=== per-plan A and B, medians across the 49 observer plans ===")
t = per_plan[per_plan.endpoint.isin(HEAD)].groupby("endpoint").agg(
    n=("A", "count"),
    A_med=("A", "median"), A_q1=("A", lambda s: s.quantile(.25)),
    A_q3=("A", lambda s: s.quantile(.75)),
    Bloo_med=("B_LOO", "median"), Bvote_med=("B_VOTE", "median"),
).reindex(HEAD)
t["R_loo"] = t.A_med / t.Bloo_med
t["R_vote"] = t.A_med / t.Bvote_med
print(t.round(4).to_string())

print("\n=== bootstrap CIs (cluster over patients, 10000 resamples) ===")
long = per_plan.melt(id_vars=["patient", "plan_set", "endpoint"],
                     value_vars=["A", "B_LOO", "B_VOTE"],
                     var_name="group", value_name="value")
out = []
for e in HEAD:
    sub = long[long.endpoint == e]
    for den, name in (("B_LOO", "loo"), ("B_VOTE", "vote")):
        b = bootstrap_ratio_of_medians(sub, value="value", group="group",
                                       numerator="A", denominator=den,
                                       cluster="patient", n_resamples=10000, seed=20260904)
        out.append({"endpoint": e, "denominator": name, **b})
boot = pd.DataFrame(out)
boot.to_parquet(r / "sec22_compression.parquet", index=False)
print(boot.round(3).to_string())
