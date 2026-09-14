"""Section 2.1 numbers: self-evaluation, plus the decomposition that justifies
computing spreads within a fixed plan rather than pooling."""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from mcx.config import load_config
from mcx.dose.store import DoseStore
from mcx.endpoints.evaluate import evaluate, oar_specs, structure_dose, target_specs
from mcx.geom.analysis_grid import load_grid
from mcx.geom.masks import MaskStore
from mcx.geom.shells import paddick_conformity

cfg = load_config()
r = cfg.results_root
pd.set_option("display.width", 250)

# ---- 1. VOTE self-evaluation: VOTE plan scored on the VOTE contours.
store, doses = MaskStore(cfg), DoseStore(cfg)
rows = []
for patient in cfg.patients:
    grid = load_grid(cfg, patient)
    vox = grid.voxel_volume_cc
    dose, covered = doses.get(patient, "VOTE")
    treated = store.get(patient, "VOTE", "Treated_Volume")

    def add(structure, specs):
        if not store.has(patient, "VOTE", structure):
            return
        d = structure_dose(dose, covered, store.get(patient, "VOTE", structure))
        for spec in specs:
            rows.append({"patient": patient, "structure": structure,
                         "endpoint": spec["id"], "value": evaluate(spec, d, vox)})

    add("CTV", target_specs(cfg))
    for structure, specs in oar_specs(cfg).items():
        add(structure, specs)
    ctv = store.get(patient, "VOTE", "CTV")
    rows.append({"patient": patient, "structure": "CTV", "endpoint": "paddick_ci",
                 "value": paddick_conformity(treated, ctv)})
    doses.drop_patient(patient)
    store.drop_patient(patient)

vote_self = pd.DataFrame(rows)
vote_self.to_parquet(r / "vote_self_evaluation.parquet", index=False)

ep = pd.read_parquet(r / "endpoints_long.parquet")
HEAD = ["ctv_d98", "ctv_d50", "ctv_v95pct", "rectum_v70pct", "rectum_dmean",
        "rectum_d2cc", "bladder_v70pct", "bladder_dmean", "paddick_ci"]

print("=== A. observer self-evaluation, across the 49 observer plans ===")
diag = ep[ep["arm"] == "diagonal"]
a = diag[diag["endpoint"].isin(HEAD)].groupby("endpoint")["value"].agg(
    n="count", med="median", q1=lambda s: s.quantile(.25),
    q3=lambda s: s.quantile(.75), lo="min", hi="max").reindex(HEAD)
print(a.round(3).to_string())

print("\n=== B. VOTE self-evaluation, across the 5 consensus plans ===")
b = vote_self[vote_self["endpoint"].isin(HEAD)].groupby("endpoint")["value"].agg(
    n="count", med="median", lo="min", hi="max").reindex(HEAD)
print(b.round(3).to_string())

print("\n=== C. is the VOTE plan typical of the observer plans? (per patient) ===")
for e in ["ctv_d98", "rectum_v70pct", "paddick_ci"]:
    print(f"\n-- {e}")
    for patient in cfg.patients:
        obs = diag[(diag.patient == patient) & (diag.endpoint == e)]["value"]
        v = vote_self[(vote_self.patient == patient) & (vote_self.endpoint == e)]["value"]
        if obs.empty or v.empty:
            continue
        pct = 100.0 * float((obs < v.iloc[0]).mean())
        print(f"  {patient}: observers {obs.min():.2f}-{obs.max():.2f} "
              f"(median {obs.median():.2f}) | VOTE {v.iloc[0]:.2f} "
              f"| percentile {pct:.0f}")

print("\n=== D. what drives the spread ON the diagonal? ===")
print("  diagonal : IQR of E(i,i) across i        -- contour AND plan vary together")
print("  plan only: fix contour j, IQR across i   -- contour held constant")
print("  cont only: fix plan i, IQR across j      -- plan held constant\n")
out = []
for e in HEAD:
    g = ep[ep.endpoint == e]
    per = []
    for patient in cfg.patients:
        gp = g[g.patient == patient]
        d = gp[gp.arm == "diagonal"]["value"]
        s = gp[gp.arm == "S"]
        if len(d) < 4 or s.empty:
            continue
        def iqr(v):
            v = np.asarray(v, float)
            return float(np.percentile(v, 75) - np.percentile(v, 25))
        plan_only = np.median([iqr(x["value"]) for _, x in s.groupby("truth_set")
                               if len(x) > 3])
        cont_only = np.median([iqr(x["value"]) for _, x in s.groupby("plan_set")
                               if len(x) > 3])
        per.append((iqr(d), plan_only, cont_only))
    if per:
        arr = np.array(per)
        out.append({"endpoint": e, "diagonal_iqr": arr[:, 0].mean(),
                    "plan_only_iqr": arr[:, 1].mean(), "contour_only_iqr": arr[:, 2].mean()})
d = pd.DataFrame(out).set_index("endpoint")
d["plan_share_pct"] = 100 * d.plan_only_iqr / (d.plan_only_iqr + d.contour_only_iqr)
print(d.round(3).to_string())
