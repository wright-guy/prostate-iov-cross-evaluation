"""Stage 11 — H5: planning-side against evaluation-side variance.

    python pipelines/11_variance_components.py

For every endpoint, decomposes the Arm S cross-evaluation matrix within each patient
into the variance attributable to the planning contour (which plan was made), the
evaluation contour (whose anatomy scored it), and their interaction.

H5 predicts that for rectal and bladder endpoints the evaluation side contributes as
much as or more than the planning side, because a DVH is scored directly on the volume
in question while the plan only ever sees that organ through an optimisation weight.

Writes results/aim2_variance_components.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import load_config  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402
from mcx.stats.bootstrap import cluster_bootstrap  # noqa: E402
from mcx.stats.variance import (  # noqa: E402
    between_patient_share,
    mixed_model_crosscheck,
    two_way_within_patient,
)

HEADLINE = [
    "ctv_d98", "ladder_d98_3mm", "ladder_d98_5mm", "paddick_ci",
    "rectum_v70pct", "rectum_v65pct", "rectum_d2cc", "rectum_dmean",
    "bladder_v70pct", "bladder_dmean",
]


def _bio_as_endpoints(bio: pd.DataFrame) -> pd.DataFrame:
    b = bio[(bio["model"] == "ntcp") | (bio["density_label"] == "calibrated")].copy()
    b["endpoint"] = np.where(
        b["model"] == "ntcp",
        b["endpoint_id"].astype(str) + "_" + b["variant"].astype(str),
        "tcp_" + b["variant"].astype(str),
    )
    return b[["patient", "plan_set", "truth_set", "arm", "structure", "endpoint", "value"]]


def main() -> int:
    cfg = load_config()
    run = RunInfo.create("11_variance_components", cfg)
    r = cfg.results_root
    boot = cfg.analysis_raw["uncertainty"]

    ep = pd.read_parquet(r / "endpoints_long.parquet")
    bio = pd.read_parquet(r / "bio_long.parquet")
    info = pd.read_parquet(r / "endpoint_informativeness.parquet")
    uninformative = set(info.loc[~info["informative"], "endpoint"])

    combined = pd.concat(
        [ep[["patient", "plan_set", "truth_set", "arm", "structure", "endpoint", "value"]],
         _bio_as_endpoints(bio)], ignore_index=True)
    arm_s = combined[combined["arm"] == "S"]

    rows = []
    for endpoint, grp in arm_s.groupby("endpoint"):
        if endpoint in uninformative:
            continue
        per_patient = two_way_within_patient(grp)
        if per_patient.empty:
            continue
        ratio = cluster_bootstrap(
            per_patient,
            lambda d: float(d["ss_truth"].sum() / d["ss_plan"].sum())
            if d["ss_plan"].sum() > 0 else np.nan,
            n_resamples=int(boot["n_resamples"]), seed=int(boot["seed"]),
            level=float(boot["level"]),
        )
        cross = mixed_model_crosscheck(grp)
        rows.append({
            "endpoint": endpoint,
            "structure": grp["structure"].iloc[0],
            "n_patients": len(per_patient),
            "pct_plan": float(per_patient["pct_plan"].median()),
            "pct_truth": float(per_patient["pct_truth"].median()),
            "pct_interaction": float(per_patient["pct_interaction"].median()),
            "truth_over_plan": ratio["point"],
            "ci_lo": ratio["lo"], "ci_hi": ratio["hi"],
            "evaluation_side_dominates": bool(
                np.isfinite(ratio["lo"]) and ratio["lo"] > 1.0),
            "between_patient_pct": between_patient_share(grp),
            "crosscheck_converged": bool(cross["converged"]),
            "crosscheck_truth_over_plan": (
                cross["truth_var"] / cross["plan_var"]
                if cross["plan_var"] and np.isfinite(cross["plan_var"])
                and cross["plan_var"] > 0 else np.nan),
        })

    out = pd.DataFrame(rows)
    write_table(out, r / "aim2_variance_components.parquet", run)

    pd.set_option("display.width", 240)
    print("=== H5: within-patient variance shares for the Arm S matrix ===")
    print("    plan  = which plan was made (planning side)")
    print("    truth = whose anatomy scored it (evaluation side)")
    print("    ratio = truth / plan sum of squares, with a cluster bootstrap over patients\n")
    show = out[out["endpoint"].isin(HEADLINE)].copy()
    show["ratio"] = show.apply(
        lambda x: f"{x.truth_over_plan:.2f} [{x.ci_lo:.2f}, {x.ci_hi:.2f}]"
        + ("  *" if x.evaluation_side_dominates else ""), axis=1)
    print(show.set_index("endpoint")[
        ["pct_plan", "pct_truth", "pct_interaction", "ratio", "between_patient_pct"]
    ].reindex([e for e in HEADLINE if e in set(show["endpoint"])]).round(1).to_string())
    print("\n  * the interval lies entirely above 1: the evaluation contour contributes")
    print("    more variance than the planning contour")

    conv = int(out["crosscheck_converged"].sum())
    print(f"\nmixed-model cross-check converged for {conv} of {len(out)} endpoints; "
          "the two-way decomposition is primary (see src/mcx/stats/variance.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
