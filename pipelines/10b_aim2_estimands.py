"""Stage 10b — Aim 2: the three estimands, compression in dose space, flip rates.

    python pipelines/10b_aim2_estimands.py

Reads results/ only, so the statistics can be revisited without recomputing any dose.

Tests H3 (dosimetric compression is smaller than geometric), H4 (compression is weakest
in the tails) and produces the constraint flip rates §7.3 wants in the abstract.

Writes aim2_estimands.parquet, aim2_compression.parquet and aim2_flip_rates.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import load_config  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402
from mcx.stats.bootstrap import bootstrap_ratio_of_medians  # noqa: E402
from mcx.stats.estimands import (  # noqa: E402
    e1_delivery,
    e2_truth_uncertainty,
    e3_systematic_loss,
    error_spread,
)

# The geometric compression measured in Aim 1, for the H3 comparison.
GEOMETRIC_BENCHMARK = 1.34

# Endpoints carried into the headline tables. Chosen from the pre-registered list for
# clinical legibility, not because of how they turned out.
HEADLINE = [
    "ctv_d98", "ctv_v95pct", "ladder_d98_3mm", "ladder_d98_5mm",
    "rectum_v70pct", "rectum_v65pct", "rectum_d2cc", "rectum_dmean",
    "bladder_v70pct", "bladder_dmean", "paddick_ci",
]


def _bio_as_endpoints(bio: pd.DataFrame) -> pd.DataFrame:
    """Reshape the biological table to the same schema as the DVH endpoints."""
    b = bio.copy()
    keep = (b["model"] == "ntcp") | (b["density_label"] == "calibrated")
    b = b[keep]
    b["endpoint"] = np.where(
        b["model"] == "ntcp",
        b["endpoint_id"].astype(str) + "_" + b["variant"].astype(str),
        "tcp_" + b["variant"].astype(str),
    )
    return b[["patient", "plan_set", "truth_set", "arm", "structure", "endpoint", "value"]]


def main() -> int:
    cfg = load_config()
    run = RunInfo.create("10b_aim2_estimands", cfg)
    r = cfg.results_root
    boot = cfg.analysis_raw["uncertainty"]

    ep = pd.read_parquet(r / "endpoints_long.parquet")
    bio = pd.read_parquet(r / "bio_long.parquet")
    info = pd.read_parquet(r / "endpoint_informativeness.parquet")
    combined = pd.concat([ep[["patient", "plan_set", "truth_set", "arm", "structure",
                              "endpoint", "value"]],
                          _bio_as_endpoints(bio)], ignore_index=True)

    uninformative = set(info.loc[~info["informative"], "endpoint"])

    # ------------------------------------------------------------- estimands
    rows = []
    for endpoint, grp in combined.groupby("endpoint"):
        e1 = e1_delivery(grp)
        e2 = e2_truth_uncertainty(grp)
        e3 = e3_systematic_loss(grp)
        for arm in ("S", "C"):
            rows.append({
                "endpoint": endpoint, "arm": arm,
                "e1_median": float(e1.loc[e1["arm"] == arm, "e1"].median())
                if (e1["arm"] == arm).any() else np.nan,
                "e2_median": float(e2.loc[e2["arm"] == arm, "e2"].median())
                if (e2["arm"] == arm).any() else np.nan,
                "e3_mean": float(e3.loc[e3["arm"] == arm, "e3_mean"].mean())
                if (e3["arm"] == arm).any() else np.nan,
                "e3_worst": float(e3.loc[e3["arm"] == arm, "e3_worst"]
                                  .abs().max()) if (e3["arm"] == arm).any() else np.nan,
            })
    est = pd.DataFrame(rows)
    write_table(est, r / "aim2_estimands.parquet", run)

    # ------------------------------------------ compression, H3 and H4, with intervals
    rows = []
    for endpoint, grp in combined.groupby("endpoint"):
        if endpoint in uninformative:
            continue
        for measure, tail in (("iqr", False), ("p5_p95_range", True), ("range", True)):
            # Two definitions, both reported.
            #
            # "error"  spread of endpoint(i,j) - endpoint(j,j). Structurally analogous to
            #          the geometric benchmark, which is a variance identity on the
            #          distance between a contour and its reference. This is the
            #          like-for-like comparison with 1.34.
            # "e2"     spread of the endpoint itself across truths, at a fixed plan. The
            #          pre-registered form. It answers a real question but is dominated
            #          by how much the ten truths differ from each other, which barely
            #          depends on which plan is scored -- so it is expected to sit near 1
            #          whatever consensus does, and cannot on its own refute H3.
            for basis, table, column in (
                ("error", error_spread(grp, measure), "error_spread"),
                ("e2", e2_truth_uncertainty(grp, measure), "e2"),
            ):
                res = bootstrap_ratio_of_medians(
                    table, value=column, group="arm", numerator="S", denominator="C",
                    n_resamples=int(boot["n_resamples"]), seed=int(boot["seed"]),
                    level=float(boot["level"]),
                )
                rows.append({
                    "endpoint": endpoint, "basis": basis,
                    "spread_measure": measure, "is_tail": tail,
                    "compression_factor": res["point"],
                    "ci_lo": res["lo"], "ci_hi": res["hi"],
                    "geometric_benchmark": GEOMETRIC_BENCHMARK,
                    "below_geometric": bool(np.isfinite(res["hi"])
                                            and res["hi"] < GEOMETRIC_BENCHMARK),
                })
    comp = pd.DataFrame(rows)
    write_table(comp, r / "aim2_compression.parquet", run)

    # --------------------------------------------------------------- flip rates
    cs = pd.read_parquet(r / "constraints_long.parquet")
    rows = []
    for (constraint, arm), grp in cs.groupby(["constraint", "arm"]):
        if arm not in ("S", "C"):
            continue
        # Two directions, and the distinction matters.
        #
        # across_plans   fix the truth, vary the plan. Zero in Arm C BY CONSTRUCTION,
        #                since there is one consensus plan -- a definitional artefact,
        #                not a merit, so it is labelled as such.
        # across_truths  fix the plan, vary the assumed true anatomy. Non-zero in both
        #                arms and directly comparable between them. This is the
        #                clinically meaningful one: for a plan that has been delivered,
        #                does its protocol compliance depend on whose contour scores it?
        across_plans = grp.groupby(["patient", "truth_set"])["tier"].nunique()
        across_truths = grp.groupby(["patient", "plan_set"])["tier"].nunique()
        pp = grp.assign(pp=grp["tier"] == "per_protocol")
        binary_plans = pp.groupby(["patient", "truth_set"])["pp"].nunique()
        binary_truths = pp.groupby(["patient", "plan_set"])["pp"].nunique()
        rows.append({
            "constraint": constraint, "label": grp["label"].iloc[0], "arm": arm,
            "n_truth_columns": int(len(across_plans)),
            "n_plan_rows": int(len(across_truths)),
            "ordinal_flip_across_plans": float((across_plans > 1).mean()),
            "binary_flip_across_plans": float((binary_plans > 1).mean()),
            "ordinal_flip_across_truths": float((across_truths > 1).mean()),
            "binary_flip_across_truths": float((binary_truths > 1).mean()),
            "n_tiers_overall": int(grp["tier"].nunique()),
            "modal_tier": grp["tier"].value_counts().idxmax(),
            "pct_per_protocol": float(100 * (grp["tier"] == "per_protocol").mean()),
        })
    flips = pd.DataFrame(rows)
    write_table(flips, r / "aim2_flip_rates.parquet", run)

    # ------------------------------------------------------------------ report
    pd.set_option("display.width", 240)
    print("=== E1 / E2 / E3 by arm, headline endpoints ===")
    show = est[est["endpoint"].isin(HEADLINE)].pivot_table(
        index="endpoint", columns="arm", values=["e1_median", "e2_median", "e3_mean"])
    print(show.reindex([e for e in HEADLINE if e in show.index]).round(4).to_string())
    print("\n  E1 is zero in Arm C by construction: there is one consensus plan.")

    print("\n\n=== H3: compression in dose space (IQR), against the geometric 1.34 ===")
    print("    error basis: spread of endpoint(i,j) - endpoint(j,j), the like-for-like")
    print("    e2 basis   : spread of the endpoint across truths at a fixed plan\n")
    for basis in ("error", "e2"):
        sub = comp[(comp["spread_measure"] == "iqr") & (comp["basis"] == basis)]
        sub = sub[sub["endpoint"].isin(HEADLINE)].copy()
        if sub.empty:
            continue
        sub["factor"] = sub.apply(
            lambda x: f"{x.compression_factor:.2f} [{x.ci_lo:.2f}, {x.ci_hi:.2f}]"
            + ("  *" if x.below_geometric else ""), axis=1)
        print(f"  basis = {basis}")
        print(sub.set_index("endpoint")[["factor"]]
              .reindex([e for e in HEADLINE if e in set(sub["endpoint"])]).to_string())
        print()
    print("  * interval lies entirely below the geometric benchmark")

    print("\n\n=== H4: centre against tails (error basis) ===")
    pivot = comp[(comp["endpoint"].isin(HEADLINE)) & (comp["basis"] == "error")].pivot_table(
        index="endpoint", columns="spread_measure", values="compression_factor")
    print(pivot.reindex([e for e in HEADLINE if e in pivot.index]).round(3).to_string())

    print("\n\n=== constraint flip rates, percent ===")
    print("    across truths: fix the delivered plan, vary whose contour scores it.")
    print("                   Comparable between arms. The clinically meaningful one.")
    print("    across plans : fix the truth, vary the plan. Zero in Arm C BY")
    print("                   CONSTRUCTION -- one consensus plan cannot disagree with")
    print("                   itself -- so it measures nothing about consensus quality.\n")
    f = flips.pivot_table(
        index=["constraint", "label"], columns="arm",
        values=["ordinal_flip_across_truths", "binary_flip_across_truths",
                "ordinal_flip_across_plans"])
    print((100 * f).round(0).astype("Int64").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
