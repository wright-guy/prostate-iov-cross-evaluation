"""Stage 10 — Tier 4 QC on the extracted endpoints.

    python pipelines/10_qc_endpoints.py

The last gate before anything is interpreted. Four things it establishes:

1.  **Completeness.** The realised cell count must equal the design matrix exactly, with
    no silent gaps and no unexplained missing values.
2.  **Internal consistency.** D98 <= D95 <= D50 <= D2, V50 >= V60 >= V65 >= V70, and an
    absolute sub-volume can never exceed its structure. These cannot fail for physical
    reasons, so a failure means a code or convention error.
3.  **Diagonal consistency.** Spread of endpoint(i, i) within a patient. This substitutes
    for the replanning noise-floor experiment that cannot be run here: if off-diagonal
    spread greatly exceeds diagonal spread, planning inconsistency is not driving the
    result. If they are comparable, that is a finding about achievable plan
    reproducibility and must be reported as one.
4.  **Informativeness.** An endpoint pinned at a floor or ceiling across all 530 cells
    carries no information about contouring variation. That is a real result about model
    applicability, not a defect, but interpreting such an endpoint as if it varied would
    be an error -- so it is detected and named here rather than discovered in review.

Writes results/qc_findings_10.parquet and results/diagonal_consistency.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import Config, load_config  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402
from mcx.qc.checks import QCLog, enforce  # noqa: E402

# Ordered endpoint chains that physics forbids from crossing.
MONOTONIC_DOSE = [
    ("CTV", ["ctv_d98", "ctv_d95", "ctv_d50", "ctv_d2"]),
]
MONOTONIC_VOLUME = [
    ("Rectum", ["rectum_v50pct", "rectum_v60pct", "rectum_v65pct", "rectum_v70pct"]),
    ("Bladder", ["bladder_v65pct", "bladder_v70pct"]),
]

# An endpoint whose full range across all 530 cells is below this fraction of its
# median is pinned, and cannot inform a compression factor.
DEGENERATE_RELATIVE_RANGE = 0.01

# A probability endpoint sitting within this of 0 or 1 across every cell is on a floor
# or ceiling of its model, which is a statement about the model, not about contouring.
PROBABILITY_EDGE = 0.01


def check_completeness(cfg: Config, ep: pd.DataFrame, log: QCLog) -> None:
    realised = set(map(tuple, ep[["patient", "plan_set", "truth_set"]]
                       .drop_duplicates().to_numpy()))
    expected = {(p, i, j) for p, i, j, _ in cfg.design_cells()}
    log.check("cells.count_matches_design", realised == expected,
              f"{len(realised)} cells realised, {len(expected)} in the design; "
              f"{len(expected - realised)} missing, {len(realised - expected)} unexpected",
              tier=4, value=float(len(realised)), expected=str(len(expected)))

    for endpoint, grp in ep.groupby("endpoint"):
        n_missing = int(grp["value"].isna().sum())
        log.check("endpoint.no_silent_gaps", n_missing == 0,
                  f"{n_missing} of {len(grp)} values missing",
                  tier=4, value=float(n_missing), expected="0", structure=str(endpoint))


def check_monotonicity(ep: pd.DataFrame, log: QCLog) -> None:
    wide = ep.pivot_table(index=["patient", "plan_set", "truth_set"],
                          columns="endpoint", values="value")
    for structure, chain in MONOTONIC_DOSE:
        present = [c for c in chain if c in wide.columns]
        for a, b in zip(present, present[1:], strict=False):
            bad = int((wide[a] > wide[b] + 1e-6).sum())
            log.check("endpoint.dose_monotonic", bad == 0,
                      f"{a} exceeds {b} in {bad} of {len(wide)} cells",
                      tier=4, value=float(bad), expected="0", structure=structure)
    for structure, chain in MONOTONIC_VOLUME:
        present = [c for c in chain if c in wide.columns]
        for a, b in zip(present, present[1:], strict=False):
            bad = int((wide[a] < wide[b] - 1e-6).sum())
            log.check("endpoint.volume_monotonic", bad == 0,
                      f"{a} is below {b} in {bad} of {len(wide)} cells",
                      tier=4, value=float(bad), expected="0", structure=structure)


def check_informativeness(ep: pd.DataFrame, bio: pd.DataFrame, log: QCLog) -> pd.DataFrame:
    rows = []
    for endpoint, grp in ep.groupby("endpoint"):
        v = grp["value"].dropna()
        if v.empty:
            continue
        med = float(v.median())
        rng = float(v.max() - v.min())
        rel = rng / abs(med) if med else np.inf
        degenerate = rel < DEGENERATE_RELATIVE_RANGE
        rows.append({"endpoint": str(endpoint), "kind": "dvh", "n": len(v),
                     "median": med, "min": float(v.min()), "max": float(v.max()),
                     "relative_range": rel, "informative": not degenerate})
        log.check("endpoint.varies_across_cells", not degenerate,
                  f"range {rng:.4g} is {100 * rel:.3f}% of the median {med:.4g}; "
                  "this endpoint cannot inform a compression factor",
                  tier=4, warn_only=True, value=rel, structure=str(endpoint))

    for (model, structure, variant), grp in bio.groupby(
        ["model", "structure", "variant"], dropna=False
    ):
        v = grp["value"].dropna()
        if v.empty:
            continue
        at_floor = float(v.max()) < PROBABILITY_EDGE
        at_ceiling = float(v.min()) > 1 - PROBABILITY_EDGE
        rows.append({"endpoint": f"{model}:{structure}:{variant}", "kind": "biological",
                     "n": len(v), "median": float(v.median()),
                     "min": float(v.min()), "max": float(v.max()),
                     "relative_range": float(v.max() - v.min()),
                     "informative": not (at_floor or at_ceiling)})
        log.check("endpoint.probability_not_pinned", not (at_floor or at_ceiling),
                  f"{model} for {structure} ({variant}) spans "
                  f"{v.min():.2e} to {v.max():.2e}: pinned at the model's "
                  f"{'floor' if at_floor else 'ceiling'} across all cells",
                  tier=4, warn_only=True, structure=f"{structure}:{variant}")
    return pd.DataFrame(rows)


def diagonal_consistency(ep: pd.DataFrame, log: QCLog) -> pd.DataFrame:
    """Spread on the diagonal against spread off it, per endpoint and patient."""
    rows = []
    for (patient, endpoint), grp in ep.groupby(["patient", "endpoint"]):
        diag = grp.loc[grp["arm"] == "diagonal", "value"].dropna()
        off = grp.loc[grp["arm"] == "S", "value"].dropna()
        if len(diag) < 3 or len(off) < 3:
            continue

        def iqr(s: pd.Series) -> float:
            q1, q3 = np.percentile(s, [25, 75])
            return float(q3 - q1)

        d_iqr, o_iqr = iqr(diag), iqr(off)
        rows.append({"patient": patient, "endpoint": endpoint,
                     "diagonal_median": float(diag.median()), "diagonal_iqr": d_iqr,
                     "off_diagonal_median": float(off.median()), "off_diagonal_iqr": o_iqr,
                     "ratio_off_to_diagonal": o_iqr / d_iqr if d_iqr > 0 else np.inf})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for endpoint, grp in out.groupby("endpoint"):
        ratio = float(grp["ratio_off_to_diagonal"].replace(np.inf, np.nan).median())
        log.check("diagonal.spread_is_smaller_than_off_diagonal",
                  not np.isfinite(ratio) or ratio > 1.0,
                  f"off-diagonal IQR is {ratio:.2f}x the diagonal IQR; below 1 would "
                  "mean plan-to-plan inconsistency exceeds the contour effect",
                  tier=4, warn_only=True, value=ratio, structure=str(endpoint))
    return out


def main() -> int:
    cfg = load_config()
    run = RunInfo.create("10_qc_endpoints", cfg)
    r = cfg.results_root
    for name in ("endpoints_long.parquet", "bio_long.parquet"):
        if not (r / name).exists():
            print(f"results/{name} not found — run stages 08 and 09 first")
            return 2
    ep = pd.read_parquet(r / "endpoints_long.parquet")
    bio = pd.read_parquet(r / "bio_long.parquet")

    log = QCLog(cfg, "10_qc_endpoints")
    print("Tier 4 — endpoints")
    check_completeness(cfg, ep, log)
    check_monotonicity(ep, log)
    info = check_informativeness(ep, bio, log)
    diag = diagonal_consistency(ep, log)

    write_table(log.to_frame(), r / "qc_findings_10.parquet", run)
    write_table(info, r / "endpoint_informativeness.parquet", run)
    if not diag.empty:
        write_table(diag, r / "diagonal_consistency.parquet", run)

    pd.set_option("display.width", 220)
    print("\n" + log.summary())

    uninformative = info[~info["informative"]]
    if len(uninformative):
        print("\n=== endpoints that carry no information about contouring ===")
        print(uninformative[["endpoint", "kind", "median", "min", "max"]]
              .round(6).to_string(index=False))

    if not diag.empty:
        print("\n=== diagonal consistency: off-diagonal IQR / diagonal IQR ===")
        print("    (high means the contour effect dominates plan-to-plan inconsistency)")
        summary = (diag.replace(np.inf, np.nan)
                   .groupby("endpoint")["ratio_off_to_diagonal"]
                   .median().sort_values(ascending=False))
        print(summary.round(2).to_string())

    for _, row in log.to_frame().query("status == 'FAIL'").head(20).iterrows():
        print(f"  FAIL [{row['check']}] {row['structure']} — {row['message']}")

    enforce(log)
    print("\nGate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
