"""Stage 07 — Aim 1: geometric agreement, the compression factor, and VOTE bias.

    python pipelines/07_geometric_metrics.py [--patient K018] [--structure CTV]

Tests H1 and H2, and measures the per-observer tendencies §7.2 asks for.

Three comparison families, all on the same metrics so they are directly comparable:
  * observer to observer     -- all 90 ordered pairs per patient per structure
  * observer to VOTE         -- the supplied consensus, 10 per patient per structure
  * observer to LOO-VOTE     -- the consensus built from the other nine

The leave-one-out family exists because comparing an observer against a consensus that
includes their own vote is circular: they have voted for their own contour, which
inflates the agreement. That leakage is what separates the 1.49 and 1.34 benchmarks.

Writes results/geometric_long.parquet, results/compression_geometric.parquet and
results/observer_tendency.parquet.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import Config, load_config  # noqa: E402
from mcx.geom.analysis_grid import load_grid  # noqa: E402
from mcx.geom.masks import MaskStore, load_mask, mask_path  # noqa: E402
from mcx.metrics.core import compare  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402

STRUCTURES = ("CTV", "Rectum", "Bladder", "PTV")

# Unsigned metrics whose ratio defines the compression factor. Signed metrics are
# reported but not used for compression: their mean is near zero by construction, so a
# ratio of means is unstable and uninformative.
COMPRESSION_METRICS = ("msd_mm", "hd95_mm", "dsc", "surface_dice")

BENCHMARK_UNCORRECTED = 1.49
BENCHMARK_LOO_CORRECTED = 1.34


def _loo_mask(cfg: Config, patient: str, held_out: str, structure: str) -> np.ndarray | None:
    path = mask_path(cfg, patient, f"LOO_{held_out}", structure)
    return load_mask(path) if path.exists() else None


def compare_all(cfg: Config, store: MaskStore, patients, structures) -> pd.DataFrame:
    rows: list[dict] = []
    tol = 3.0  # surface DSC tolerance, fixed in config/metrics.yaml

    for patient in patients:
        grid = load_grid(cfg, patient)
        observers = [o for o in cfg.observers if cfg.is_known_absent(patient, o) is None]
        t0 = time.perf_counter()

        for structure in structures:
            have = [o for o in observers if store.has(patient, o, structure)]
            masks = {o: store.get(patient, o, structure) for o in have}
            vote = (
                store.get(patient, "VOTE", structure)
                if store.has(patient, "VOTE", structure) else None
            )

            # observer to observer: ordered pairs, so signed metrics keep direction
            for a, b in itertools.permutations(have, 2):
                rows.append({
                    "patient": patient, "structure": structure, "family": "obs_obs",
                    "set_a": a, "set_b": b,
                    **compare(masks[a], masks[b], grid, surface_dice_tolerance_mm=tol),
                })

            for o in have:
                if vote is not None:
                    rows.append({
                        "patient": patient, "structure": structure, "family": "obs_vote",
                        "set_a": o, "set_b": "VOTE",
                        **compare(masks[o], vote, grid, surface_dice_tolerance_mm=tol),
                    })
                loo = _loo_mask(cfg, patient, o, structure)
                if loo is not None:
                    rows.append({
                        "patient": patient, "structure": structure, "family": "obs_loo",
                        "set_a": o, "set_b": f"LOO_{o}",
                        **compare(masks[o], loo, grid, surface_dice_tolerance_mm=tol),
                    })

        store.drop_patient(patient)
        print(f"  {patient}: {time.perf_counter() - t0:.0f}s")

    return pd.DataFrame(rows)


def compression(df: pd.DataFrame) -> pd.DataFrame:
    """Agreement to consensus against agreement between observers.

    For a distance metric (smaller is better) the compression factor is
    obs-obs / obs-consensus. For an overlap metric (larger is better) the disagreement
    is 1 - metric, so the factor is (1 - obs_obs) / (1 - obs_consensus). Both are then
    "how many times better is agreement with the consensus", directly comparable to the
    1.34-1.49 benchmarks.
    """
    rows = []
    for (patient, structure, metric), grp in _melt(df).groupby(
        ["patient", "structure", "metric"]
    ):
        by_family = grp.groupby("family")["value"].median()
        if "obs_obs" not in by_family:
            continue
        higher_is_better = metric in ("dsc", "surface_dice", "jaccard")
        base = by_family["obs_obs"]
        for family, label in (("obs_vote", "uncorrected"), ("obs_loo", "loo_corrected")):
            if family not in by_family:
                continue
            comp = by_family[family]
            if higher_is_better:
                ratio = (1 - base) / (1 - comp) if (1 - comp) > 0 else np.nan
            else:
                ratio = base / comp if comp > 0 else np.nan
            rows.append({
                "patient": patient, "structure": structure, "metric": metric,
                "comparison": label,
                "obs_obs": base, "obs_consensus": comp, "compression_factor": ratio,
                "benchmark": BENCHMARK_UNCORRECTED if label == "uncorrected"
                else BENCHMARK_LOO_CORRECTED,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["exceeds_benchmark"] = out["compression_factor"] > out["benchmark"]
    return out


def _melt(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in COMPRESSION_METRICS if c in df.columns]
    return df.melt(
        id_vars=["patient", "structure", "family", "set_a", "set_b"],
        value_vars=keep, var_name="metric", value_name="value",
    ).dropna(subset=["value"])


def observer_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """Per-observer systematic tendency, and whether it carries across structures.

    §7.2 asks whether an observer who contours a large target also contours a large
    OAR. Using the observer-to-LOO-consensus comparison rather than pairwise means each
    observer is measured against a reference that excludes them.
    """
    loo = df[df["family"] == "obs_loo"]
    if loo.empty:
        return pd.DataFrame()
    rows = []
    for (observer, structure), grp in loo.groupby(["set_a", "structure"]):
        rows.append({
            "observer": observer, "structure": structure, "n_patients": len(grp),
            "median_signed_volume_difference": float(grp["signed_volume_difference"].median()),
            "median_signed_msd_mm": float(grp["signed_msd_mm"].median()),
            "median_dsc": float(grp["dsc"].median()),
            "median_hd95_mm": float(grp["hd95_mm"].median()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", action="append")
    ap.add_argument("--structure", action="append")
    args = ap.parse_args()

    cfg = load_config()
    run = RunInfo.create("07_geometric_metrics", cfg)
    store = MaskStore(cfg)

    patients = args.patient or cfg.patients
    structures = args.structure or list(STRUCTURES)

    print(f"Aim 1 — geometric agreement over {len(structures)} structures")
    df = compare_all(cfg, store, patients, structures)
    write_table(df, cfg.results_root / "geometric_long.parquet", run)

    comp = compression(df)
    write_table(comp, cfg.results_root / "compression_geometric.parquet", run)

    tend = observer_tendency(df)
    if not tend.empty:
        write_table(tend, cfg.results_root / "observer_tendency.parquet", run)

    pd.set_option("display.width", 220)
    print(f"\n{len(df)} comparisons")
    print("\n=== agreement by family (median over all pairs) ===")
    print(df.groupby(["structure", "family"])[["dsc", "msd_mm", "hd95_mm", "surface_dice"]]
          .median().round(4).to_string())

    print("\n=== H1: compression factor against the geometric benchmarks ===")
    for label, bench in (("uncorrected", BENCHMARK_UNCORRECTED),
                         ("loo_corrected", BENCHMARK_LOO_CORRECTED)):
        sub = comp[comp["comparison"] == label]
        if sub.empty:
            continue
        print(f"\n  {label} (benchmark {bench})")
        print(sub.pivot_table(index="structure", columns="metric",
                              values="compression_factor", aggfunc="median")
              .round(3).to_string())

    print("\n=== H2: is the consensus an unbiased estimator of the observers? ===")
    for family, label in (("obs_vote", "supplied VOTE"), ("obs_loo", "LOO consensus")):
        sub = df[df["family"] == family]
        if sub.empty:
            continue
        agg = sub.groupby("structure")[["signed_volume_difference", "signed_msd_mm"]].median()
        print(f"\n  observer relative to {label} (positive = observer larger)")
        print(agg.round(4).to_string())

    if not tend.empty:
        print("\n=== observer tendency: signed volume difference against LOO consensus ===")
        print(tend.pivot_table(index="observer", columns="structure",
                               values="median_signed_volume_difference")
              .round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
