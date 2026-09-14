"""Stage 06 — Tier 3 QC on the resampled dose.

    python pipelines/06_qc_dose.py [--sample 60]

The gate that decides whether the endpoints in §5 can be trusted. Four things it
establishes, in descending order of how badly they would corrupt the results:

1.  **Dose-grid containment, per dose and per structure.** D2cc, V70cc and V60cc are
    silently wrong if any part of the scoring structure lies outside the calculation
    box. Checked for every (plan, contour set, structure) combination that will become
    a cell, not once per patient.
2.  **The resampling did not change the answer.** Endpoints computed end-to-end on the
    native dose grid are compared against the same endpoints on the analysis grid.
3.  **The DVH code is right.** The sort-based estimator used in the pipeline is checked
    against an independently implemented histogram-based one on real cells; both are
    pinned to closed-form answers in the tests.
4.  **Normalisation is consistent**, recomputed on the analysis grid where no contour
    is dropped — the definitive version of the §7.1 check.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import Config, load_config  # noqa: E402
from mcx.dose.resample import coverage_fraction  # noqa: E402
from mcx.dose.store import DoseStore  # noqa: E402
from mcx.endpoints import dvh, dvh_reference  # noqa: E402
from mcx.geom.analysis_grid import load_grid  # noqa: E402
from mcx.geom.masks import MaskStore  # noqa: E402
from mcx.geom.rasterise import grid_from_dose, rasterise  # noqa: E402
from mcx.io.rtdose import load_dose  # noqa: E402
from mcx.io.rtstruct import load_structure_set  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402
from mcx.qc.checks import QCLog, enforce  # noqa: E402

# Endpoints computed on the native dose grid and on the analysis grid must agree to
# this, in Gy. The two differ in mask resolution as well as dose interpolation, so this
# is an end-to-end tolerance on the whole geometric pipeline, not on interpolation alone.
NATIVE_VS_ANALYSIS_TOL_GY = 1.0

# The two DVH estimators must agree far more tightly: they see identical inputs.
ESTIMATOR_TOL_GY = 0.05

# Trilinear interpolation cannot exceed the source maximum, and it under-reads a peak
# whenever the analysis grid does not sample the source voxel that holds it. The loss
# therefore scales with dose, so the bound is relative rather than absolute.
#
# Observed: 0.12% median for the four patients whose CT and dose grids share a 2.5 mm
# slice spacing, rising to 0.29% median and 0.61% worst for V027, whose 2.0 mm analysis
# planes essentially never coincide with its 2.5 mm dose planes. 1% leaves headroom
# over that mechanism while still catching a scaling or geometry error, which would
# show up as a whole-number percentage.
#
# This is a smoke test, not a clinical one: no endpoint in section 5 uses the global
# maximum. D2% is the endpoint that carries that information, and it is checked
# directly below against the native-grid pipeline.
MAX_DOSE_TOL_PCT = 1.0

NORMALISATION_TOL_GY = 1.0


def _endpoints(values_desc: np.ndarray, voxel_cc: float) -> dict[str, float]:
    return {
        "D98": dvh.d_at_volume_pct(values_desc, 98),
        "D50": dvh.d_at_volume_pct(values_desc, 50),
        "D2": dvh.d_at_volume_pct(values_desc, 2),
        "Dmean": dvh.mean_dose(values_desc),
        "D2cc": dvh.d_at_volume_cc(values_desc, 2.0, voxel_cc),
        "V70pct": dvh.v_at_dose_pct(values_desc, 70.0),
    }


# ------------------------------------------------------------------- containment


def check_containment(
    cfg: Config, masks: MaskStore, doses: DoseStore, log: QCLog
) -> pd.DataFrame:
    """Coverage of every scoring structure by every plan's dose grid."""
    rows = []
    for patient in cfg.patients:
        sets = [s for s in cfg.all_sets if cfg.is_known_absent(patient, s) is None]
        for plan_set in sets:
            _, covered = doses.get(patient, plan_set)
            fully = bool(covered.all())
            for eval_set in sets:
                for structure in cfg.analysis_structures:
                    if structure not in cfg.structures_for_set(eval_set):
                        continue
                    if not masks.has(patient, eval_set, structure):
                        continue
                    mask = masks.get(patient, eval_set, structure)
                    frac = 1.0 if fully else coverage_fraction(mask, covered)
                    rows.append(
                        {
                            "patient": patient,
                            "plan_set": plan_set,
                            "eval_set": eval_set,
                            "structure": structure,
                            "coverage_fraction": frac,
                        }
                    )
                    if frac < 1.0:
                        needs_absolute = cfg.structures[structure].absolute_volume_endpoints
                        log.check(
                            "dose.structure_fully_covered", False,
                            f"{structure} of {eval_set} is {frac * 100:.3f}% inside the "
                            f"{plan_set} dose grid; absolute-volume endpoints "
                            f"(D2cc, V70cc) are invalid for this cell",
                            tier=3, warn_only=not needs_absolute,
                            patient=patient, contour_set=plan_set, structure=structure,
                            value=frac, expected="1.0",
                        )
        doses.drop_patient(patient)
        masks.drop_patient(patient)

    df = pd.DataFrame(rows)
    n_bad = int((df["coverage_fraction"] < 1.0).sum()) if not df.empty else 0
    log.check(
        "dose.all_cells_fully_covered", n_bad == 0,
        f"{n_bad} of {len(df)} (plan, contour set, structure) combinations are not "
        "fully inside their dose grid",
        tier=3, warn_only=True, value=float(n_bad),
    )
    return df


# ------------------------------------------------- native vs analysis grid, and DVH


def check_grid_equivalence(
    cfg: Config, masks: MaskStore, doses: DoseStore, log: QCLog, *, sample: int
) -> pd.DataFrame:
    """Compute the same endpoints two ways and compare.

    Native path: rasterise onto the dose grid, read dose directly.
    Analysis path: rasterise onto the analysis grid, read the resampled dose.

    V027 is excluded from the gate: its 2.0 mm CT against 2.5 mm dose grids means the
    native path drops a quarter of every structure, which is the reason the analysis
    grid exists. It is still computed and reported, as evidence of what was avoided.
    """
    rng = np.random.default_rng(20260904)
    rows = []
    for patient in cfg.patients:
        ss = load_structure_set(cfg, patient)
        grid = load_grid(cfg, patient)
        sets = [s for s in cfg.analysis_sets if cfg.is_known_absent(patient, s) is None]
        picks = rng.choice(len(sets), size=min(sample // len(cfg.patients), len(sets)),
                           replace=False)
        for i in picks:
            plan_set = sets[int(i)]
            native = load_dose(cfg, patient, plan_set)
            native_grid = grid_from_dose(native)
            arr, covered = doses.get(patient, plan_set)

            for structure in ("CTV", "Rectum", "Bladder"):
                roi = ss.get(plan_set, structure)
                if roi is None or roi.is_empty:
                    continue
                nmask, _ = rasterise(roi, native_grid)
                if not nmask.any():
                    continue
                nat = _endpoints(np.sort(native.array[nmask])[::-1], native_grid.voxel_volume_cc)

                amask = masks.get(patient, plan_set, structure)
                values = arr[amask & covered]
                if values.size == 0:
                    continue
                ana_desc = np.sort(values)[::-1]
                ana = _endpoints(ana_desc, grid.voxel_volume_cc)

                # Independent estimator on identical input.
                ref = {
                    "D98": dvh_reference.d_at_volume_pct(values, 98),
                    "D50": dvh_reference.d_at_volume_pct(values, 50),
                    "D2": dvh_reference.d_at_volume_pct(values, 2),
                    "Dmean": dvh_reference.mean_dose(values),
                    "D2cc": dvh_reference.d_at_volume_cc(values, 2.0, grid.voxel_volume_cc),
                    "V70pct": dvh_reference.v_at_dose_pct(values, 70.0),
                }

                for key in ("D98", "D50", "D2", "Dmean", "D2cc"):
                    rows.append(
                        {
                            "patient": patient, "plan_set": plan_set,
                            "structure": structure, "endpoint": key,
                            "native": nat[key], "analysis": ana[key], "reference": ref[key],
                            "native_minus_analysis": nat[key] - ana[key],
                            "analysis_minus_reference": ana[key] - ref[key],
                        }
                    )
        doses.drop_patient(patient)
        masks.drop_patient(patient)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 1. The two estimators see identical input, so they must agree tightly.
    for (patient, plan_set, structure), grp in df.groupby(
        ["patient", "plan_set", "structure"]
    ):
        worst = float(grp["analysis_minus_reference"].abs().max())
        log.check(
            "dvh.estimators_agree", worst <= ESTIMATOR_TOL_GY,
            f"sort-based and histogram-based DVH differ by at most {worst:.4f} Gy",
            tier=3, patient=str(patient), contour_set=str(plan_set),
            structure=str(structure), value=worst,
            expected=f"<= {ESTIMATOR_TOL_GY} Gy",
        )

    # 2. Native and analysis pipelines should give the same clinical answer. Reported
    #    per endpoint rather than per structure, so that a sensitive endpoint is named
    #    instead of being averaged into a worst case.
    for (patient, plan_set, structure, endpoint), grp in df.groupby(
        ["patient", "plan_set", "structure", "endpoint"]
    ):
        worst = float(grp["native_minus_analysis"].abs().max())
        log.check(
            f"dose.native_and_analysis_agree.{endpoint}",
            worst <= NATIVE_VS_ANALYSIS_TOL_GY,
            f"{endpoint} differs by {worst:.3f} Gy between the native dose grid and "
            "the analysis grid",
            tier=3, warn_only=(patient == "V027"),
            patient=str(patient), contour_set=str(plan_set), structure=str(structure),
            value=worst, expected=f"<= {NATIVE_VS_ANALYSIS_TOL_GY} Gy",
        )
    return df


# ----------------------------------------------------------------- normalisation


def check_normalisation(
    cfg: Config, masks: MaskStore, doses: DoseStore, log: QCLog
) -> pd.DataFrame:
    """Diagonal CTV D50 on the analysis grid — the definitive §7.1 check."""
    rows = []
    for patient in cfg.patients:
        grid = load_grid(cfg, patient)
        for cset in cfg.analysis_sets:
            if cfg.is_known_absent(patient, cset) is not None:
                continue
            arr, covered = doses.get(patient, cset)
            mask = masks.get(patient, cset, "CTV")
            values = arr[mask & covered]
            if values.size == 0:
                continue
            desc = np.sort(values)[::-1]
            rows.append(
                {
                    "patient": patient, "contour_set": cset,
                    "volume_cc": float(mask.sum()) * grid.voxel_volume_cc,
                    **_endpoints(desc, grid.voxel_volume_cc),
                }
            )
        doses.drop_patient(patient)
        masks.drop_patient(patient)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    centre = float(df["D50"].median())
    for _, r in df.iterrows():
        log.check(
            "dose.normalisation_consistent",
            abs(float(r["D50"]) - centre) <= NORMALISATION_TOL_GY,
            f"diagonal CTV D50 {r['D50']:.2f} Gy against a cohort median of "
            f"{centre:.2f} Gy",
            tier=3, patient=str(r["patient"]), contour_set=str(r["contour_set"]),
            structure="CTV", value=float(r["D50"]),
            expected=f"{centre:.2f} +/- {NORMALISATION_TOL_GY} Gy",
        )
    return df


# ------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=60,
                    help="plans sampled for the grid-equivalence check")
    args = ap.parse_args()

    cfg = load_config()
    run = RunInfo.create("06_qc_dose", cfg)

    resampled = cfg.results_root / "dose_resampled.parquet"
    if not resampled.exists():
        print("results/dose_resampled.parquet not found — run 05_resample_dose.py first")
        return 2
    summary = pd.read_parquet(resampled)

    log = QCLog(cfg, "06_qc_dose")
    masks, doses = MaskStore(cfg), DoseStore(cfg)

    print("Tier 3 — dose")

    for _, r in summary.iterrows():
        src = float(r["source_max_gy"])
        drop_pct = 100.0 * (src - float(r["resampled_max_gy"])) / src
        log.check(
            "dose.max_preserved_by_resampling", abs(drop_pct) <= MAX_DOSE_TOL_PCT,
            f"maximum fell {drop_pct:.3f}% on resampling "
            f"({src:.2f} -> {r['resampled_max_gy']:.2f} Gy)",
            tier=3, patient=str(r["patient"]), contour_set=str(r["contour_set"]),
            value=drop_pct, expected=f"<= {MAX_DOSE_TOL_PCT}%",
        )

    print("  dose-grid containment, per plan and structure")
    t0 = time.perf_counter()
    cover = check_containment(cfg, masks, doses, log)
    if not cover.empty:
        write_table(cover, cfg.results_root / "dose_coverage.parquet", run)
        bad = cover[cover["coverage_fraction"] < 1.0]
        print(f"    {len(cover)} combinations checked in {time.perf_counter() - t0:.0f}s; "
              f"{len(bad)} not fully covered")
        if len(bad):
            worst = bad.nsmallest(5, "coverage_fraction")
            for _, r in worst.iterrows():
                print(f"      {r['patient']} {r['plan_set']} dose vs {r['eval_set']} "
                      f"{r['structure']}: {r['coverage_fraction'] * 100:.3f}%")

    print("  native grid vs analysis grid, and DVH cross-validation")
    t0 = time.perf_counter()
    equiv = check_grid_equivalence(cfg, masks, doses, log, sample=args.sample)
    if not equiv.empty:
        write_table(equiv, cfg.results_root / "grid_equivalence.parquet", run)
        clean = equiv[equiv["patient"] != "V027"]
        print(f"    {len(equiv)} comparisons in {time.perf_counter() - t0:.0f}s")
        print(f"    native vs analysis, excluding V027: median "
              f"{clean['native_minus_analysis'].abs().median():.3f} Gy, "
              f"worst {clean['native_minus_analysis'].abs().max():.3f} Gy")
        v = equiv[equiv["patient"] == "V027"]
        if len(v):
            print(f"    V027 (native path drops ~25% of each structure): worst "
                  f"{v['native_minus_analysis'].abs().max():.3f} Gy")
        print(f"    sort-based vs histogram-based DVH: worst "
              f"{equiv['analysis_minus_reference'].abs().max():.4f} Gy")

    print("  plan normalisation on the analysis grid")
    norm = check_normalisation(cfg, masks, doses, log)
    if not norm.empty:
        write_table(norm, cfg.results_root / "diagonal_endpoints.parquet", run)
        print(f"    diagonal CTV D50: median {norm['D50'].median():.2f} Gy, range "
              f"{norm['D50'].min():.2f}-{norm['D50'].max():.2f} "
              f"(SD {norm['D50'].std():.3f}) across {len(norm)} plans")

    write_table(log.to_frame(), cfg.results_root / "qc_findings_06.parquet", run)

    print("\n" + log.summary())
    df = log.to_frame()
    warns = df[df["status"] == "WARN"]
    if len(warns):
        print(f"\nWARN by check:\n{warns['check'].value_counts().to_string()}")
    for _, r in df[df["status"] == "FAIL"].head(20).iterrows():
        scope = " ".join(
            str(r[k]) for k in ("patient", "contour_set", "structure") if pd.notna(r[k])
        )
        print(f"  FAIL [{r['check']}] {scope} — {r['message']}")

    enforce(log)
    print("\nGate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
