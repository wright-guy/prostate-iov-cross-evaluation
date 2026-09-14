"""Stage 01 — Tier 0 (inventory and integrity) and Tier 1 (provenance and mapping) QC.

    python pipelines/01_qc_raw.py [--skip-identifiability]

This is the gate that decides whether the dataset is fit to analyse at all. It runs
before any science and writes results/qc_findings_01.parquet.

Tier 1 exists because the dose folder name is the ONLY thing linking a dose to an
observer: PatientName is '<initials>_Atlas_<patient>_CT' on every file. The self-plan
identifiability test verifies that mapping from the dosimetry itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import Config, load_config  # noqa: E402
from mcx.endpoints.dvh import d_at_volume_pct, dose_in_mask  # noqa: E402
from mcx.geom.rasterise import grid_from_dose, rasterise  # noqa: E402
from mcx.io.ct import load_ct_series  # noqa: E402
from mcx.io.rtdose import find_dose, read_dose  # noqa: E402
from mcx.io.rtstruct import load_structure_set  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402
from mcx.qc.checks import Finding, QCLog, Status, enforce  # noqa: E402
from mcx.qc.identifiability import assess  # noqa: E402

# Significance level for the per-patient permutation test that the dose folder labels
# really do identify the contour set each plan was optimised on.
IDENTIFIABILITY_ALPHA = 0.001

# Diagonal CTV D50 defines the plan normalisation. This is the half-width of the band
# the whole cohort must sit inside for §7.1's "identical normalisation" claim to hold.
NORMALISATION_TOL_GY = 1.0


# --------------------------------------------------------------------------- Tier 0


def tier0(cfg: Config, inv: pd.DataFrame, log: QCLog) -> None:
    print("Tier 0 — inventory and integrity")

    # -- CT series geometry
    ct_for = {}
    for patient in cfg.patients:
        try:
            ct = load_ct_series(cfg, patient)
        except (FileNotFoundError, ValueError) as exc:
            log.check("ct.series_loads", False, str(exc), tier=0, patient=patient)
            continue
        ct_for[patient] = ct
        log.check(
            "ct.series_loads", True,
            f"{ct.n_slices} slices, {ct.slice_spacing:.2f} mm, {ct.manufacturer}",
            tier=0, patient=patient, value=float(ct.n_slices),
        )
        log.check(
            "ct.spacing_uniform", ct.spacing_uniformity_mm() < 1e-3,
            f"slice-spacing peak-to-peak {ct.spacing_uniformity_mm():.4f} mm",
            tier=0, patient=patient, value=ct.spacing_uniformity_mm(), expected="< 0.001 mm",
        )
        log.check(
            "ct.no_duplicate_slices", ct.duplicate_z_count() == 0,
            f"{ct.duplicate_z_count()} duplicate slice position(s)",
            tier=0, patient=patient, value=float(ct.duplicate_z_count()), expected="0",
        )
        log.check(
            "ct.head_first_supine", ct.patient_position == "HFS",
            f"PatientPosition = {ct.patient_position!r}",
            tier=0, patient=patient, expected="HFS",
        )

    # -- Frame of reference agreement across CT, RTSTRUCT and RTDOSE
    for patient, ct in ct_for.items():
        ss = load_structure_set(cfg, patient)
        log.check(
            "for.struct_matches_ct", ss.frame_of_reference_uid == ct.frame_of_reference_uid,
            "RTSTRUCT frame of reference matches the CT series",
            tier=0, patient=patient,
        )
        for cset in cfg.all_sets:
            path = find_dose(cfg, patient, cset)
            absent_reason = cfg.is_known_absent(patient, cset)
            if path is None:
                log.add(
                    Finding(
                        check="dose.present",
                        status=Status.PASS if absent_reason else Status.FAIL,
                        message=absent_reason or "no RTDOSE found",
                        tier=0, patient=patient, contour_set=cset,
                    )
                )
                continue
            dose = read_dose(path)
            log.check(
                "dose.present", True, f"{path.name} ({dose.max_gy:.2f} Gy max)",
                tier=0, patient=patient, contour_set=cset, value=dose.max_gy,
            )
            log.check(
                "for.dose_matches_ct",
                dose.frame_of_reference_uid == ct.frame_of_reference_uid,
                "RTDOSE frame of reference matches the CT series",
                tier=0, patient=patient, contour_set=cset,
            )
            log.check(
                "dose.units_gy", dose.units == "GY", f"DoseUnits = {dose.units!r}",
                tier=0, patient=patient, contour_set=cset, expected="GY",
            )
            log.check(
                "dose.type_physical", dose.dose_type == "PHYSICAL",
                f"DoseType = {dose.dose_type!r}",
                tier=0, patient=patient, contour_set=cset, expected="PHYSICAL",
            )
            log.check(
                "dose.summation_type_is_plan", dose.summation_type == "PLAN",
                f"DoseSummationType = {dose.summation_type!r} "
                "(single-arc plans; see config/waivers.yaml)",
                tier=0, patient=patient, contour_set=cset, expected="PLAN",
            )

    # -- ROI availability against the expected grid
    for patient in cfg.patients:
        ss = load_structure_set(cfg, patient)
        for cset in cfg.all_sets:
            expected = cfg.structures_for_set(cset)
            complete = set(expected) == set(cfg.structures)
            log.check(
                "struct.roi_set_complete", complete,
                f"{cset} carries {len(expected)}/{len(cfg.structures)} structures",
                tier=0, patient=patient, contour_set=cset,
            )
            for structure in expected:
                name = cfg.roi_name(patient, cset, structure)
                roi = ss.roi(name)
                present = roi is not None and not roi.is_empty
                log.check(
                    "struct.roi_present", present,
                    f"{name}: {'present' if present else 'MISSING or empty'}",
                    tier=0, patient=patient, contour_set=cset, structure=structure,
                )
                # Duplicate detection, e.g. K042's Rectum2_O7.
                prefix = cfg.structures[structure].roi_template.split("{")[0].rstrip("_")
                dupes = [n for n in ss.matching_rois(prefix, cfg.raw_set(cset))
                         if n.lower() != name.lower()]
                log.check(
                    "struct.no_duplicate_roi", not dupes,
                    f"additional ROI(s) matching {prefix}*_{cset}: {dupes}",
                    tier=0, warn_only=True, patient=patient,
                    contour_set=cset, structure=structure,
                )

    # -- One dose per pair, and one plan per dose
    doses = inv[inv["modality"] == "RTDOSE"]
    log.check(
        "dose.count", len(doses) == len(cfg.expected_pairs()),
        f"{len(doses)} RTDOSE files for {len(cfg.expected_pairs())} expected pairs",
        tier=0, value=float(len(doses)), expected=str(len(cfg.expected_pairs())),
    )
    n_unique_plans = doses["referenced_rtplan_uid"].nunique(dropna=True)
    log.check(
        "dose.one_plan_each", n_unique_plans == len(doses),
        f"{n_unique_plans} distinct referenced plan UIDs across {len(doses)} doses",
        tier=0, value=float(n_unique_plans), expected=str(len(doses)),
    )


# --------------------------------------------------------------------------- Tier 1


def tier1(
    cfg: Config, inv: pd.DataFrame, log: QCLog, *, identifiability: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Tier 1 — provenance and mapping")

    # -- The plans reference structure sets that were not exported with this dataset.
    supplied = set(inv.loc[inv["modality"] == "RTSTRUCT", "sop_instance_uid"])
    referenced = set(inv["referenced_structure_set_uid"].dropna())
    if referenced:
        log.check(
            "struct.plan_referenced_structure_set_present", referenced <= supplied,
            f"{len(referenced - supplied)} of {len(referenced)} plan-referenced structure "
            "sets are absent from this dataset",
            tier=1, value=float(len(referenced - supplied)),
        )

    if not identifiability:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, object]] = []
    for patient in cfg.patients:
        ss = load_structure_set(cfg, patient)
        sets = [s for s in cfg.analysis_sets if cfg.is_known_absent(patient, s) is None]
        print(f"  {patient}: {len(sets)} plans x {len(sets)} contour sets")

        for plan_set in sets:
            dose = read_dose(find_dose(cfg, patient, plan_set))  # type: ignore[arg-type]
            grid = grid_from_dose(dose)

            for eval_set in sets:
                for structure in ("PTV", "CTV"):
                    roi = ss.get(eval_set, structure)
                    if roi is None or roi.is_empty:
                        continue
                    mask, rep = rasterise(roi, grid)
                    if not mask.any():
                        continue
                    if plan_set == eval_set:
                        # Rasterising onto the NATIVE dose grid is lossy wherever the
                        # CT slice spacing differs from the dose grid's. Recorded here
                        # so the Tier 1 result carries its own caveat; stage 05 moves
                        # everything onto a common CT-derived grid where this is moot.
                        log.check(
                            "raster.no_slices_dropped_on_dose_grid",
                            rep.n_loops_off_grid_z == 0 and rep.n_slices_z_collapsed == 0,
                            f"{rep.n_loops_off_grid_z} of {rep.n_loops} contours fall "
                            f"between dose-grid planes, {rep.n_slices_z_collapsed} "
                            f"collision(s), {rep.z_gap_slices} resulting gap(s); "
                            f"mask {rep.mask_volume_cc:.2f} cc vs polygon "
                            f"{rep.polygon_volume_cc:.2f} cc "
                            f"({100 * rep.volume_agreement:.1f}% apart)",
                            tier=1, patient=patient, contour_set=eval_set,
                            structure=structure, value=float(rep.volume_agreement),
                            expected="0 dropped contours",
                        )
                    d = dose_in_mask(dose.array, mask)
                    rows.append(
                        {
                            "patient": patient,
                            "plan_set": plan_set,
                            "eval_set": eval_set,
                            "structure": structure,
                            "is_diagonal": plan_set == eval_set,
                            "volume_cc": float(mask.sum()) * grid.voxel_volume_cc,
                            "D98": d_at_volume_pct(d, 98),
                            "D50": d_at_volume_pct(d, 50),
                            "loops_clipped_xy": rep.n_loops_clipped_xy,
                        }
                    )

    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return matrix, pd.DataFrame()

    # -- Self-plan identifiability, by permutation test. See mcx.qc.identifiability
    #    for why this is column-oriented and why there is no hand-chosen threshold.
    ptv = matrix[matrix["structure"] == "PTV"]
    per_column = []
    for patient, grp in ptv.groupby("patient"):
        res = assess(grp, patient=str(patient), metric="D98")
        per_column.append(res.per_column)
        log.check(
            "map.self_plan_identifiable", res.p_value < IDENTIFIABILITY_ALPHA,
            f"diagonal z = {res.statistic:+.2f}, p = {res.p_value:.2e} "
            f"({res.n_columns_won}/{res.n_sets} columns won, "
            f"worst margin {res.worst_margin_gy:.2f} Gy)",
            tier=1, patient=str(patient),
            value=res.p_value, expected=f"p < {IDENTIFIABILITY_ALPHA}",
        )
        print(
            f"  {patient}: diagonal z = {res.statistic:+.2f}, p = {res.p_value:.2e}, "
            f"{res.n_columns_won}/{res.n_sets} columns won"
        )
        # Individual columns the diagonal does not win are informative about plan
        # quality, not about the mapping, so they warn rather than fail.
        for _, r in res.per_column[res.per_column["diagonal_rank"] > 1].iterrows():
            log.check(
                "map.diagonal_wins_column", False,
                f"on {r['eval_set']}'s PTV, the {r['column_argmax']} plan beats the "
                f"{r['eval_set']} plan by {r['margin_gy']:.2f} Gy "
                f"(diagonal rank {int(r['diagonal_rank'])} of {res.n_sets})",
                tier=1, warn_only=True, patient=str(patient),
                contour_set=str(r["eval_set"]), value=float(r["margin_gy"]),
            )
    detail = pd.concat(per_column, ignore_index=True) if per_column else pd.DataFrame()

    # -- Plan normalisation consistency: diagonal CTV D50 across the whole cohort.
    diag_ctv = matrix[(matrix["structure"] == "CTV") & matrix["is_diagonal"]]
    if not diag_ctv.empty:
        centre = float(diag_ctv["D50"].median())
        for _, r in diag_ctv.iterrows():
            log.check(
                "dose.normalisation_consistent",
                abs(float(r["D50"]) - centre) <= NORMALISATION_TOL_GY,
                f"diagonal CTV D50 {r['D50']:.2f} Gy vs cohort median {centre:.2f} Gy",
                tier=1, patient=str(r["patient"]), contour_set=str(r["plan_set"]),
                value=float(r["D50"]), expected=f"{centre:.2f} +/- {NORMALISATION_TOL_GY} Gy",
            )
        spread = float(diag_ctv["D50"].max() - diag_ctv["D50"].min())
        print(
            f"  diagonal CTV D50: median {centre:.2f} Gy, "
            f"range {diag_ctv['D50'].min():.2f}-{diag_ctv['D50'].max():.2f} "
            f"(spread {spread:.2f} Gy, SD {diag_ctv['D50'].std():.3f})"
        )

    # -- PTV must contain its own CTV.
    for (patient, cset), grp in matrix[matrix["is_diagonal"]].groupby(["patient", "plan_set"]):
        vols = grp.set_index("structure")["volume_cc"]
        if {"CTV", "PTV"} <= set(vols.index):
            ratio = float(vols["PTV"] / vols["CTV"])
            log.check(
                "struct.ptv_encloses_ctv", ratio > 1.0,
                f"PTV/CTV volume ratio {ratio:.2f}",
                tier=1, patient=patient, contour_set=cset, value=ratio, expected="> 1",
            )

    return matrix, detail


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-identifiability",
        action="store_true",
        help="skip the Tier 1 dosimetric mapping test (fast Tier 0 only)",
    )
    args = ap.parse_args()

    cfg = load_config()
    run = RunInfo.create("01_qc_raw", cfg)

    inv_path = cfg.results_root / "inventory.parquet"
    if not inv_path.exists():
        print("results/inventory.parquet not found — run pipelines/00_index_dicom.py first")
        return 2
    inv = pd.read_parquet(inv_path)

    log = QCLog(cfg, "01_qc_raw")
    tier0(cfg, inv, log)
    matrix, detail = tier1(cfg, inv, log, identifiability=not args.skip_identifiability)

    write_table(log.to_frame(), cfg.results_root / "qc_findings_01.parquet", run)
    if not matrix.empty:
        write_table(matrix, cfg.results_root / "tier1_cross_matrix.parquet", run)
    if not detail.empty:
        write_table(detail, cfg.results_root / "tier1_identifiability.parquet", run)

    print("\n" + log.summary())
    df = log.to_frame()
    for status in ("FAIL", "WARN"):
        sub = df[df["status"] == status]
        if len(sub):
            print(f"\n{status}:")
            for _, r in sub.iterrows():
                scope = " ".join(
                    str(r[k]) for k in ("patient", "contour_set", "structure") if pd.notna(r[k])
                )
                print(f"  [{r['check']}] {scope} — {r['message']}")

    enforce(log)
    print("\nGate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
