"""Stage 04 — build every structure that is derived rather than drawn.

    python pipelines/04_derived_structures.py [--overwrite] [--patient K018]

Produces, per patient:
  * leave-one-out consensus contours, one per held-out observer, for CTV, PTV, Rectum
    and Bladder -- what H1's 1.34 benchmark needs
  * a reproduction check of the supplied VOTE against a plain majority vote
  * CTV expansion shells at +3, +5 and +7 mm for the coverage-degradation ladder
  * the rectum PRV (3 mm), rectum-inside-PTV and rectum-outside-PTV, built from the
    evaluation-side anatomy only
  * peripheral tissue: EXTERNAL minus PTV + 35 mm
  * the ICRU 83 treated volume (95% isodose) for each plan

Writes results/consensus_reproduction.parquet and results/derived_structures.parquet.
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
from mcx.dose.store import DoseStore  # noqa: E402
from mcx.geom.analysis_grid import load_grid  # noqa: E402
from mcx.geom.consensus import (  # noqa: E402
    find_threshold,
    leave_one_out_votes,
    majority_vote,
    signed_distance_bias,
)
from mcx.geom.grid import Grid  # noqa: E402
from mcx.geom.masks import MaskStore, load_mask, mask_path, save_mask  # noqa: E402
from mcx.geom.rasterise import rasterise  # noqa: E402
from mcx.geom.shells import expand, subtract  # noqa: E402
from mcx.io.rtstruct import load_structure_set  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402

CONSENSUS_STRUCTURES = ("CTV", "PTV", "Rectum", "Bladder")
LADDER_MM = (3.0, 5.0, 7.0)
RECTUM_PRV_MM = 3.0
PERIPHERAL_MARGIN_MM = 35.0

# The reproduction is judged against the supplied VOTE. It will not be exact: the
# supplied contours went through a mask -> contour -> mask round trip that a
# reimplementation does not. What matters is that the disagreement is a thin surface
# skin rather than bulk, which is checked here and reported.
REPRODUCTION_MIN_DICE = 0.90


def _put(cfg: Config, patient: str, cset: str, structure: str, mask: np.ndarray,
         *, overwrite: bool) -> bool:
    path = mask_path(cfg, patient, cset, structure)
    if path.exists() and not overwrite:
        return False
    save_mask(path, mask)
    return True


def _get(cfg: Config, patient: str, cset: str, structure: str) -> np.ndarray:
    return load_mask(mask_path(cfg, patient, cset, structure))


def external_mask(cfg: Config, patient: str, grid: Grid, *, overwrite: bool) -> np.ndarray:
    """The patient-level EXTERNAL contour, cached under the pseudo-set 'PATIENT'."""
    path = mask_path(cfg, patient, "PATIENT", "External")
    if path.exists() and not overwrite:
        return load_mask(path)
    ss = load_structure_set(cfg, patient)
    roi = ss.roi("EXTERNAL")
    if roi is None or roi.is_empty:
        raise FileNotFoundError(f"{patient}: no EXTERNAL contour")
    mask, _ = rasterise(roi, grid)
    save_mask(path, mask)
    return mask


def build_for_patient(
    cfg: Config, patient: str, store: MaskStore, doses: DoseStore, *, overwrite: bool
) -> tuple[list[dict], list[dict]]:
    grid = load_grid(cfg, patient)
    vox = grid.voxel_volume_cc
    observers = [o for o in cfg.observers if cfg.is_known_absent(patient, o) is None]
    isodose = float(cfg.endpoints_raw["isodose_95pct_gy"])

    repro_rows: list[dict] = []
    derived_rows: list[dict] = []

    # ---- consensus: verify the algorithm, then build leave-one-out variants
    for structure in CONSENSUS_STRUCTURES:
        available = [o for o in observers if store.has(patient, o, structure)]
        if not available or not store.has(patient, "VOTE", structure):
            continue
        masks = {o: store.get(patient, o, structure) for o in available}
        supplied = store.get(patient, "VOTE", structure)

        res = find_threshold(
            list(masks.values()), supplied,
            patient=patient, structure=structure, voxel_volume_cc=vox,
        )
        strict = len(available) // 2 + 1
        reproduced = majority_vote(list(masks.values()))
        bias = signed_distance_bias(list(masks.values()), supplied, grid)
        repro_rows.append({
            "patient": patient,
            "structure": structure,
            "n_observers": len(available),
            "strict_majority_k": strict,
            "best_matching_k": res.best_threshold,
            "matches_strict_majority": res.best_threshold == strict,
            "dice_at_strict_majority": res.dice_by_threshold[strict],
            "dice_at_best": res.best_dice,
            "supplied_volume_cc": res.supplied_volume_cc,
            "reproduced_volume_cc": float(reproduced.sum()) * vox,
            "vote_bias_mean_signed_mm": bias["mean_signed_mm"],
            "vote_bias_sd_signed_mm": bias["sd_signed_mm"],
            "vote_volume_over_mean_observer": bias["volume_ratio"],
        })

        for held_out, loo in leave_one_out_votes(masks).items():
            _put(cfg, patient, f"LOO_{held_out}", structure, loo, overwrite=overwrite)
            derived_rows.append({
                "patient": patient, "contour_set": f"LOO_{held_out}",
                "structure": structure, "kind": "leave_one_out_consensus",
                "volume_cc": float(loo.sum()) * vox, "held_out": held_out,
            })

    # ---- coverage ladder, rectum PRV and boolean combinations, per contour set
    for cset in [*observers, "VOTE"]:
        if cfg.is_known_absent(patient, cset) is not None:
            continue
        if store.has(patient, cset, "CTV"):
            ctv = store.get(patient, cset, "CTV")
            for mm in LADDER_MM:
                name = f"CTV_plus_{int(mm)}mm"
                m = expand(ctv, grid, mm)
                _put(cfg, patient, cset, name, m, overwrite=overwrite)
                derived_rows.append({
                    "patient": patient, "contour_set": cset, "structure": name,
                    "kind": "expansion_shell", "volume_cc": float(m.sum()) * vox,
                    "margin_mm": mm,
                })
        if store.has(patient, cset, "Rectum"):
            rectum = store.get(patient, cset, "Rectum")
            prv = expand(rectum, grid, RECTUM_PRV_MM)
            _put(cfg, patient, cset, "Rectum_PRV_3mm", prv, overwrite=overwrite)
            derived_rows.append({
                "patient": patient, "contour_set": cset, "structure": "Rectum_PRV_3mm",
                "kind": "prv", "volume_cc": float(prv.sum()) * vox,
                "margin_mm": RECTUM_PRV_MM,
            })
            if store.has(patient, cset, "PTV"):
                ptv = store.get(patient, cset, "PTV")
                # Evaluation-side only: j's rectum with j's PTV, never mixed with i's.
                inside = rectum & ptv
                outside = subtract(rectum, ptv)
                for name, m in (("Rectum_in_PTV", inside), ("Rectum_outside_PTV", outside)):
                    _put(cfg, patient, cset, name, m, overwrite=overwrite)
                    derived_rows.append({
                        "patient": patient, "contour_set": cset, "structure": name,
                        "kind": "boolean", "volume_cc": float(m.sum()) * vox,
                    })

    # ---- peripheral tissue: EXTERNAL minus PTV + 35 mm
    ext = external_mask(cfg, patient, grid, overwrite=overwrite)
    for cset in [*observers, "VOTE"]:
        if cfg.is_known_absent(patient, cset) is not None or not store.has(patient, cset, "PTV"):
            continue
        peripheral = subtract(ext, expand(store.get(patient, cset, "PTV"), grid,
                                          PERIPHERAL_MARGIN_MM))
        _put(cfg, patient, cset, "Peripheral_Tissue", peripheral, overwrite=overwrite)
        derived_rows.append({
            "patient": patient, "contour_set": cset, "structure": "Peripheral_Tissue",
            "kind": "boolean", "volume_cc": float(peripheral.sum()) * vox,
        })

    # ---- treated volume, one per plan (a property of the plan, not of any contour)
    for cset in cfg.analysis_sets:
        if cfg.is_known_absent(patient, cset) is not None:
            continue
        dose, covered = doses.get(patient, cset)
        tv = covered & (np.nan_to_num(dose, nan=-1.0) >= isodose)
        _put(cfg, patient, cset, "Treated_Volume", tv, overwrite=overwrite)
        derived_rows.append({
            "patient": patient, "contour_set": cset, "structure": "Treated_Volume",
            "kind": "treated_volume", "volume_cc": float(tv.sum()) * vox,
            "isodose_gy": isodose,
        })

    store.drop_patient(patient)
    doses.drop_patient(patient)
    return repro_rows, derived_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--patient", action="append")
    args = ap.parse_args()

    cfg = load_config()
    run = RunInfo.create("04_derived_structures", cfg)
    store, doses = MaskStore(cfg), DoseStore(cfg)

    repro, derived = [], []
    for patient in (args.patient or cfg.patients):
        t0 = time.perf_counter()
        r, d = build_for_patient(cfg, patient, store, doses, overwrite=args.overwrite)
        repro.extend(r)
        derived.extend(d)
        print(f"  {patient}: {len(d)} derived structures, {time.perf_counter() - t0:.1f}s")

    rp, dv = pd.DataFrame(repro), pd.DataFrame(derived)
    write_table(rp, cfg.results_root / "consensus_reproduction.parquet", run)
    write_table(dv, cfg.results_root / "derived_structures.parquet", run)

    print("\n=== consensus reproduction ===")
    print(rp[["patient", "structure", "n_observers", "strict_majority_k",
              "best_matching_k", "matches_strict_majority", "dice_at_strict_majority",
              "vote_bias_mean_signed_mm", "vote_volume_over_mean_observer"]]
          .round(4).to_string(index=False))
    ok = int(rp["matches_strict_majority"].sum())
    print(f"\nstrict majority is the best-matching threshold in {ok}/{len(rp)} cases")
    print(f"Dice at strict majority: min {rp['dice_at_strict_majority'].min():.4f}, "
          f"median {rp['dice_at_strict_majority'].median():.4f}")
    if rp["dice_at_strict_majority"].min() < REPRODUCTION_MIN_DICE:
        print("WARNING: reproduction below tolerance for at least one structure")

    print("\n=== derived structures ===")
    print(dv.groupby("kind")["volume_cc"].agg(["count", "median"]).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
