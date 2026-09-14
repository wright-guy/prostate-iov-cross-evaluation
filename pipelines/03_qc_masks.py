"""Stage 03 — Tier 2 QC on the rasterised masks.

    python pipelines/03_qc_masks.py [--no-montages] [--no-convergence]

Gates on the analysis structures (CTV, PTV, Rectum, Bladder) and warns on the
descriptive ones. Colon and bowel bag deliberately extend beyond the analysis grid,
which is set by the analysis structures plus a margin; failing on them would be
failing on a design choice rather than on a defect.

Also measures the PTV expansion margin back out of the data, runs a grid-convergence
sweep so the 1 mm choice is evidenced rather than asserted, and writes visual montages.
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
from mcx.geom.analysis_grid import load_grid  # noqa: E402
from mcx.geom.grid import Grid  # noqa: E402
from mcx.geom.masks import MaskStore  # noqa: E402
from mcx.geom.rasterise import rasterise  # noqa: E402
from mcx.geom.surface import expansion_distance, hausdorff_percentile  # noqa: E402
from mcx.io.ct import load_ct_series  # noqa: E402
from mcx.io.rtstruct import load_structure_set  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402
from mcx.qc.checks import QCLog, enforce  # noqa: E402
from mcx.qc.montage import ct_volume_on_grid, structure_montage  # noqa: E402

# Robust z-score above which an observer's volume is flagged against the cohort.
# Modified z from the median absolute deviation, so it is not dragged by the outlier
# it is trying to find.
OUTLIER_MODIFIED_Z = 3.5


def _tolerances(cfg: Config) -> tuple[float, float]:
    va = cfg.grid_raw["volume_agreement"]
    return float(va["warn_above"]), float(va["fail_above"])


# --------------------------------------------------------------- per-mask checks


def check_masks(cfg: Config, structures: pd.DataFrame, log: QCLog) -> None:
    warn_above, fail_above = _tolerances(cfg)
    analysis = set(cfg.analysis_structures)

    present = structures[structures["present"]]
    for _, r in present.iterrows():
        scope = {
            "patient": str(r["patient"]),
            "contour_set": str(r["contour_set"]),
            "structure": str(r["structure"]),
        }
        # Descriptive structures may legitimately fall outside the analysis grid.
        is_analysis = r["structure"] in analysis
        soft = not is_analysis

        agreement = float(r["volume_agreement"])
        log.check(
            "raster.volume_roundtrip",
            not np.isnan(agreement) and agreement <= fail_above,
            f"mask {r['volume_cc']:.2f} cc vs polygon {r['polygon_volume_cc']:.2f} cc "
            f"({agreement * 100:.2f}% apart)",
            tier=2, warn_only=soft or agreement <= warn_above,
            value=agreement, expected=f"<= {fail_above * 100:.0f}%", **scope,
        )
        log.check(
            "raster.contained_in_grid",
            int(r["n_loops_clipped_xy"]) == 0 and int(r["n_loops_off_grid_z"]) == 0,
            f"{int(r['n_loops_clipped_xy'])} loop(s) clipped in plane, "
            f"{int(r['n_loops_off_grid_z'])} outside the grid in z",
            tier=2, warn_only=soft, **scope,
        )
        # A collision means two source slices landed on one grid plane. On this grid
        # that must never happen, because the z planes are the CT slice positions.
        log.check(
            "raster.no_slice_collision", int(r["n_slices_z_collapsed"]) == 0,
            f"{int(r['n_slices_z_collapsed'])} grid plane(s) received more than one "
            "source slice",
            tier=2, **scope,
        )
        log.check(
            "raster.no_internal_gap", int(r["z_gap_slices"]) == 0,
            f"{int(r['z_gap_slices'])} empty slice(s) inside the structure's extent",
            tier=2, warn_only=soft, **scope,
        )
        log.check(
            "raster.no_degenerate_loops", int(r["n_loops_degenerate"]) == 0,
            f"{int(r['n_loops_degenerate'])} loop(s) with fewer than 3 vertices; "
            f"{float(r['area_dropped_mm2']):.3f} mm2 of polygon area discarded in total",
            tier=2, warn_only=soft, value=float(r["area_dropped_mm2"]), **scope,
        )
        log.check(
            "raster.no_subvoxel_slivers", int(r["n_loops_zero_area"]) == 0,
            f"{int(r['n_loops_zero_area'])} loop(s) narrower than one voxel; "
            f"{float(r['area_dropped_mm2']):.3f} mm2 of polygon area discarded in total",
            tier=2, warn_only=soft, value=float(r["area_dropped_mm2"]), **scope,
        )

        lo, hi = cfg.structures[str(r["structure"])].plausible_volume_cc
        vol = float(r["volume_cc"])
        log.check(
            "struct.volume_plausible", lo <= vol <= hi,
            f"{vol:.1f} cc outside the plausible band {lo:.0f}-{hi:.0f} cc",
            tier=2, warn_only=soft, value=vol, expected=f"{lo:.0f}-{hi:.0f} cc", **scope,
        )


def check_cohort_outliers(cfg: Config, structures: pd.DataFrame, log: QCLog) -> None:
    """Flag an observer whose volume sits far from the other nine, per structure."""
    obs = structures[structures["present"] & structures["contour_set"].isin(cfg.observers)]
    for (patient, structure), grp in obs.groupby(["patient", "structure"]):
        v = grp["volume_cc"].to_numpy(dtype=float)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if mad == 0:
            continue
        z = 0.6745 * (v - med) / mad
        for cset, zi, vi in zip(grp["contour_set"], z, v, strict=True):
            log.check(
                "struct.cohort_volume_outlier", abs(zi) <= OUTLIER_MODIFIED_Z,
                f"{vi:.1f} cc against a cohort median of {med:.1f} cc "
                f"(modified z = {zi:+.1f})",
                tier=2, warn_only=True, value=float(zi),
                patient=str(patient), contour_set=str(cset), structure=str(structure),
            )


# ------------------------------------------------------------------- PTV margin


def measure_ptv_margin(cfg: Config, store: MaskStore, log: QCLog) -> pd.DataFrame:
    """Measure the CTV->PTV expansion back out of the masks.

    The user states the PTVs are isotropic expansions of the CTV. If that is right the
    distance from the CTV surface to the PTV surface is tight around one value, and
    that value is the margin -- which is not recorded anywhere in the DICOM.
    """
    rows = []
    for patient in cfg.patients:
        grid = load_grid(cfg, patient)
        for cset in cfg.all_sets:
            if cfg.is_known_absent(patient, cset) is not None:
                continue
            if not (store.has(patient, cset, "CTV") and store.has(patient, cset, "PTV")):
                continue
            ctv = store.get(patient, cset, "CTV")
            ptv = store.get(patient, cset, "PTV")
            d = expansion_distance(ctv, ptv, grid)
            rows.append({"patient": patient, "contour_set": cset, **d})
            log.check(
                "struct.ptv_encloses_ctv", bool(d["contained"]),
                "CTV is fully inside its PTV",
                tier=2, patient=patient, contour_set=cset, structure="PTV",
            )
        store.drop_patient(patient)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- convergence


def convergence_sweep(cfg: Config, log: QCLog) -> pd.DataFrame:
    """Recompute volume and HD95 across the resolution ladder.

    Evidence for the 1 mm in-plane choice. z spacing stays native throughout, because
    changing it would mean interpolating contours that only exist at CT slice
    positions -- see config/grid.yaml.
    """
    ladder = [float(v) for v in cfg.grid_raw["convergence_ladder_mm"]]
    rows = []
    for patient in cfg.patients:
        ss = load_structure_set(cfg, patient)
        base = load_grid(cfg, patient)
        lo, _ = base.bounds()
        for spacing in ladder:
            grid = Grid(
                origin=base.origin,
                spacing=np.array([spacing, spacing, base.spacing[2]]),
                shape=(
                    base.shape[0],
                    int(np.ceil(base.shape[1] * base.spacing[1] / spacing)) + 1,
                    int(np.ceil(base.shape[2] * base.spacing[0] / spacing)) + 1,
                ),
            )
            for structure in ("CTV", "Rectum", "Bladder"):
                a = ss.get("O5", structure)
                b = ss.get("O10", structure)
                if a is None or b is None or a.is_empty or b.is_empty:
                    continue
                ma, _ = rasterise(a, grid)
                mb, _ = rasterise(b, grid)
                rows.append(
                    {
                        "patient": patient,
                        "in_plane_mm": spacing,
                        "structure": structure,
                        "volume_cc": float(ma.sum()) * grid.voxel_volume_cc,
                        "hd95_mm": hausdorff_percentile(ma, mb, grid),
                    }
                )
    conv = pd.DataFrame(rows)
    if conv.empty:
        return conv

    # The 1 mm result must be close to the finest rung, or 1 mm is not converged.
    finest = min(ladder)
    chosen = float(cfg.grid_raw["analysis_grid"]["in_plane_mm"])
    for (patient, structure), grp in conv.groupby(["patient", "structure"]):
        g = grp.set_index("in_plane_mm")
        if finest not in g.index or chosen not in g.index:
            continue
        dv = abs(g.loc[chosen, "volume_cc"] - g.loc[finest, "volume_cc"])
        rel = dv / g.loc[finest, "volume_cc"]
        log.check(
            "grid.volume_converged", rel < 0.01,
            f"volume at {chosen} mm differs from {finest} mm by {rel * 100:.2f}%",
            tier=2, warn_only=True, value=float(rel),
            patient=str(patient), structure=str(structure), expected="< 1%",
        )
    return conv


# ------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-montages", action="store_true")
    ap.add_argument("--no-convergence", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    run = RunInfo.create("03_qc_masks", cfg)

    path = cfg.results_root / "structures.parquet"
    if not path.exists():
        print("results/structures.parquet not found — run pipelines/02_build_masks.py first")
        return 2
    structures = pd.read_parquet(path)

    log = QCLog(cfg, "03_qc_masks")
    store = MaskStore(cfg)

    print("Tier 2 — contour geometry")
    check_masks(cfg, structures, log)
    check_cohort_outliers(cfg, structures, log)

    print("  measuring the CTV->PTV expansion")
    margins = measure_ptv_margin(cfg, store, log)
    if not margins.empty:
        write_table(margins, cfg.results_root / "ptv_margin.parquet", run)
        print(
            f"    median expansion {margins['median'].median():.2f} mm "
            f"(5th-95th {margins['p05'].median():.2f}-{margins['p95'].median():.2f} mm, "
            f"median IQR {margins['iqr'].median():.2f} mm across "
            f"{len(margins)} contour sets)"
        )

    conv = pd.DataFrame()
    if not args.no_convergence:
        print("  grid convergence sweep")
        t0 = time.perf_counter()
        conv = convergence_sweep(cfg, log)
        if not conv.empty:
            write_table(conv, cfg.results_root / "grid_convergence.parquet", run)
            piv = conv.pivot_table(index="in_plane_mm", columns="structure",
                                   values="volume_cc", aggfunc="sum")
            print(f"    total volume (cc) by in-plane spacing, {time.perf_counter() - t0:.0f}s")
            print(piv.to_string(float_format=lambda v: f"{v:8.2f}"))

    if not args.no_montages:
        print("  montages")
        for patient in cfg.patients:
            grid = load_grid(cfg, patient)
            ct = ct_volume_on_grid(load_ct_series(cfg, patient), grid)
            for structure in cfg.analysis_structures + ["SeminalVesicles"]:
                out = cfg.qc_root / "montages" / f"{patient}_{structure}.png"
                if structure_montage(cfg, patient, structure, grid, store, ct, out):
                    print(f"    {out.relative_to(cfg.qc_root.parent)}")
            store.drop_patient(patient)

    write_table(log.to_frame(), cfg.results_root / "qc_findings_03.parquet", run)

    print("\n" + log.summary())
    df = log.to_frame()
    warns = df[df["status"] == "WARN"]
    if len(warns):
        print(f"\nWARN by check:\n{warns['check'].value_counts().to_string()}")
    for _, r in df[df["status"] == "FAIL"].iterrows():
        scope = " ".join(
            str(r[k]) for k in ("patient", "contour_set", "structure") if pd.notna(r[k])
        )
        print(f"  FAIL [{r['check']}] {scope} — {r['message']}")

    enforce(log)
    print("\nGate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
