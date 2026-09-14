"""Stage 02 — build the analysis grid and rasterise every ROI onto it.

    python pipelines/02_build_masks.py [--overwrite] [--patient K018]

Writes results/structures.parquet and results/grids.parquet, and caches every mask
under the derived root (outside OneDrive by default).

This is the stage that makes cross-evaluation cheap: after it, every contour set and
every dose for a patient share one geometry, so a cell is array arithmetic.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import load_config  # noqa: E402
from mcx.geom.analysis_grid import build_analysis_grid, save_grid  # noqa: E402
from mcx.geom.masks import build_masks_for_patient, mask_dir  # noqa: E402
from mcx.io.rtstruct import load_structure_set  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true", help="rebuild cached masks")
    ap.add_argument("--patient", action="append", help="restrict to these patients")
    args = ap.parse_args()

    cfg = load_config()
    run = RunInfo.create("02_build_masks", cfg)
    patients = args.patient or cfg.patients

    print(f"analysis grid: {cfg.grid_raw['analysis_grid']['in_plane_mm']} mm in plane, "
          f"native CT slice spacing in z, "
          f"{cfg.grid_raw['analysis_grid']['margin_mm']} mm margin")
    print(f"mask cache: {mask_dir(cfg, '<patient>')}\n")

    grid_rows, struct_rows = [], []
    for patient in patients:
        t0 = time.perf_counter()
        ss = load_structure_set(cfg, patient)
        grid = build_analysis_grid(cfg, patient, ss)
        save_grid(cfg, patient, grid)

        rows = build_masks_for_patient(cfg, patient, ss, grid, overwrite=args.overwrite)
        struct_rows.extend(rows)

        nz, ny, nx = grid.shape
        grid_rows.append(
            {
                "patient": patient,
                "nx": nx, "ny": ny, "nz": nz,
                "n_voxels": nx * ny * nz,
                "spacing_x": grid.spacing[0],
                "spacing_y": grid.spacing[1],
                "spacing_z": grid.spacing[2],
                "origin_x": grid.origin[0],
                "origin_y": grid.origin[1],
                "origin_z": grid.origin[2],
                "voxel_volume_cc": grid.voxel_volume_cc,
            }
        )
        built = sum(1 for r in rows if r["present"])
        print(
            f"  {patient}: grid {nx}x{ny}x{nz} = {nx * ny * nz / 1e6:.1f}M voxels "
            f"@ {grid.spacing[2]:.1f} mm slices, {built} masks, "
            f"{time.perf_counter() - t0:.1f}s"
        )

    structures = pd.DataFrame(struct_rows)
    grids = pd.DataFrame(grid_rows)
    write_table(structures, cfg.results_root / "structures.parquet", run)
    write_table(grids, cfg.results_root / "grids.parquet", run)

    present = structures[structures["present"]]
    print(f"\n{len(present)} masks built, {len(structures) - len(present)} absent")
    print(
        f"volume agreement (mask vs polygon): median "
        f"{present['volume_agreement'].median() * 100:.2f}%, "
        f"worst {present['volume_agreement'].max() * 100:.2f}%"
    )
    dirty = present[~present["raster_clean"].astype(bool)]
    print(f"{len(dirty)} mask(s) with a rasterisation flag — see stage 03")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
