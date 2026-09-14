"""Stage 05 — resample all 59 doses onto the per-patient analysis grid.

    python pipelines/05_resample_dose.py [--overwrite] [--patient K018]

Writes results/dose_resampled.parquet and caches each resampled distribution under the
derived root. After this stage every contour set and every dose for a patient share one
geometry, which is what makes the 530-cell cross-evaluation array arithmetic.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import load_config  # noqa: E402
from mcx.dose.store import build_dose_for_patient, dose_dir  # noqa: E402
from mcx.geom.analysis_grid import load_grid  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--patient", action="append")
    args = ap.parse_args()

    cfg = load_config()
    run = RunInfo.create("05_resample_dose", cfg)
    patients = args.patient or cfg.patients

    print(f"dose cache: {dose_dir(cfg, '<patient>')}\n")

    rows = []
    for patient in patients:
        t0 = time.perf_counter()
        grid = load_grid(cfg, patient)
        got = build_dose_for_patient(cfg, patient, grid, overwrite=args.overwrite)
        rows.extend(got)
        cov = min(float(r["grid_coverage_fraction"]) for r in got)
        print(
            f"  {patient}: {len(got)} doses onto {grid.shape[2]}x{grid.shape[1]}x"
            f"{grid.shape[0]}, worst grid coverage {cov * 100:.1f}%, "
            f"{time.perf_counter() - t0:.1f}s"
        )

    df = pd.DataFrame(rows)
    write_table(df, cfg.results_root / "dose_resampled.parquet", run)

    print(f"\n{len(df)} doses resampled")
    print(
        f"max dose after resampling: {df['resampled_max_gy'].min():.2f}-"
        f"{df['resampled_max_gy'].max():.2f} Gy "
        f"(source {df['source_max_gy'].min():.2f}-{df['source_max_gy'].max():.2f} Gy)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
