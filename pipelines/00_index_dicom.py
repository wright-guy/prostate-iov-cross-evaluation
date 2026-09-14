"""Stage 00 — index every DICOM file with checksums and key headers.

    python pipelines/00_index_dicom.py [--no-checksums]

Writes results/inventory.parquet (+ .csv). Reads data/, writes nothing to it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import load_config  # noqa: E402
from mcx.io.dicom_index import index_dicom  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-checksums",
        action="store_true",
        help="skip SHA-256 (fast re-index; the integrity guarantee is lost)",
    )
    args = ap.parse_args()

    cfg = load_config()
    run = RunInfo.create("00_index_dicom", cfg)

    print(f"indexing {cfg.dicom_root} and {cfg.dose_root} ...")
    df = index_dicom(cfg, checksums=not args.no_checksums)

    out = write_table(df, cfg.results_root / "inventory.parquet", run)

    print(f"\n{len(df)} files indexed -> {out}")
    print(df.groupby(["source", "modality"], dropna=False).size().to_string())
    unresolved = df[df["patient"].isna()]
    if len(unresolved):
        print(f"\n{len(unresolved)} file(s) could not be assigned to a patient:")
        print(unresolved["relpath"].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
