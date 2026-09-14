"""Resampled-dose cache.

Dose is stored as float32 with its coverage mask alongside, keyed by the config hash
so a change to the grid definition invalidates every cached dose rather than silently
mixing geometries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mcx.config import Config
from mcx.dose.resample import resample_to_grid
from mcx.geom.grid import Grid
from mcx.io.rtdose import load_dose


def dose_dir(cfg: Config, patient: str) -> Path:
    return cfg.derived_root / "dose" / cfg.geometry_hash / patient


def dose_path(cfg: Config, patient: str, contour_set: str) -> Path:
    return dose_dir(cfg, patient) / f"{contour_set}.npz"


def save_resampled(path: Path, dose: np.ndarray, covered: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dose=dose.astype(np.float32),
        covered=np.packbits(covered.reshape(-1)),
        shape=np.asarray(covered.shape, dtype=np.int64),
    )


def load_resampled(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        shape = tuple(int(v) for v in data["shape"])
        covered = (
            np.unpackbits(data["covered"], count=int(np.prod(shape)))
            .astype(bool)
            .reshape(shape)
        )
        return data["dose"], covered


class DoseStore:
    """Read access to the resampled-dose cache, one patient in memory at a time."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def has(self, patient: str, contour_set: str) -> bool:
        return dose_path(self.cfg, patient, contour_set).exists()

    def get(self, patient: str, contour_set: str) -> tuple[np.ndarray, np.ndarray]:
        key = (patient, contour_set)
        if key not in self._cache:
            path = dose_path(self.cfg, patient, contour_set)
            if not path.exists():
                raise FileNotFoundError(f"no resampled dose for {patient}/{contour_set}")
            self._cache[key] = load_resampled(path)
        return self._cache[key]

    def drop_patient(self, patient: str) -> None:
        for key in [k for k in self._cache if k[0] == patient]:
            del self._cache[key]


def build_dose_for_patient(
    cfg: Config, patient: str, grid: Grid, *, overwrite: bool = False
) -> list[dict[str, object]]:
    """Resample every dose for one patient onto the analysis grid and cache it."""
    rows: list[dict[str, object]] = []
    for contour_set in cfg.all_sets:
        if cfg.is_known_absent(patient, contour_set) is not None:
            continue
        path = dose_path(cfg, patient, contour_set)
        src = load_dose(cfg, patient, contour_set)

        if path.exists() and not overwrite:
            arr, covered = load_resampled(path)
        else:
            arr, covered = resample_to_grid(src, grid)
            save_resampled(path, arr, covered)

        finite = np.isfinite(arr)
        rows.append(
            {
                "patient": patient,
                "contour_set": contour_set,
                "source_file": src.path.name,
                "source_shape": str(src.array.shape),
                "source_max_gy": src.max_gy,
                "resampled_max_gy": float(np.nanmax(arr)) if finite.any() else float("nan"),
                "grid_coverage_fraction": float(covered.mean()),
                "n_voxels_covered": int(covered.sum()),
                "summation_type": src.summation_type,
                "dose_type": src.dose_type,
                "units": src.units,
                "referenced_rtplan_uid": src.referenced_rtplan_uid,
            }
        )
    return rows
