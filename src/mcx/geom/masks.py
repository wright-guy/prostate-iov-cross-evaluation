"""Binary mask cache.

Masks are bit-packed before compression: a 1 mm grid over a pelvis is a few million
voxels, and 480 of them at one byte each would be gigabytes. Packed and deflated they
are a few tens of kilobytes.

The cache key is the geometry hash -- the subset of config that can actually move a
voxel -- so changing the grid or an ROI mapping invalidates every mask rather than
silently mixing geometries, while editing a comment or a waiver does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mcx.config import Config
from mcx.geom.grid import Grid
from mcx.geom.rasterise import RasterReport, rasterise
from mcx.io.rtstruct import StructureSet


def mask_dir(cfg: Config, patient: str) -> Path:
    return cfg.derived_root / "masks" / cfg.geometry_hash / patient


def mask_path(cfg: Config, patient: str, contour_set: str, structure: str) -> Path:
    return mask_dir(cfg, patient) / f"{contour_set}__{structure}.npz"


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        packed=np.packbits(mask.reshape(-1)),
        shape=np.asarray(mask.shape, dtype=np.int64),
    )


def load_mask(path: Path) -> np.ndarray:
    with np.load(path) as data:
        shape = tuple(int(v) for v in data["shape"])
        n = int(np.prod(shape))
        return np.unpackbits(data["packed"], count=n).astype(bool).reshape(shape)


class MaskStore:
    """Read access to the cache, with a small in-memory LRU per patient."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._cache: dict[tuple[str, str, str], np.ndarray] = {}

    def has(self, patient: str, contour_set: str, structure: str) -> bool:
        return mask_path(self.cfg, patient, contour_set, structure).exists()

    def get(self, patient: str, contour_set: str, structure: str) -> np.ndarray:
        key = (patient, contour_set, structure)
        if key not in self._cache:
            path = mask_path(self.cfg, patient, contour_set, structure)
            if not path.exists():
                raise FileNotFoundError(f"no cached mask for {patient}/{contour_set}/{structure}")
            self._cache[key] = load_mask(path)
        return self._cache[key]

    def drop_patient(self, patient: str) -> None:
        for key in [k for k in self._cache if k[0] == patient]:
            del self._cache[key]


def build_masks_for_patient(
    cfg: Config,
    patient: str,
    ss: StructureSet,
    grid: Grid,
    *,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    """Rasterise every available ROI for one patient and cache it.

    Returns one summary row per (contour set, structure), whether or not it existed.
    """
    rows: list[dict[str, object]] = []
    for contour_set in cfg.all_sets:
        if cfg.is_known_absent(patient, contour_set) is not None:
            continue
        for structure in cfg.structures_for_set(contour_set):
            roi_name = cfg.roi_name(patient, contour_set, structure)
            roi = ss.get(contour_set, structure)
            row: dict[str, object] = {
                "patient": patient,
                "contour_set": contour_set,
                "structure": structure,
                "roi_name": roi_name,
                "present": roi is not None and not roi.is_empty,
            }
            if roi is None or roi.is_empty:
                rows.append(row)
                continue

            path = mask_path(cfg, patient, contour_set, structure)
            if path.exists() and not overwrite:
                mask = load_mask(path)
                report = _report_only(roi, grid)
            else:
                mask, report = rasterise(roi, grid)
                save_mask(path, mask)

            row.update(_describe(mask, grid, report))
            rows.append(row)
    return rows


def _report_only(roi, grid: Grid) -> RasterReport:  # noqa: ANN001
    _, report = rasterise(roi, grid)
    return report


def _describe(mask: np.ndarray, grid: Grid, report: RasterReport) -> dict[str, object]:
    idx = np.argwhere(mask)
    if idx.size == 0:
        centroid = (np.nan, np.nan, np.nan)
        extent = (np.nan, np.nan, np.nan)
    else:
        # argwhere gives (z, y, x); convert to patient coordinates.
        world = grid.origin + idx[:, ::-1] * grid.spacing
        centroid = tuple(world.mean(axis=0))
        extent = tuple(world.max(axis=0) - world.min(axis=0) + grid.spacing)

    return {
        "volume_cc": float(mask.sum()) * grid.voxel_volume_cc,
        "polygon_volume_cc": report.polygon_volume_cc,
        "volume_agreement": report.volume_agreement,
        "n_voxels": int(mask.sum()),
        "n_slices_used": int(mask.any(axis=(1, 2)).sum()),
        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
        "centroid_z": centroid[2],
        "extent_x": extent[0],
        "extent_y": extent[1],
        "extent_z": extent[2],
        "n_loops": report.n_loops,
        "n_loops_off_grid_z": report.n_loops_off_grid_z,
        "n_loops_degenerate": report.n_loops_degenerate,
        "n_loops_zero_area": report.n_loops_zero_area,
        "area_dropped_mm2": report.area_dropped_mm2,
        "n_loops_clipped_xy": report.n_loops_clipped_xy,
        "n_slices_multiloop": report.n_slices_multiloop,
        "max_loops_per_slice": report.max_loops_per_slice,
        "n_slices_z_collapsed": report.n_slices_z_collapsed,
        "z_gap_slices": report.z_gap_slices,
        "raster_clean": report.clean,
    }
