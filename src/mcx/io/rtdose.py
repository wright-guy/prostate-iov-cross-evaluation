"""RTDOSE reading.

Dose grids differ in origin and extent for every plan, so the grid geometry is kept
explicit rather than being implied by array indices. Stage 05 resamples onto the
common analysis grid; this module only reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

from mcx.config import Config


@dataclass(frozen=True)
class DoseGrid:
    """A dose distribution with its geometry, in patient coordinates (mm, Gy)."""

    array: np.ndarray  # (nz, ny, nx), Gy
    origin: np.ndarray  # (3,) ImagePositionPatient of the first frame
    spacing: np.ndarray  # (3,) dx, dy, dz
    z: np.ndarray  # (nz,) absolute slice positions
    path: Path
    sop_instance_uid: str
    referenced_rtplan_uid: str | None
    summation_type: str
    dose_type: str
    units: str
    frame_of_reference_uid: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.array.shape  # type: ignore[return-value]

    @property
    def voxel_volume_cc(self) -> float:
        return float(np.prod(self.spacing)) / 1000.0

    @property
    def max_gy(self) -> float:
        return float(self.array.max())

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box (lo, hi) in patient coordinates, voxel centres."""
        nz, ny, nx = self.array.shape
        lo = np.array([self.origin[0], self.origin[1], self.z.min()])
        hi = np.array(
            [
                self.origin[0] + (nx - 1) * self.spacing[0],
                self.origin[1] + (ny - 1) * self.spacing[1],
                self.z.max(),
            ]
        )
        return lo, hi

    def contains(self, points: np.ndarray, *, margin_mm: float = 0.0) -> np.ndarray:
        """Boolean mask of which points lie inside the grid bounds."""
        lo, hi = self.bounds()
        return np.all((points >= lo - margin_mm) & (points <= hi + margin_mm), axis=1)


def _uniform(values: np.ndarray, *, tol: float = 1e-3) -> float:
    d = np.diff(values)
    if d.size == 0:
        raise ValueError("dose grid has a single frame")
    if float(d.max() - d.min()) > tol:
        raise ValueError(f"non-uniform dose grid spacing: {d.min()}..{d.max()}")
    return float(d.mean())


def read_dose(path: Path) -> DoseGrid:
    ds = pydicom.dcmread(path)
    array = ds.pixel_array.astype(np.float32) * float(ds.DoseGridScaling)
    origin = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype=float)

    orientation = [float(v) for v in ds.ImageOrientationPatient]
    if orientation != [1, 0, 0, 0, 1, 0]:
        raise ValueError(f"{path.name}: unsupported ImageOrientationPatient {orientation}")

    offsets = np.asarray([float(v) for v in ds.GridFrameOffsetVector], dtype=float)
    dz = _uniform(offsets)
    py, px = (float(v) for v in ds.PixelSpacing)  # rows (y), columns (x)

    return DoseGrid(
        array=array,
        origin=origin,
        spacing=np.array([px, py, dz]),
        z=origin[2] + offsets,
        path=path,
        sop_instance_uid=str(ds.SOPInstanceUID),
        referenced_rtplan_uid=_referenced_plan(ds),
        summation_type=str(getattr(ds, "DoseSummationType", "")),
        dose_type=str(getattr(ds, "DoseType", "")),
        units=str(getattr(ds, "DoseUnits", "")),
        frame_of_reference_uid=str(ds.FrameOfReferenceUID),
    )


def _referenced_plan(ds: pydicom.Dataset) -> str | None:
    seq = ds.get("ReferencedRTPlanSequence")
    if not seq:
        return None
    return str(seq[0].ReferencedSOPInstanceUID)


def find_dose(cfg: Config, patient: str, contour_set: str) -> Path | None:
    """Locate the dose file for a (patient, contour set), honouring the folder alias."""
    for folder in cfg.dose_folders(contour_set):
        d = cfg.dose_root / patient / folder
        if d.is_dir():
            files = sorted(d.glob("*.dcm"))
            if len(files) == 1:
                return files[0]
            if len(files) > 1:
                raise ValueError(f"{d}: expected one RTDOSE, found {len(files)}")
    return None


def load_dose(cfg: Config, patient: str, contour_set: str) -> DoseGrid:
    path = find_dose(cfg, patient, contour_set)
    if path is None:
        raise FileNotFoundError(f"no RTDOSE for {patient}/{contour_set}")
    return read_dose(path)
