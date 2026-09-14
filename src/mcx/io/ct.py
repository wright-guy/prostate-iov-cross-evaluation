"""Planning CT series geometry.

Only the geometry is needed for Phase A — slice positions, spacing, orientation and
frame of reference. Pixel data is loaded later, and only for QC montages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

from mcx.config import Config


@dataclass(frozen=True)
class CTSeries:
    patient: str
    paths: tuple[Path, ...]
    z: np.ndarray  # ascending slice positions
    origin_xy: np.ndarray  # (2,) x, y of ImagePositionPatient
    pixel_spacing: np.ndarray  # (2,) x, y
    slice_thickness: float
    rows: int
    columns: int
    series_instance_uid: str
    frame_of_reference_uid: str
    patient_position: str
    manufacturer: str

    @property
    def n_slices(self) -> int:
        return len(self.z)

    @property
    def slice_spacing(self) -> float:
        """Median spacing between consecutive slices."""
        return float(np.median(np.diff(self.z))) if self.n_slices > 1 else self.slice_thickness

    def spacing_uniformity_mm(self) -> float:
        """Peak-to-peak variation in slice spacing. Zero for a clean series."""
        if self.n_slices < 3:
            return 0.0
        d = np.diff(self.z)
        return float(d.max() - d.min())

    def duplicate_z_count(self) -> int:
        return int(self.n_slices - len(np.unique(np.round(self.z, 3))))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array(
            [self.origin_xy[0], self.origin_xy[1], float(self.z.min())]
        )
        hi = np.array(
            [
                self.origin_xy[0] + (self.columns - 1) * self.pixel_spacing[0],
                self.origin_xy[1] + (self.rows - 1) * self.pixel_spacing[1],
                float(self.z.max()),
            ]
        )
        return lo, hi


def load_ct_series(cfg: Config, patient: str) -> CTSeries:
    folder = next(iter(sorted(cfg.dicom_root.glob(f"{patient}_*"))), None)
    if folder is None:
        raise FileNotFoundError(f"no CT folder for {patient}")

    headers = []
    for path in sorted(folder.glob("CT.*.dcm")):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        headers.append((float(ds.ImagePositionPatient[2]), path, ds))
    if not headers:
        raise FileNotFoundError(f"no CT slices for {patient}")
    headers.sort(key=lambda t: t[0])

    ref = headers[0][2]
    series_uids = {str(ds.SeriesInstanceUID) for _, _, ds in headers}
    if len(series_uids) != 1:
        raise ValueError(f"{patient}: {len(series_uids)} CT series present, expected 1")

    return CTSeries(
        patient=patient,
        paths=tuple(p for _, p, _ in headers),
        z=np.array([z for z, _, _ in headers]),
        origin_xy=np.asarray([float(v) for v in ref.ImagePositionPatient[:2]]),
        pixel_spacing=np.asarray([float(v) for v in ref.PixelSpacing][::-1]),
        slice_thickness=float(ref.SliceThickness),
        rows=int(ref.Rows),
        columns=int(ref.Columns),
        series_instance_uid=series_uids.pop(),
        frame_of_reference_uid=str(ref.FrameOfReferenceUID),
        patient_position=str(getattr(ref, "PatientPosition", "")),
        manufacturer=str(getattr(ref, "Manufacturer", "")),
    )
