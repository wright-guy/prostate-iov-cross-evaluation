"""RTSTRUCT reading.

The dataset gives one merged structure set per patient holding every observer's
contours, so this module's job is to resolve a (contour_set, structure) pair to the
right ROI and hand back its planar polygons in patient coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pydicom

from mcx.config import Config


@dataclass(frozen=True)
class ContourSlice:
    """One closed planar loop, in patient coordinates (mm)."""

    z: float
    points: np.ndarray  # (n, 3)
    geometric_type: str

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    def area_mm2(self) -> float:
        """Signed shoelace area in the axial plane. Sign encodes winding direction."""
        x, y = self.points[:, 0], self.points[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@dataclass(frozen=True)
class ROI:
    name: str
    number: int
    slices: tuple[ContourSlice, ...]

    @property
    def is_empty(self) -> bool:
        return len(self.slices) == 0

    @property
    def z_positions(self) -> list[float]:
        return sorted({s.z for s in self.slices})

    def polygon_volume_cc(self) -> float:
        """Volume from |shoelace| area x slice spacing.

        Independent of the rasteriser, which is the point: stage 03 compares this
        against the mask volume as a round-trip accuracy check. Nested loops (rectal
        gas) are handled by the signed sum within a slice.
        """
        zs = self.z_positions
        if not zs:
            return 0.0
        spacing = float(np.median(np.diff(zs))) if len(zs) > 1 else 1.0
        total = 0.0
        for z in zs:
            loops = [s for s in self.slices if s.z == z]
            areas = [s.area_mm2() for s in loops]
            if len(areas) == 1:
                total += abs(areas[0])
            else:
                # Outer loop is the largest; inner loops of opposite winding subtract.
                order = np.argsort([-abs(a) for a in areas])
                ref_sign = np.sign(areas[order[0]])
                total += sum(
                    abs(a) if np.sign(a) == ref_sign else -abs(a)
                    for a in (areas[i] for i in order)
                )
        return total * spacing / 1000.0


class StructureSet:
    """A merged RTSTRUCT, addressed by (contour_set, structure)."""

    def __init__(self, path: Path, cfg: Config, patient: str) -> None:
        self.path = path
        self.cfg = cfg
        self.patient = patient
        self._ds = pydicom.dcmread(path)
        self._by_number = {
            int(r.ROINumber): str(r.ROIName) for r in self._ds.StructureSetROISequence
        }
        self._by_lower = {v.lower(): k for k, v in self._by_number.items()}

    # ------------------------------------------------------------------ meta

    @property
    def sop_instance_uid(self) -> str:
        return str(self._ds.SOPInstanceUID)

    @property
    def label(self) -> str:
        return str(getattr(self._ds, "StructureSetLabel", ""))

    @property
    def frame_of_reference_uid(self) -> str:
        return str(self._ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID)

    @property
    def roi_names(self) -> list[str]:
        return [self._by_number[n] for n in sorted(self._by_number)]

    def has_roi(self, name: str) -> bool:
        return name.lower() in self._by_lower

    def matching_rois(self, prefix: str, contour_set: str) -> list[str]:
        """ROI names that look like ``<prefix>*_<set>`` — used to spot duplicates
        such as K042's Rectum2_O7."""
        pl, sl = prefix.lower(), contour_set.lower()
        return [
            n
            for n in self.roi_names
            if n.lower().startswith(pl) and n.lower().endswith("_" + sl)
        ]

    # ------------------------------------------------------------------ read

    @lru_cache(maxsize=None)  # noqa: B019 - cache lives with the instance
    def roi(self, name: str) -> ROI | None:
        key = name.lower()
        if key not in self._by_lower:
            return None
        number = self._by_lower[key]
        seq = None
        for c in self._ds.ROIContourSequence:
            if int(c.ReferencedROINumber) == number:
                seq = getattr(c, "ContourSequence", None)
                break
        slices: list[ContourSlice] = []
        for item in seq or []:
            pts = np.asarray(item.ContourData, dtype=float).reshape(-1, 3)
            if pts.shape[0] == 0:
                continue
            slices.append(
                ContourSlice(
                    z=round(float(pts[0, 2]), 3),
                    points=pts,
                    geometric_type=str(getattr(item, "ContourGeometricType", "")),
                )
            )
        return ROI(name=self._by_number[number], number=number, slices=tuple(slices))

    def get(self, contour_set: str, structure: str) -> ROI | None:
        """Resolve via config so overrides and exceptions are honoured."""
        return self.roi(self.cfg.roi_name(self.patient, contour_set, structure))


def find_structure_set(cfg: Config, patient: str) -> Path:
    """The single RTSTRUCT for a patient. Raises if it is not unique."""
    candidates = sorted(cfg.dicom_root.glob(f"{patient}_*/RTSTRUCT*.dcm"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one RTSTRUCT for {patient}, found {len(candidates)}"
        )
    return candidates[0]


def load_structure_set(cfg: Config, patient: str) -> StructureSet:
    return StructureSet(find_structure_set(cfg, patient), cfg, patient)
