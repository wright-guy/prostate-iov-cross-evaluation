"""Stage 00 — index every DICOM file under ``data/`` with checksums and key headers.

Nothing downstream reads the filesystem directly; everything works from this index.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom

from mcx.config import REPO_ROOT, Config

SOP_CLASS = {
    "1.2.840.10008.5.1.4.1.1.2": "CT",
    "1.2.840.10008.5.1.4.1.1.481.2": "RTDOSE",
    "1.2.840.10008.5.1.4.1.1.481.3": "RTSTRUCT",
    "1.2.840.10008.5.1.4.1.1.481.5": "RTPLAN",
}

CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class FileRef:
    path: Path
    modality: str
    patient: str
    contour_set: str | None


def _iter_files(root: Path) -> Iterator[Path]:
    yield from sorted(p for p in root.rglob("*.dcm") if p.is_file())


def _first_of(ds: pydicom.Dataset, *keywords: str) -> Any:
    for kw in keywords:
        if kw in ds:
            return ds[kw].value
    return None


def _seq_uid(ds: pydicom.Dataset, seq_kw: str) -> str | None:
    seq = ds.get(seq_kw)
    if not seq:
        return None
    return str(seq[0].get("ReferencedSOPInstanceUID") or "") or None


def index_dicom(cfg: Config, *, checksums: bool = True) -> pd.DataFrame:
    """Walk every configured root and read one header row per file."""
    rows: list[dict[str, Any]] = []

    sources = [
        ("dicom", cfg.dicom_root, _patient_from_dicom_root),
        ("dose", cfg.dose_root, _patient_set_from_dose_root),
    ]
    if cfg.rtplan_root.exists():
        sources.append(("rtplan", cfg.rtplan_root, _patient_set_from_dose_root))

    for source, root, resolver in sources:
        if not root.exists():
            continue
        for path in _iter_files(root):
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            sop_class = str(getattr(ds, "SOPClassUID", "")) or ""
            patient, contour_set = resolver(cfg, path, root)
            rows.append(
                {
                    "source": source,
                    "relpath": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "patient": patient,
                    "contour_set": contour_set,
                    "modality": SOP_CLASS.get(sop_class, str(getattr(ds, "Modality", "?"))),
                    "sop_class_uid": sop_class,
                    "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
                    "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
                    "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
                    "frame_of_reference_uid": _frame_of_reference(ds),
                    "dicom_patient_name": str(getattr(ds, "PatientName", "")),
                    "dicom_patient_id": str(getattr(ds, "PatientID", "")),
                    "instance_creation_date": str(getattr(ds, "InstanceCreationDate", "")),
                    "instance_creation_time": str(getattr(ds, "InstanceCreationTime", "")),
                    "manufacturer": str(getattr(ds, "Manufacturer", "")),
                    "model": str(getattr(ds, "ManufacturerModelName", "")),
                    "software_versions": str(getattr(ds, "SoftwareVersions", "")),
                    "referenced_rtplan_uid": _seq_uid(ds, "ReferencedRTPlanSequence"),
                    "referenced_structure_set_uid": _seq_uid(
                        ds, "ReferencedStructureSetSequence"
                    ),
                    "structure_set_label": str(getattr(ds, "StructureSetLabel", "")) or None,
                    "rt_plan_label": str(getattr(ds, "RTPlanLabel", "")) or None,
                    "approval_status": str(getattr(ds, "ApprovalStatus", "")) or None,
                    "dose_summation_type": str(getattr(ds, "DoseSummationType", "")) or None,
                    "dose_units": str(getattr(ds, "DoseUnits", "")) or None,
                    "dose_type": str(getattr(ds, "DoseType", "")) or None,
                    "n_fractions_planned": _n_fractions(ds),
                    "slice_z": _slice_z(ds),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path) if checksums else None,
                }
            )

    return pd.DataFrame(rows)


def _frame_of_reference(ds: pydicom.Dataset) -> str | None:
    if "FrameOfReferenceUID" in ds:
        return str(ds.FrameOfReferenceUID)
    seq = ds.get("ReferencedFrameOfReferenceSequence")
    if seq:
        return str(seq[0].FrameOfReferenceUID)
    return None


def _n_fractions(ds: pydicom.Dataset) -> int | None:
    seq = ds.get("FractionGroupSequence")
    if not seq:
        return None
    v = seq[0].get("NumberOfFractionsPlanned")
    return int(v) if v is not None else None


def _slice_z(ds: pydicom.Dataset) -> float | None:
    ipp = _first_of(ds, "ImagePositionPatient")
    if ipp is None or len(ipp) < 3:
        return None
    return float(ipp[2])


def _patient_from_dicom_root(cfg: Config, path: Path, root: Path) -> tuple[str | None, None]:
    """``<root>/<PATIENT>_PLAN_CT/<file>`` — CTs and the merged structure set."""
    rel = path.relative_to(root).parts
    if not rel:
        return None, None
    patient = rel[0].split("_")[0]
    return (patient if patient in cfg.patients else None), None


def _patient_set_from_dose_root(
    cfg: Config, path: Path, root: Path
) -> tuple[str | None, str | None]:
    """``<root>/<PATIENT>/<SET>/<file>`` — one dose (or plan) per contour set."""
    rel = path.relative_to(root).parts
    if len(rel) < 2:
        return None, None
    patient = rel[0]
    contour_set = cfg.canonical_set(rel[1])
    return (patient if patient in cfg.patients else None), contour_set
