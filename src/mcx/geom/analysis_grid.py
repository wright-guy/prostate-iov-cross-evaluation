"""The common per-patient analysis grid.

Built once from the union of the extent-defining structures across every contour set,
so that all twelve sets and all twelve doses live on identical geometry. Its z planes
coincide exactly with the planning CT slice positions; see config/grid.yaml for why.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mcx.config import Config
from mcx.geom.grid import Grid
from mcx.io.ct import load_ct_series
from mcx.io.rtstruct import StructureSet


def contour_bounds(
    ss: StructureSet, cfg: Config, patient: str, structures: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box of every named structure across every contour set."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for cset in cfg.all_sets:
        if cfg.is_known_absent(patient, cset) is not None:
            continue
        for structure in structures:
            if structure not in cfg.structures_for_set(cset):
                continue
            roi = ss.get(cset, structure)
            if roi is None or roi.is_empty:
                continue
            for loop in roi.slices:
                lo = np.minimum(lo, loop.points.min(axis=0))
                hi = np.maximum(hi, loop.points.max(axis=0))
    if not np.isfinite(lo).all():
        raise ValueError(f"{patient}: no contours found for {structures}")
    return lo, hi


def build_analysis_grid(cfg: Config, patient: str, ss: StructureSet) -> Grid:
    """Construct the analysis grid for one patient."""
    spec = cfg.grid_raw["analysis_grid"]
    lo, hi = contour_bounds(ss, cfg, patient, list(spec["extent_structures"]))
    margin = float(spec["margin_mm"])
    in_plane = float(spec["in_plane_mm"])

    ct = load_ct_series(cfg, patient)
    if spec["slice_spacing"] != "native_ct":
        raise ValueError(f"unsupported slice_spacing {spec['slice_spacing']!r}")
    dz = ct.slice_spacing

    # z planes must coincide exactly with CT slice positions, so snap the padded
    # bounds outward onto the CT slice grid rather than starting from an arbitrary
    # origin. Anything else would put contours between planes.
    z_lo, z_hi = lo[2] - margin, hi[2] + margin
    keep = ct.z[(ct.z >= z_lo) & (ct.z <= z_hi)]
    if keep.size == 0:
        raise ValueError(f"{patient}: analysis extent falls outside the CT series")

    origin = np.array([lo[0] - margin, lo[1] - margin, float(keep[0])])
    nx = int(np.ceil(((hi[0] + margin) - origin[0]) / in_plane)) + 1
    ny = int(np.ceil(((hi[1] + margin) - origin[1]) / in_plane)) + 1
    return Grid(
        origin=origin,
        spacing=np.array([in_plane, in_plane, dz]),
        shape=(int(keep.size), ny, nx),
    )


def grid_path(cfg: Config, patient: str) -> Path:
    return cfg.derived_root / "grids" / f"{patient}.json"


def save_grid(cfg: Config, patient: str, grid: Grid) -> Path:
    path = grid_path(cfg, patient)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "patient": patient,
                "origin": grid.origin.tolist(),
                "spacing": grid.spacing.tolist(),
                "shape": list(grid.shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_grid(cfg: Config, patient: str) -> Grid:
    d = json.loads(grid_path(cfg, patient).read_text(encoding="utf-8"))
    return Grid(
        origin=np.asarray(d["origin"]),
        spacing=np.asarray(d["spacing"]),
        shape=tuple(d["shape"]),  # type: ignore[arg-type]
    )


def get_or_build_grid(cfg: Config, patient: str, ss: StructureSet) -> Grid:
    if grid_path(cfg, patient).exists():
        return load_grid(cfg, patient)
    grid = build_analysis_grid(cfg, patient, ss)
    save_grid(cfg, patient, grid)
    return grid
