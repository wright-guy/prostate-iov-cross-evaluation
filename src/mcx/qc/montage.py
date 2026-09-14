"""Visual QC montages.

Numbers catch what you thought to check; a picture catches what you did not. One page
per patient and structure, with every contour set overlaid on the planning CT at the
axial slice through the structure's centroid, plus mid-sagittal and mid-coronal views.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pydicom  # noqa: E402

from mcx.config import Config  # noqa: E402
from mcx.geom.grid import Grid  # noqa: E402
from mcx.geom.masks import MaskStore  # noqa: E402
from mcx.io.ct import CTSeries  # noqa: E402

WINDOW = (-160.0, 240.0)  # soft-tissue/pelvis window in HU


def ct_volume_on_grid(ct: CTSeries, grid: Grid) -> np.ndarray:
    """Resample the planning CT onto the analysis grid by nearest neighbour.

    Nearest neighbour is deliberate: this is a QC backdrop, and interpolating would
    make a misregistration look smoother than it is.
    """
    out = np.zeros(grid.shape, dtype=np.float32)
    # Grid z planes coincide with CT slice positions by construction.
    ct_index = {round(float(z), 3): i for i, z in enumerate(ct.z)}
    col = np.clip(
        np.round((grid.origin[0] + np.arange(grid.shape[2]) * grid.spacing[0]
                  - ct.origin_xy[0]) / ct.pixel_spacing[0]).astype(int),
        0, ct.columns - 1,
    )
    row = np.clip(
        np.round((grid.origin[1] + np.arange(grid.shape[1]) * grid.spacing[1]
                  - ct.origin_xy[1]) / ct.pixel_spacing[1]).astype(int),
        0, ct.rows - 1,
    )
    for k, z in enumerate(grid.z):
        i = ct_index.get(round(float(z), 3))
        if i is None:
            continue
        ds = pydicom.dcmread(ct.paths[i])
        hu = (ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0))
              + float(getattr(ds, "RescaleIntercept", 0.0)))
        out[k] = hu[np.ix_(row, col)]
    return out


def _outline(ax, mask2d: np.ndarray, colour: str, label: str | None, lw: float) -> None:
    if mask2d.any():
        ax.contour(mask2d.astype(float), levels=[0.5], colors=[colour], linewidths=lw)
    if label:
        ax.plot([], [], color=colour, lw=lw, label=label)


def structure_montage(
    cfg: Config,
    patient: str,
    structure: str,
    grid: Grid,
    store: MaskStore,
    ct_volume: np.ndarray,
    out_path: Path,
) -> Path | None:
    """One page: all contour sets for one structure, three orthogonal views."""
    sets = [
        s for s in cfg.all_sets
        if cfg.is_known_absent(patient, s) is None
        and structure in cfg.structures_for_set(s)
        and store.has(patient, s, structure)
    ]
    if not sets:
        return None

    masks = {s: store.get(patient, s, structure) for s in sets}
    union = np.zeros(grid.shape, dtype=bool)
    for m in masks.values():
        union |= m
    if not union.any():
        return None

    idx = np.argwhere(union)
    kz, ky, kx = (int(round(v)) for v in idx.mean(axis=0))

    observers = cfg.observers
    palette = plt.get_cmap("tab10")
    colours = {s: palette(i % 10) for i, s in enumerate(observers)}
    colours["VOTE"] = "#000000"
    colours["STAPLE"] = "#888888"

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    views = [
        ("axial", ct_volume[kz], lambda m: m[kz], f"z index {kz}"),
        ("coronal", ct_volume[:, ky, :], lambda m: m[:, ky, :], f"y index {ky}"),
        ("sagittal", ct_volume[:, :, kx], lambda m: m[:, :, kx], f"x index {kx}"),
    ]
    for ax, (name, backdrop, slicer, subtitle) in zip(axes, views, strict=True):
        aspect = grid.spacing[2] / grid.spacing[0] if name != "axial" else 1.0
        ax.imshow(backdrop, cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1],
                  origin="lower", aspect=aspect)
        for s in sets:
            lw = 2.2 if s in ("VOTE", "STAPLE") else 1.1
            _outline(ax, slicer(masks[s]), colours[s], s if name == "axial" else None, lw)
        ax.set_title(f"{name} — {subtitle}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.85)
    fig.suptitle(
        f"{patient} · {structure} · {len(sets)} contour sets · "
        f"{grid.spacing[0]:.1f} mm in plane, {grid.spacing[2]:.1f} mm slices",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
