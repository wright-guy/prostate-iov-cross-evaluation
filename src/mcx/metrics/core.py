"""Contour-comparison metrics.

Two things govern the implementation.

**Cropping.** Every distance-based metric needs a Euclidean distance transform, which
costs the whole grid. A prostate occupies a few per cent of a 2-million-voxel pelvis
grid, so the transforms are computed inside the joint bounding box of the two masks
plus a margin. The margin has to exceed the largest distance that will be read back
out, or a surface voxel near the crop edge would be measured against a boundary that is
not really there.

**Symmetry.** Unsigned surface metrics are symmetric by construction. The one-sided
form is not a metric and gives a different answer depending on which contour is called
the reference, which is exactly the ambiguity this study is trying to remove.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from mcx.geom.grid import Grid

# Distances beyond this are not read back out of a cropped transform, so the crop
# margin must be at least this large.
CROP_MARGIN_MM = 40.0


@dataclass(frozen=True)
class Crop:
    """A sub-volume of the analysis grid, with the spacing needed for distances."""

    slices: tuple[slice, slice, slice]
    sampling: tuple[float, float, float]   # (z, y, x) mm, for scipy

    def apply(self, array: np.ndarray) -> np.ndarray:
        return array[self.slices]


def joint_crop(a: np.ndarray, b: np.ndarray, grid: Grid,
               *, margin_mm: float = CROP_MARGIN_MM) -> Crop | None:
    """Bounding box containing both masks plus a margin, or None if both are empty."""
    union = a | b
    if not union.any():
        return None
    idx = np.argwhere(union)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    spacing_zyx = grid.spacing[::-1]
    pad = np.ceil(margin_mm / spacing_zyx).astype(int)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, np.array(a.shape))
    return Crop(
        slices=tuple(slice(int(l), int(h)) for l, h in zip(lo, hi, strict=True)),
        sampling=tuple(float(v) for v in spacing_zyx),
    )


def _edt(mask: np.ndarray, sampling: tuple[float, float, float]) -> np.ndarray:
    """Signed distance in mm: negative inside, positive outside."""
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, -np.inf, dtype=np.float32)
    outside = ndimage.distance_transform_edt(~mask, sampling=sampling)
    inside = ndimage.distance_transform_edt(mask, sampling=sampling)
    return (outside - inside).astype(np.float32)


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask)
    return mask & ~ndimage.binary_erosion(
        mask, structure=ndimage.generate_binary_structure(3, 1)
    )


# ------------------------------------------------------------------------ volume


def dice(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 and nb == 0:
        return float("nan")
    if na == 0 or nb == 0:
        return 0.0
    return float(2.0 * np.count_nonzero(a & b) / (na + nb))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float(np.count_nonzero(a & b) / union) if union else float("nan")


def signed_volume_difference(plan: np.ndarray, truth: np.ndarray) -> float:
    """(V_plan - V_truth) / V_truth. Positive when the planning contour is larger."""
    nt = int(truth.sum())
    if nt == 0:
        return float("nan")
    return float((int(plan.sum()) - nt) / nt)


def region_volumes(plan: np.ndarray, truth: np.ndarray, voxel_cc: float) -> dict[str, float]:
    """False-positive and false-negative volumes, kept separate on purpose."""
    return {
        "fp_volume_cc": float(np.count_nonzero(plan & ~truth)) * voxel_cc,
        "fn_volume_cc": float(np.count_nonzero(truth & ~plan)) * voxel_cc,
        "intersection_cc": float(np.count_nonzero(plan & truth)) * voxel_cc,
    }


# ----------------------------------------------------------------------- surface


def surface_metrics(plan: np.ndarray, truth: np.ndarray, grid: Grid,
                    *, surface_dice_tolerance_mm: float = 3.0) -> dict[str, float]:
    """Every surface-based metric in one pass, so the transforms are computed once.

    Distances are measured SURFACE TO SURFACE: from each surface voxel of one contour
    to the nearest surface voxel of the other. The tempting alternative -- evaluating a
    signed distance field of the whole region at the other contour's surface voxels --
    is wrong, because a surface voxel lies inside its own mask, so two identical
    contours report a non-zero mean surface distance of about half a voxel. This
    implementation returns exactly zero for identical contours, which is the property
    that makes the metric a metric.

    Sign is applied separately: a surface voxel of the planning contour lying outside
    the truth mask is positive, inside is negative.
    """
    empty = {
        "hd95_mm": float("nan"), "hd_max_mm": float("nan"),
        "msd_mm": float("nan"), "signed_msd_mm": float("nan"),
        "surface_dice": float("nan"), "apl_mm": float("nan"),
    }
    crop = joint_crop(plan, truth, grid)
    if crop is None:
        return empty
    p, t = crop.apply(plan), crop.apply(truth)
    if not p.any() or not t.any():
        return empty

    sp, st = _surface(p), _surface(t)
    if not sp.any() or not st.any():
        return empty

    # Distance to the nearest surface voxel of the other contour: zero on contact.
    to_t = ndimage.distance_transform_edt(~st, sampling=crop.sampling)
    to_p = ndimage.distance_transform_edt(~sp, sampling=crop.sampling)

    p_to_t = to_t[sp]          # unsigned, one value per planning-surface voxel
    t_to_p = to_p[st]
    both = np.concatenate([p_to_t, t_to_p])

    # Sign from containment, magnitude from the surface-to-surface distance.
    outside = ~t[sp]
    signed = np.where(outside, p_to_t, -p_to_t)

    tol = surface_dice_tolerance_mm
    within = np.count_nonzero(p_to_t <= tol) + np.count_nonzero(t_to_p <= tol)
    total = p_to_t.size + t_to_p.size

    # Added path length: planning-contour surface further than tolerance from the
    # truth surface, scaled to a length by the in-plane voxel size. A voxel-count
    # proxy for the contour length a user would have to redraw.
    apl_voxels = np.count_nonzero(p_to_t > tol)

    return {
        "hd95_mm": float(np.percentile(both, 95)),
        "hd_max_mm": float(both.max()),
        "msd_mm": float(both.mean()),
        "signed_msd_mm": float(signed.mean()),
        "surface_dice": float(within / total) if total else float("nan"),
        "apl_mm": float(apl_voxels) * float(grid.spacing[0]),
    }


# --------------------------------------------------------------------- dose-aware


def dose_aware_metrics(
    plan: np.ndarray,
    truth: np.ndarray,
    dose: np.ndarray,
    covered: np.ndarray,
    grid: Grid,
    *,
    high_dose_gy: float,
) -> dict[str, float]:
    """Metrics that only a dose distribution can supply.

    The point of restricting to the high-dose region is H7: only the irradiated part of
    a surface can influence a DVH, so whole-structure overlap dilutes the signal with
    disagreement that cannot matter dosimetrically.
    """
    voxel_cc = grid.voxel_volume_cc
    fp = plan & ~truth & covered
    fn = truth & ~plan & covered

    def _mean(region: np.ndarray) -> float:
        return float(dose[region].mean()) if region.any() else float("nan")

    def _integral(region: np.ndarray) -> float:
        return float(dose[region].sum()) * voxel_cc if region.any() else 0.0

    hot = covered & (np.nan_to_num(dose, nan=-1.0) >= high_dose_gy)
    out = {
        "mean_dose_in_fp_gy": _mean(fp),
        "mean_dose_in_fn_gy": _mean(fn),
        "integral_dose_in_fp_gy_cc": _integral(fp),
        "integral_dose_in_fn_gy_cc": _integral(fn),
        "dsc_in_high_dose": dice(plan & hot, truth & hot),
    }

    # Dose-weighted surface distance: surface disagreement counts for more where the
    # dose is high. Reported in mm.Gy.
    crop = joint_crop(plan, truth, grid)
    if crop is not None:
        p, t = crop.apply(plan), crop.apply(truth)
        if p.any() and t.any() and _surface(t).any():
            st = _surface(t)
            sp = _surface(p)
            to_t = ndimage.distance_transform_edt(~st, sampling=crop.sampling)
            d_here = np.nan_to_num(crop.apply(dose), nan=0.0)[sp]
            dist = to_t[sp]
            weight = d_here.sum()
            out["dose_weighted_surface_distance_mm"] = (
                float((dist * d_here).sum() / weight) if weight > 0 else float("nan")
            )
        else:
            out["dose_weighted_surface_distance_mm"] = float("nan")
    else:
        out["dose_weighted_surface_distance_mm"] = float("nan")
    return out


def compare(
    plan: np.ndarray,
    truth: np.ndarray,
    grid: Grid,
    *,
    surface_dice_tolerance_mm: float = 3.0,
) -> dict[str, float]:
    """All purely geometric metrics for one ordered (plan, truth) pair."""
    out: dict[str, float] = {
        "dsc": dice(plan, truth),
        "jaccard": jaccard(plan, truth),
        "signed_volume_difference": signed_volume_difference(plan, truth),
        "volume_plan_cc": float(plan.sum()) * grid.voxel_volume_cc,
        "volume_truth_cc": float(truth.sum()) * grid.voxel_volume_cc,
    }
    out.update(region_volumes(plan, truth, grid.voxel_volume_cc))
    out.update(surface_metrics(plan, truth, grid,
                               surface_dice_tolerance_mm=surface_dice_tolerance_mm))
    return out
