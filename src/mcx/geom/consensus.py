"""Majority-vote consensus contours, and their leave-one-out variants.

The supplied VOTE contours were produced by an unnamed Python library. Section 7.2
requires ten leave-one-out consensus contours per patient per structure, which means
the algorithm has to be reproduced -- and a reimplementation that merely looks
reasonable is not good enough, because H1 and H2 would then be measuring my code rather
than the consensus method.

The approach agreed with the user: reimplement simple majority vote, verify against the
SUPPLIED VOTE contours across every patient and structure, and only generate the
leave-one-out variants if the reproduction succeeds to a stated tolerance. If it fails,
H1 is reported against the 1.49 uncorrected benchmark alone and the omission is stated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def vote_count(masks: list[np.ndarray]) -> np.ndarray:
    """Number of observers including each voxel."""
    if not masks:
        raise ValueError("no masks supplied")
    total = np.zeros(masks[0].shape, dtype=np.int16)
    for m in masks:
        total += m.astype(np.int16)
    return total


def majority_vote(masks: list[np.ndarray], *, threshold: int | None = None) -> np.ndarray:
    """Voxels included by at least ``threshold`` observers.

    ``threshold`` defaults to a strict majority, ceil(n/2) for odd n and n/2 + 1 for
    even n. With ten observers that is 6, not 5: a 5-5 split is not a majority.
    """
    counts = vote_count(masks)
    n = len(masks)
    k = threshold if threshold is not None else (n // 2 + 1)
    return counts >= k


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice similarity coefficient. 1.0 for identical non-empty masks."""
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 and nb == 0:
        return float("nan")
    if na == 0 or nb == 0:
        return 0.0
    return float(2.0 * np.count_nonzero(a & b) / (na + nb))


@dataclass(frozen=True)
class ReproductionResult:
    """How well a reimplemented vote matches the supplied VOTE contour."""

    patient: str
    structure: str
    n_observers: int
    best_threshold: int
    best_dice: float
    supplied_volume_cc: float
    reproduced_volume_cc: float
    dice_by_threshold: dict[int, float]

    @property
    def volume_ratio(self) -> float:
        if self.supplied_volume_cc == 0:
            return float("nan")
        return self.reproduced_volume_cc / self.supplied_volume_cc


def find_threshold(
    observer_masks: list[np.ndarray],
    supplied: np.ndarray,
    *,
    patient: str,
    structure: str,
    voxel_volume_cc: float,
) -> ReproductionResult:
    """Sweep every possible vote threshold and report which best matches ``supplied``.

    Sweeping rather than assuming a strict majority is the point: if the supplied VOTE
    turns out to match a different threshold, that is a fact about how it was generated
    and it needs to be known before any leave-one-out contour is built on the
    assumption.
    """
    n = len(observer_masks)
    counts = vote_count(observer_masks)
    scores = {k: dice(counts >= k, supplied) for k in range(1, n + 1)}
    best = max(scores, key=lambda k: (scores[k] if not np.isnan(scores[k]) else -1.0))
    return ReproductionResult(
        patient=patient,
        structure=structure,
        n_observers=n,
        best_threshold=best,
        best_dice=scores[best],
        supplied_volume_cc=float(supplied.sum()) * voxel_volume_cc,
        reproduced_volume_cc=float((counts >= best).sum()) * voxel_volume_cc,
        dice_by_threshold=scores,
    )


def leave_one_out_votes(
    observer_masks: dict[str, np.ndarray], *, threshold_rule: str = "strict_majority"
) -> dict[str, np.ndarray]:
    """One consensus per held-out observer, built from the other n-1.

    This is what H1's 1.34 benchmark needs. Comparing an observer against a consensus
    that includes that observer is circular: the observer has voted for their own
    contour, which inflates the agreement. Holding them out removes that leakage.
    """
    if threshold_rule != "strict_majority":
        raise ValueError(f"unsupported threshold rule {threshold_rule!r}")
    out: dict[str, np.ndarray] = {}
    for held_out in observer_masks:
        others = [m for name, m in observer_masks.items() if name != held_out]
        out[held_out] = majority_vote(others)
    return out


def signed_distance_bias(
    observer_masks: list[np.ndarray], consensus: np.ndarray, grid
) -> dict[str, float]:  # noqa: ANN001 - Grid, avoids a circular import
    """Where the consensus surface sits relative to the observers' mean surface.

    H2 says a majority vote is a median-like morphological operator, not a mean of
    signed distance fields, and is therefore a biased estimator of the observer mean
    surface. This measures that bias directly: the mean over observers of the signed
    distance from each observer's surface to the consensus surface. Zero would mean the
    consensus sits exactly at the mean surface.
    """
    from mcx.geom.surface import signed_distance, surface_voxels

    if not consensus.any():
        return {"mean_signed_mm": float("nan"), "volume_ratio": float("nan")}

    shell = surface_voxels(consensus)
    per_observer = []
    for m in observer_masks:
        if not m.any():
            continue
        # Positive where the consensus surface lies outside that observer's contour.
        per_observer.append(float(np.mean(signed_distance(m, grid)[shell])))

    consensus_cc = float(consensus.sum())
    mean_observer_cc = float(np.mean([m.sum() for m in observer_masks]))
    return {
        "mean_signed_mm": float(np.mean(per_observer)) if per_observer else float("nan"),
        "sd_signed_mm": float(np.std(per_observer, ddof=1)) if len(per_observer) > 1
        else float("nan"),
        "volume_ratio": consensus_cc / mean_observer_cc if mean_observer_cc else float("nan"),
    }
