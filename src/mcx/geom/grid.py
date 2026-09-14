"""Voxel grids in patient coordinates.

One ``Grid`` type covers the native dose grid, the CT grid and the common analysis
grid, so nothing downstream has to care which it is holding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Axis-aligned voxel grid. ``origin`` is the centre of voxel (0, 0, 0)."""

    origin: np.ndarray  # (3,) x, y, z in mm
    spacing: np.ndarray  # (3,) dx, dy, dz in mm
    shape: tuple[int, int, int]  # (nz, ny, nx)

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=float))
        object.__setattr__(self, "spacing", np.asarray(self.spacing, dtype=float))
        if np.any(self.spacing <= 0):
            raise ValueError(f"non-positive spacing {self.spacing}")

    @property
    def nz(self) -> int:
        return self.shape[0]

    @property
    def voxel_volume_cc(self) -> float:
        return float(np.prod(self.spacing)) / 1000.0

    @property
    def z(self) -> np.ndarray:
        return self.origin[2] + np.arange(self.shape[0]) * self.spacing[2]

    def world_to_voxel(self, points: np.ndarray) -> np.ndarray:
        """(n, 3) patient mm -> (n, 3) fractional voxel indices in (x, y, z) order."""
        return (np.asarray(points, dtype=float) - self.origin) / self.spacing

    def voxel_to_world(self, idx: np.ndarray) -> np.ndarray:
        return np.asarray(idx, dtype=float) * self.spacing + self.origin

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Voxel-centre bounding box (lo, hi)."""
        extent = (np.array(self.shape[::-1]) - 1) * self.spacing
        return self.origin.copy(), self.origin + extent

    def contains(self, points: np.ndarray, *, margin_mm: float = 0.0) -> np.ndarray:
        lo, hi = self.bounds()
        pts = np.asarray(points, dtype=float)
        return np.all((pts >= lo - margin_mm) & (pts <= hi + margin_mm), axis=1)

    def nearest_slice(self, z: float) -> int:
        return int(np.clip(round((z - self.origin[2]) / self.spacing[2]), 0, self.shape[0] - 1))

    @classmethod
    def from_bounds(
        cls,
        lo: np.ndarray,
        hi: np.ndarray,
        spacing: float | np.ndarray,
        *,
        margin_mm: float = 0.0,
    ) -> Grid:
        """Smallest grid of the given spacing covering ``lo``..``hi`` plus a margin."""
        sp = np.full(3, float(spacing)) if np.isscalar(spacing) else np.asarray(spacing, float)
        lo = np.asarray(lo, float) - margin_mm
        hi = np.asarray(hi, float) + margin_mm
        n = np.ceil((hi - lo) / sp).astype(int) + 1
        return cls(origin=lo, spacing=sp, shape=(int(n[2]), int(n[1]), int(n[0])))
