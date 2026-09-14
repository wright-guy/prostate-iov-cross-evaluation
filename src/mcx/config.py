"""Configuration loading and path resolution.

Every path the pipeline touches is resolved here, so no module ever contains a
literal path and no stage can accidentally write into ``data/``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
OBSERVER_KEY_FILE = "observer_key.local.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _hash_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


@dataclass(frozen=True)
class Structure:
    """One canonical structure, resolved from ``config/structures.yaml``."""

    name: str
    roi_template: str
    role: str
    in_analysis: bool
    fill: str
    plausible_volume_cc: tuple[float, float]
    absolute_volume_endpoints: bool = False
    note: str | None = None


@dataclass
class Config:
    dataset: dict[str, Any] = field(default_factory=dict)
    structures_raw: dict[str, Any] = field(default_factory=dict)
    waivers_raw: dict[str, Any] = field(default_factory=dict)
    grid_raw: dict[str, Any] = field(default_factory=dict)
    constraints_raw: dict[str, Any] = field(default_factory=dict)
    endpoints_raw: dict[str, Any] = field(default_factory=dict)
    metrics_raw: dict[str, Any] = field(default_factory=dict)
    analysis_raw: dict[str, Any] = field(default_factory=dict)
    models_raw: dict[str, Any] = field(default_factory=dict)
    # Optional, never committed: maps each contour-set label used in config and results
    # to the name the raw DICOM uses, so observers can be anonymised in everything the
    # pipeline writes. Absent, every label is its own raw name.
    observer_key_raw: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, config_dir: Path | None = None) -> Config:
        cd = config_dir or CONFIG_DIR
        key_path = cd / OBSERVER_KEY_FILE
        return cls(
            observer_key_raw=_load_yaml(key_path) if key_path.exists() else {},
            dataset=_load_yaml(cd / "dataset.yaml"),
            structures_raw=_load_yaml(cd / "structures.yaml"),
            waivers_raw=_load_yaml(cd / "waivers.yaml"),
            grid_raw=_load_yaml(cd / "grid.yaml"),
            constraints_raw=_load_yaml(cd / "constraints.yaml"),
            endpoints_raw=_load_yaml(cd / "endpoints.yaml"),
            metrics_raw=_load_yaml(cd / "metrics.yaml"),
            analysis_raw=_load_yaml(cd / "analysis.yaml"),
            models_raw=_load_yaml(cd / "models.yaml"),
        )

    @cached_property
    def config_hash(self) -> str:
        """Full configuration fingerprint, stamped on every results row."""
        return _hash_obj(
            [
                self.dataset, self.structures_raw, self.waivers_raw,
                self.grid_raw, self.constraints_raw, self.endpoints_raw,
                self.metrics_raw, self.analysis_raw, self.models_raw,
            ]
        )

    @cached_property
    def geometry_hash(self) -> str:
        """Fingerprint of only what the cached masks and doses actually depend on.

        Cache keys use this rather than ``config_hash`` so that editing a comment, a
        plausibility band or a waiver justification does not discard hundreds of
        megabytes of correct geometry. It errs toward over-invalidation: anything that
        could plausibly move a voxel is included, and anything excluded is inert text or
        QC policy that runs downstream of the cache.
        """
        parts = [
            {
                k: self.dataset.get(k)
                for k in ("patients", "observers", "derived_sets", "folder_alias",
                          "known_absent", "roots")
            },
            {
                name: {k: spec.get(k) for k in ("roi", "fill")}
                for name, spec in self.structures_raw["structures"].items()
            },
            self.structures_raw.get("overrides"),
            self.structures_raw.get("roi_exceptions"),
            self.structures_raw.get("partial_sets"),
            self.grid_raw.get("analysis_grid"),
        ]
        if self.observer_key_raw:  # appended only when present, so existing caches keep their key
            parts.append(self.observer_key_raw)
        return _hash_obj(parts)

    # ------------------------------------------------------------------ paths

    def _root(self, key: str) -> Path:
        return REPO_ROOT / self.dataset["roots"][key]

    @property
    def dicom_root(self) -> Path:
        return self._root("dicom")

    @property
    def dose_root(self) -> Path:
        return self._root("dose")

    @property
    def rtplan_root(self) -> Path:
        return self._root("rtplan")

    @cached_property
    def derived_root(self) -> Path:
        """Cache directory, deliberately resolvable outside OneDrive."""
        env = os.environ.get("MCX_DERIVED")
        if env:
            p = Path(env)
        elif self.dataset.get("derived"):
            p = REPO_ROOT / self.dataset["derived"]
        else:
            base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
            if base:
                p = Path(base) / "MultiContouring" / "derived"
            else:
                p = REPO_ROOT / "derived"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results_root(self) -> Path:
        p = REPO_ROOT / "results"
        p.mkdir(exist_ok=True)
        return p

    @property
    def qc_root(self) -> Path:
        p = REPO_ROOT / "qc"
        p.mkdir(exist_ok=True)
        return p

    # ------------------------------------------------------------- dataset API

    @property
    def patients(self) -> list[str]:
        return list(self.dataset["patients"])

    @property
    def observers(self) -> list[str]:
        return list(self.dataset["observers"])

    @property
    def derived_sets(self) -> list[str]:
        return list(self.dataset.get("derived_sets", []))

    @property
    def all_sets(self) -> list[str]:
        return self.observers + self.derived_sets

    @property
    def analysis_sets(self) -> list[str]:
        """Contour sets whose plans enter the analysis (index i)."""
        return list(self.dataset["analysis_sets"])

    @property
    def truth_sets(self) -> list[str]:
        """Contour sets admissible as the assumed true anatomy (index j)."""
        return list(self.dataset["truth_sets"])

    @property
    def consensus_set(self) -> str:
        return str(self.dataset["consensus_set"])

    def arm(self, plan_set: str, truth_set: str) -> str:
        """Which arm of the design a cross-evaluation cell belongs to."""
        if plan_set == truth_set:
            return "diagonal"
        if plan_set == self.consensus_set:
            return "C"
        if plan_set in self.observers and truth_set in self.observers:
            return "S"
        return "other"

    def design_cells(self) -> list[tuple[str, str, str, str]]:
        """Every (patient, plan_set i, truth_set j, arm) in the primary design.

        This is the ground truth for the Tier 4 completeness check: the realised cell
        count must equal len(design_cells()) exactly.
        """
        cells = []
        for p in self.patients:
            for i in self.analysis_sets:
                if self.is_known_absent(p, i) is not None:
                    continue
                for j in self.truth_sets:
                    if self.is_known_absent(p, j) is not None:
                        continue
                    cells.append((p, i, j, self.arm(i, j)))
        return cells

    @property
    def _folder_alias(self) -> dict[str, str]:
        return {**(self.dataset.get("folder_alias") or {}),
                **(self.observer_key_raw.get("folder_alias") or {})}

    def raw_set(self, contour_set: str) -> str:
        """The name the raw DICOM uses for a contour-set label (see ``observer_key_raw``)."""
        return (self.observer_key_raw.get("observers") or {}).get(contour_set, contour_set)

    def canonical_set(self, folder_name: str) -> str:
        """Map a dose folder name to its canonical contour-set id.

        Never fall back to case-folding: an unmapped mismatch is a QC failure, not
        something to paper over.
        """
        raw = self._folder_alias.get(folder_name, folder_name)
        labels = {v: k for k, v in (self.observer_key_raw.get("observers") or {}).items()}
        return labels.get(raw, raw)

    def dose_folders(self, contour_set: str) -> list[str]:
        """Folder names under which a contour set's dose may be stored."""
        raw = self.raw_set(contour_set)
        return [raw] + [k for k, v in self._folder_alias.items() if v == raw]

    def is_known_absent(self, patient: str, contour_set: str) -> str | None:
        """Return the documented reason a (patient, set) pair is missing, else None."""
        for entry in self.dataset.get("known_absent") or []:
            if entry["patient"] == patient and entry["contour_set"] == contour_set:
                return entry["reason"]
        return None

    def expected_pairs(self, sets: list[str] | None = None) -> list[tuple[str, str]]:
        """Every (patient, contour_set) that must exist, minus documented gaps."""
        sets = sets if sets is not None else self.all_sets
        return [
            (p, s)
            for p in self.patients
            for s in sets
            if self.is_known_absent(p, s) is None
        ]

    # ----------------------------------------------------------- structure API

    @cached_property
    def structures(self) -> dict[str, Structure]:
        out: dict[str, Structure] = {}
        for name, spec in self.structures_raw["structures"].items():
            lo, hi = spec["plausible_volume_cc"]
            out[name] = Structure(
                name=name,
                roi_template=spec["roi"],
                role=spec["role"],
                in_analysis=bool(spec["in_analysis"]),
                fill=spec.get("fill", "even_odd"),
                plausible_volume_cc=(float(lo), float(hi)),
                absolute_volume_endpoints=bool(spec.get("absolute_volume_endpoints", False)),
                note=spec.get("note"),
            )
        return out

    @property
    def analysis_structures(self) -> list[str]:
        return [n for n, s in self.structures.items() if s.in_analysis]

    def structures_for_set(self, contour_set: str) -> list[str]:
        """Structures a given contour set is expected to carry."""
        partial: dict[str, list[str]] = self.structures_raw.get("partial_sets") or {}
        if contour_set in partial:
            return list(partial[contour_set])
        return list(self.structures)

    def roi_name(self, patient: str, contour_set: str, structure: str) -> str:
        """The RTSTRUCT ROI name expected for this triple, honouring overrides.

        Override and exception names may use ``{set}`` for the raw contour-set name.
        """
        raw = self.raw_set(contour_set)
        for ov in self.structures_raw.get("overrides") or []:
            if (
                ov["patient"] == patient
                and ov["contour_set"] == contour_set
                and ov["structure"] == structure
            ):
                return ov["roi"].format(set=raw)
        for ex in self.structures_raw.get("roi_exceptions") or []:
            if ex["contour_set"] == contour_set and ex["structure"] == structure:
                return ex["roi"].format(set=raw)
        return self.structures[structure].roi_template.format(set=raw)

    # ---------------------------------------------------------- constraint API

    @property
    def prescription_gy(self) -> float:
        return float(self.dataset["prescription"]["total_dose_gy"])

    @property
    def n_fractions(self) -> int:
        return int(self.dataset["prescription"]["n_fractions"])

    @cached_property
    def constraints(self) -> list[dict[str, Any]]:
        """Every protocol constraint, including the ones we cannot compute.

        The unavailable ones are kept rather than deleted so the methods can state
        which parts of the department protocol this dataset cannot address, and why.
        """
        return list(self.constraints_raw.get("constraints") or [])

    @property
    def computable_constraints(self) -> list[dict[str, Any]]:
        return [c for c in self.constraints if c["applicability"] != "unavailable"]

    @staticmethod
    def constraint_tier(constraint: dict[str, Any], value: float) -> str:
        """Classify a value into per_protocol / minor / major.

        Returns "under_dose" when a two-sided constraint falls below its floor: the
        protocol table does not tier that case, and guessing would be worse than
        naming it.
        """
        import math

        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "undefined"
        tiers = constraint["tiers"]
        floor = constraint.get("floor")
        if floor is not None and value < float(floor):
            return "under_dose"
        if constraint["better"] == "lower":
            if value < float(tiers["per_protocol"]):
                return "per_protocol"
            return "minor" if value < float(tiers["minor"]) else "major"
        if constraint["better"] == "higher":
            if value >= float(tiers["per_protocol"]):
                return "per_protocol"
            return "minor" if value >= float(tiers["minor"]) else "major"
        raise ValueError(f"{constraint['id']}: unsupported direction {constraint['better']!r}")

    # -------------------------------------------------------------- waiver API

    @cached_property
    def waivers(self) -> list[dict[str, Any]]:
        return list(self.waivers_raw.get("waivers") or [])

    def waiver_for(self, check: str, **scope: str) -> dict[str, Any] | None:
        """Find a waiver covering this check at this scope, or None.

        A waiver with no ``scope`` covers every occurrence of the check; a waiver with
        a scope covers only occurrences whose keys all match.
        """
        for w in self.waivers:
            if w["check"] != check:
                continue
            wscope = w.get("scope") or {}
            if all(scope.get(k) == v for k, v in wscope.items()):
                return w
        return None


def load_config(config_dir: Path | None = None) -> Config:
    return Config.load(config_dir)
