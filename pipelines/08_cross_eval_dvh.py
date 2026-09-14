"""Stage 08 — the cross-evaluation. Every plan scored against every plausible anatomy.

    python pipelines/08_cross_eval_dvh.py [--patient K018]

For each of the 530 primary cells (patient, plan i, truth j) this computes every
endpoint in config/endpoints.yaml on truth j's structures using plan i's dose, plus the
coverage-degradation ladder, the conformity indices and every computable protocol
constraint.

The one rule that governs the whole stage: **the plan enters only through the dose
distribution.** Every structure used to score a cell belongs to the assumed true anatomy
j, including the composite ones. Mixing j's rectum with i's PTV would look defensible
and would quietly confound plan-side with evaluation-side effects, which is exactly the
separation H5 exists to test.

Writes results/endpoints_long.parquet and results/constraints_long.parquet.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import Config, load_config  # noqa: E402
from mcx.dose.store import DoseStore  # noqa: E402
from mcx.endpoints.evaluate import (  # noqa: E402
    boundary_gradient,
    evaluate,
    ladder_specs,
    oar_specs,
    structure_dose,
    target_specs,
)
from mcx.geom.analysis_grid import load_grid  # noqa: E402
from mcx.geom.masks import MaskStore  # noqa: E402
from mcx.geom.shells import paddick_conformity  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402

# Composite and derived structures that a constraint may reference.
DERIVED_FOR_CONSTRAINTS = (
    "Rectum_PRV_3mm", "Rectum_in_PTV", "Rectum_outside_PTV", "Peripheral_Tissue",
)


def evaluate_cell(
    cfg: Config, store: MaskStore, patient: str, plan_set: str, truth_set: str,
    dose: np.ndarray, covered: np.ndarray, treated: np.ndarray, voxel_cc: float,
) -> tuple[list[dict], list[dict]]:
    """All endpoints and constraints for one (plan, truth) cell."""
    arm = cfg.arm(plan_set, truth_set)
    base = {"patient": patient, "plan_set": plan_set, "truth_set": truth_set, "arm": arm}
    endpoints: list[dict] = []
    constraints: list[dict] = []

    cached: dict[str, np.ndarray] = {}

    def desc_for(structure: str) -> np.ndarray | None:
        if structure not in cached:
            if not store.has(patient, truth_set, structure):
                cached[structure] = np.empty(0)
            else:
                cached[structure] = structure_dose(
                    dose, covered, store.get(patient, truth_set, structure)
                )
        d = cached[structure]
        return d if d.size else None

    # ---- target
    for spec in target_specs(cfg):
        d = desc_for("CTV")
        endpoints.append({**base, "structure": "CTV", "endpoint": spec["id"],
                          "value": evaluate(spec, d, voxel_cc) if d is not None
                          else float("nan"), "unit": spec.get("unit")})

    # ---- OARs
    for structure, specs in oar_specs(cfg).items():
        d = desc_for(structure)
        for spec in specs:
            endpoints.append({**base, "structure": structure, "endpoint": spec["id"],
                              "value": evaluate(spec, d, voxel_cc) if d is not None
                              else float("nan"), "unit": spec.get("unit")})

    # ---- coverage-degradation ladder and its boundary gradient
    margins, ladder = ladder_specs(cfg)
    d98_by_margin: dict[float, float] = {}
    for margin in margins:
        name = "CTV" if margin == 0 else f"CTV_plus_{int(margin)}mm"
        d = desc_for(name)
        for spec in ladder:
            value = evaluate(spec, d, voxel_cc) if d is not None else float("nan")
            endpoints.append({**base, "structure": name,
                              "endpoint": f"{spec['id']}_{int(margin)}mm",
                              "value": value, "unit": spec.get("unit")})
            if spec["id"] == "ladder_d98":
                d98_by_margin[margin] = value
    endpoints.append({
        **base, "structure": "CTV", "endpoint": "boundary_dose_gradient",
        "value": boundary_gradient(d98_by_margin, cfg.prescription_gy),
        "unit": "percent_per_mm",
    })

    # ---- conformity: the treated volume is a property of the plan, scored against j
    ctv_mask = (store.get(patient, truth_set, "CTV")
                if store.has(patient, truth_set, "CTV") else None)
    endpoints.append({**base, "structure": "CTV", "endpoint": "treated_volume_cc",
                      "value": float(treated.sum()) * voxel_cc, "unit": "cc"})
    endpoints.append({
        **base, "structure": "CTV", "endpoint": "paddick_ci",
        "value": paddick_conformity(treated, ctv_mask) if ctv_mask is not None
        else float("nan"), "unit": "dimensionless",
    })

    # ---- protocol constraints
    for c in cfg.computable_constraints:
        structure = c.get("structure")
        if not structure or c["applicability"] == "patient":
            continue
        if c["metric"]["kind"] not in (
            "d_at_volume_pct", "v_at_dose_pct", "v_at_dose_cc", "d_at_volume_cc",
            "mean_dose", "d_max",
        ):
            continue
        d = desc_for(structure)
        value = evaluate(c["metric"], d, voxel_cc) if d is not None else float("nan")
        constraints.append({
            **base, "constraint": c["id"], "structure": structure,
            "label": c["label"], "value": value, "unit": c["unit"],
            "tier": cfg.constraint_tier(c, value),
        })

    return endpoints, constraints


def run_patient(cfg: Config, patient: str, store: MaskStore, doses: DoseStore):
    grid = load_grid(cfg, patient)
    vox = grid.voxel_volume_cc
    plans = [s for s in cfg.analysis_sets if cfg.is_known_absent(patient, s) is None]
    truths = [s for s in cfg.truth_sets if cfg.is_known_absent(patient, s) is None]

    endpoints: list[dict] = []
    constraints: list[dict] = []
    for plan_set in plans:
        dose, covered = doses.get(patient, plan_set)
        treated = store.get(patient, plan_set, "Treated_Volume")
        for truth_set in truths:
            e, c = evaluate_cell(cfg, store, patient, plan_set, truth_set,
                                 dose, covered, treated, vox)
            endpoints.extend(e)
            constraints.extend(c)
        doses.drop_patient(patient)
    store.drop_patient(patient)
    return endpoints, constraints


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", action="append")
    args = ap.parse_args()

    cfg = load_config()
    run = RunInfo.create("08_cross_eval_dvh", cfg)
    store, doses = MaskStore(cfg), DoseStore(cfg)

    all_e, all_c = [], []
    for patient in (args.patient or cfg.patients):
        t0 = time.perf_counter()
        e, c = run_patient(cfg, patient, store, doses)
        all_e.extend(e)
        all_c.extend(c)
        cells = len({(r["plan_set"], r["truth_set"]) for r in e})
        print(f"  {patient}: {cells} cells, {len(e)} endpoint values, "
              f"{time.perf_counter() - t0:.0f}s")

    ep = pd.DataFrame(all_e)
    cs = pd.DataFrame(all_c)
    write_table(ep, cfg.results_root / "endpoints_long.parquet", run)
    write_table(cs, cfg.results_root / "constraints_long.parquet", run)

    cells = ep[["patient", "plan_set", "truth_set"]].drop_duplicates()
    expected = len(cfg.design_cells())
    print(f"\n{len(cells)} cells (design expects {expected})")
    print(f"{len(ep)} endpoint values, {ep['endpoint'].nunique()} distinct endpoints")
    print(f"{len(cs)} constraint evaluations, {cs['constraint'].nunique()} constraints")
    print("\ncells by arm:")
    print(cells.merge(ep[["patient", "plan_set", "truth_set", "arm"]].drop_duplicates(),
                      on=["patient", "plan_set", "truth_set"])["arm"]
          .value_counts().to_string())
    nan_rate = 100 * ep["value"].isna().mean()
    print(f"\nmissing endpoint values: {nan_rate:.2f}%")
    if nan_rate > 0:
        worst = ep[ep["value"].isna()]["endpoint"].value_counts().head(8)
        print(worst.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
