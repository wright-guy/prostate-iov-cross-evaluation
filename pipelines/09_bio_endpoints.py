"""Stage 09 — TCP and NTCP for every cell, and the volume-normalisation artefact.

    python pipelines/09_bio_endpoints.py [--patient K018]

Computes, for all 530 cells:
  * NTCP for rectum and bladder, under RELATIVE and ABSOLUTE volume normalisation
  * TCP under FIXED CLONOGEN DENSITY and FIXED CLONOGEN NUMBER
  * each TCP at the calibrated density and at the published one, as a sensitivity

Reporting both members of each pair is what isolates §5's volume-normalisation
artefact. A larger scoring contour dilutes a relative DVH and inflates a fixed-density
clonogen count, both of which move the biological endpoint without any change in the
delivered dose. The difference between the two members of a pair IS that artefact, and
quantifying it turns a caveat into a number.

Reference volumes are recomputed from results/structures.parquet rather than hard-coded,
so they cannot drift away from the cohort they describe.

Writes results/bio_long.parquet and results/bio_calibration.parquet.
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
from mcx.endpoints.biological import (  # noqa: E402
    calibrate_clonogen_density,
    lkb_ntcp,
    poisson_tcp,
)
from mcx.geom.analysis_grid import load_grid  # noqa: E402
from mcx.geom.masks import MaskStore  # noqa: E402
from mcx.provenance import RunInfo, write_table  # noqa: E402


def reference_volumes(cfg: Config) -> dict[str, float]:
    """Cohort median volume per structure, over the observer contour sets."""
    df = pd.read_parquet(cfg.results_root / "structures.parquet")
    obs = df[df["present"] & df["contour_set"].isin(cfg.observers)]
    return obs.groupby("structure")["volume_cc"].median().to_dict()


def calibration(cfg: Config, ctv_reference_cc: float) -> pd.DataFrame:
    """Solve for the clonogen density and record the model's operating point."""
    t = cfg.models_raw["tcp"]
    cal = t["calibration"]
    rho = calibrate_clonogen_density(
        target_tcp=float(cal["target_tcp"]),
        uniform_dose_gy=float(cal["at_uniform_dose_gy"]),
        volume_cc=ctv_reference_cc,
        alpha_mean=float(t["alpha_mean_per_gy"]),
        sigma_alpha=float(t["sigma_alpha_per_gy"]),
        alpha_beta_gy=float(t["alpha_beta_gy"]),
        n_fractions=cfg.n_fractions,
    )
    rows = []
    for label, density in (("calibrated", rho),
                           ("as_published", float(t["clonogen_density_per_cc"]))):
        n_vox = 20_000
        def _tcp(dose_gy: float, d=density, n=n_vox) -> float:
            return poisson_tcp(
                np.full(n, dose_gy), alpha_mean=float(t["alpha_mean_per_gy"]),
                sigma_alpha=float(t["sigma_alpha_per_gy"]),
                alpha_beta_gy=float(t["alpha_beta_gy"]),
                clonogen_density_per_cc=d, n_fractions=cfg.n_fractions,
                voxel_volume_cc=ctv_reference_cc / n,
            )["TCP"]
        rx = cfg.prescription_gy
        rows.append({
            "density_label": label, "clonogen_density_per_cc": density,
            "reference_volume_cc": ctv_reference_cc,
            "tcp_at_prescription": _tcp(rx),
            "local_slope_per_gy": (_tcp(rx + 1.0) - _tcp(rx - 1.0)) / 2.0,
        })
    return pd.DataFrame(rows)


def run_patient(cfg: Config, patient: str, store: MaskStore, doses: DoseStore,
                refs: dict[str, float], densities: dict[str, float]) -> list[dict]:
    grid = load_grid(cfg, patient)
    vox = grid.voxel_volume_cc
    t = cfg.models_raw["tcp"]
    plans = [s for s in cfg.analysis_sets if cfg.is_known_absent(patient, s) is None]
    truths = [s for s in cfg.truth_sets if cfg.is_known_absent(patient, s) is None]

    rows: list[dict] = []
    for plan_set in plans:
        dose, covered = doses.get(patient, plan_set)
        for truth_set in truths:
            base = {"patient": patient, "plan_set": plan_set, "truth_set": truth_set,
                    "arm": cfg.arm(plan_set, truth_set)}

            # ---- NTCP, both normalisations
            for spec in cfg.models_raw["ntcp"]["endpoints"]:
                structure = spec["structure"]
                if not store.has(patient, truth_set, structure):
                    continue
                mask = store.get(patient, truth_set, structure)
                values = dose[mask & covered]
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                for norm, ref in (("relative", None),
                                  ("absolute", refs.get(structure))):
                    if norm == "absolute" and ref is None:
                        continue
                    out = lkb_ntcp(
                        values, n=float(spec["n"]), m=float(spec["m"]),
                        td50_gy=float(spec["td50_gy"]), n_fractions=cfg.n_fractions,
                        alpha_beta_gy=float(spec["alpha_beta_gy"]),
                        voxel_volume_cc=vox, reference_volume_cc=ref,
                    )
                    rows.append({**base, "model": "ntcp", "endpoint_id": spec["id"],
                                 "structure": structure, "variant": norm,
                                 "density_label": None,
                                 "value": out["NTCP"], "geud_gy": out["gEUD"],
                                 "volume_cc": out["volume_cc"]})

            # ---- TCP, both clonogen conventions and both densities
            if store.has(patient, truth_set, "CTV"):
                mask = store.get(patient, truth_set, "CTV")
                values = dose[mask & covered]
                values = values[np.isfinite(values)]
                if values.size:
                    for label, density in densities.items():
                        for conv, ref in (("fixed_density", None),
                                          ("fixed_number", refs.get("CTV"))):
                            out = poisson_tcp(
                                values, alpha_mean=float(t["alpha_mean_per_gy"]),
                                sigma_alpha=float(t["sigma_alpha_per_gy"]),
                                alpha_beta_gy=float(t["alpha_beta_gy"]),
                                clonogen_density_per_cc=density,
                                n_fractions=cfg.n_fractions,
                                voxel_volume_cc=vox, reference_volume_cc=ref,
                            )
                            rows.append({**base, "model": "tcp", "endpoint_id": "tcp",
                                         "structure": "CTV", "variant": conv,
                                         "density_label": label,
                                         "value": out["TCP"], "geud_gy": np.nan,
                                         "volume_cc": out["volume_cc"]})
        doses.drop_patient(patient)
    store.drop_patient(patient)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", action="append")
    args = ap.parse_args()

    cfg = load_config()
    run = RunInfo.create("09_bio_endpoints", cfg)
    store, doses = MaskStore(cfg), DoseStore(cfg)

    refs = reference_volumes(cfg)
    print("reference volumes (cohort median over observers):")
    for k in ("CTV", "Rectum", "Bladder"):
        if k in refs:
            print(f"  {k:8} {refs[k]:.2f} cc")

    cal = calibration(cfg, refs["CTV"])
    write_table(cal, cfg.results_root / "bio_calibration.parquet", run)
    print("\nTCP model operating point:")
    print(cal.round(6).to_string(index=False))
    densities = dict(zip(cal["density_label"], cal["clonogen_density_per_cc"], strict=True))

    rows = []
    for patient in (args.patient or cfg.patients):
        t0 = time.perf_counter()
        got = run_patient(cfg, patient, store, doses, refs, densities)
        rows.extend(got)
        print(f"  {patient}: {len(got)} values, {time.perf_counter() - t0:.0f}s")

    df = pd.DataFrame(rows)
    write_table(df, cfg.results_root / "bio_long.parquet", run)

    pd.set_option("display.width", 220)
    print(f"\n{len(df)} biological values over "
          f"{df[['patient', 'plan_set', 'truth_set']].drop_duplicates().shape[0]} cells")
    print("\n=== NTCP by structure and normalisation (median [min, max] over all cells) ===")
    n = df[df["model"] == "ntcp"]
    print(n.groupby(["structure", "variant"])["value"]
          .agg(median="median", lo="min", hi="max").round(4).to_string())
    print("\n=== TCP by convention and density ===")
    t = df[df["model"] == "tcp"]
    print(t.groupby(["density_label", "variant"])["value"]
          .agg(median="median", lo="min", hi="max").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
