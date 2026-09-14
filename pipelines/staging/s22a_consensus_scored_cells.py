"""Section 2.2 -- fixed dose, varied evaluation contour.

Needs two cells the main pipeline does not produce, because VOTE and the
leave-one-out consensus are excluded from truth_sets:

    E_{i,VOTE}      observer i's dose scored on the consensus contour
    E_{i,LOO_i}     observer i's dose scored on the consensus that excludes i
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from mcx.config import load_config
from mcx.dose.store import DoseStore
from mcx.endpoints.evaluate import evaluate, oar_specs, structure_dose, target_specs
from mcx.geom.analysis_grid import load_grid
from mcx.geom.masks import MaskStore
from mcx.geom.shells import paddick_conformity

cfg = load_config()
r = cfg.results_root
store, doses = MaskStore(cfg), DoseStore(cfg)

CONSENSUS_STRUCTURES = ["CTV", "Rectum", "Bladder"]

rows = []
for patient in cfg.patients:
    grid = load_grid(cfg, patient)
    vox = grid.voxel_volume_cc
    observers = sorted(
        s for s in cfg.observers
        if s not in ("VOTE", "STAPLE") and store.has(patient, s, "CTV")
    )
    for i in observers:
        dose, covered = doses.get(patient, i)
        treated = store.get(patient, i, "Treated_Volume")

        for label, cset in (("VOTE", "VOTE"), ("LOO", f"LOO_{i}")):
            if not store.has(patient, cset, "CTV"):
                print(f"  missing {patient}/{cset}")
                continue
            for structure in CONSENSUS_STRUCTURES:
                if not store.has(patient, cset, structure):
                    continue
                specs = target_specs(cfg) if structure == "CTV" else oar_specs(cfg)[structure]
                d = structure_dose(dose, covered, store.get(patient, cset, structure))
                for spec in specs:
                    rows.append({"patient": patient, "plan_set": i, "truth_kind": label,
                                 "endpoint": spec["id"], "value": evaluate(spec, d, vox)})
            rows.append({"patient": patient, "plan_set": i, "truth_kind": label,
                         "endpoint": "paddick_ci",
                         "value": paddick_conformity(treated, store.get(patient, cset, "CTV"))})
        doses.drop_patient(patient)
    store.drop_patient(patient)

cons = pd.DataFrame(rows)
cons.to_parquet(r / "consensus_scored_cells.parquet", index=False)
print(f"computed {len(cons)} consensus-scored cells")
print(cons.groupby("truth_kind").size().to_string())
