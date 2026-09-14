# Interobserver contouring variation in prostate VMAT: cross-evaluation code

Code and analysis protocol for the study *Dosimetric and protocol-compliance
consequences of interobserver contouring variation in prostate VMAT, and the failure of
geometric agreement metrics to predict them* (manuscript in preparation).

Ten observers contoured the prostate CTV, rectum and bladder on five planning CTs from
the TROG 03.04 RADAR trial. One VMAT plan was optimised on each observer's contours and
on a majority-vote consensus contour, and every plan was evaluated on every observer's
contours, separating the effect of the contour a dose is assessed on from the effect of
the contour a plan is optimised on.

## Contents

```
config/      dataset, structures, protocol constraints, models and waivers
src/mcx/     the package: io, geom, dose, endpoints, metrics, qc, stats
pipelines/   numbered stages, each writing one table to results/
analysis/    scripts that read results/ and produce the figures
docs/        PROTOCOL (pre-registered analysis plan and amendments), DECISIONS, DATA_ISSUES
tests/       unit tests
```

## Data

The data are patient data from the TROG 03.04 RADAR trial and its companion contouring
studies. They are private and are not included in this repository, and neither are any
results tables, figures or caches derived from them. Observers are identified in the
code and documentation only as O1-O10; `VOTE` is the consensus contour.

## Running

```bash
python -m pip install -e ".[dev]"
```

```bash
python -m pytest
```

The tests run without the data. The pipeline (`pipelines/00_index_dicom.py` onwards) and
the analysis scripts need the planning CTs, contours and dose distributions in `data/`,
and a `config/observer_key.local.yaml` mapping O1-O10 to the names used in the DICOM
(see `config/observer_key.example.yaml`). That file is never committed.

## Analysis protocol

`docs/PROTOCOL.md` was fixed before any hypothesis-relevant result was computed, and
later changes are recorded as amendments in the same file and in `docs/DECISIONS.md`.
The version history of the working repository cannot be published because it contains
observer identities; the protocol was frozen at the tag `protocol-v1` in that repository
(commit 5adbb9ce25f4, 2026-09-07). This repository starts from a single snapshot.
References to `paper/` and `deliverables/` in the documentation are to parts of the
working repository not included here.
