"""Stage 13 — assemble the results dossier.

    python pipelines/13_build_dossier.py

Emits three artefacts from one source of truth:
  results/quotable.csv          every citable number, keyed by a stable identifier
  deliverables/dossier.json     the section structure, for the two renderers
  deliverables/tables/*.csv     each table as it should appear in the paper

The .docx and the HTML page are rendered from dossier.json, so they cannot disagree
with each other or with the pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import REPO_ROOT, load_config  # noqa: E402
from mcx.dossier import build  # noqa: E402
from mcx.provenance import RunInfo, git_sha  # noqa: E402


def main() -> int:
    cfg = load_config()
    run = RunInfo.create("13_build_dossier", cfg)
    out = REPO_ROOT / "deliverables"
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    sections, quotable = build(cfg)

    quotable.to_csv(cfg.results_root / "quotable.csv", index=False)
    print(f"{len(quotable)} quotable numbers -> results/quotable.csv")

    payload = {
        "title": "Does consensus contouring mitigate the dosimetric impact of "
                 "interobserver variation?",
        "subtitle": "Results dossier — a cross-evaluation study in prostate VMAT",
        "generated_utc": run.started_utc,
        "git_sha": git_sha(),
        "config_hash": cfg.config_hash,
        "sections": [],
    }

    n_tables = 0
    for section in sections:
        entry = {"title": section.title, "lead": section.lead, "blocks": []}
        for block in section.blocks:
            b = {
                "heading": block.heading,
                "statement": block.statement,
                "interpretation": block.interpretation,
                "source": block.source,
                "caveat": block.caveat,
                "figure": block.figure,
                "figure_caption": block.figure_caption,
                "table_caption": block.table_caption,
                "table": None,
                "table_file": None,
            }
            if block.table is not None and not block.table.empty:
                n_tables += 1
                name = f"T{n_tables:02d}_{block.heading[:36]}"
                name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
                path = tables / f"{name}.csv"
                block.table.to_csv(path, index=False)
                b["table"] = {
                    "columns": [str(c) for c in block.table.columns],
                    "rows": block.table.astype(object).where(
                        pd.notna(block.table), None).values.tolist(),
                }
                b["table_file"] = path.name
            entry["blocks"].append(b)
        payload["sections"].append(entry)

    (out / "dossier.json").write_text(json.dumps(payload, indent=2, default=str),
                                      encoding="utf-8")
    print(f"{len(sections)} sections, {sum(len(s['blocks']) for s in payload['sections'])} "
          f"blocks, {n_tables} tables -> deliverables/dossier.json")
    print(f"tables also written individually to {tables.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
