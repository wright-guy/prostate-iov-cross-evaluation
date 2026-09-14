# Staging: Section 2 scripts not yet folded into the numbered pipeline

These produced results that the paper and stage 14 depend on. They were written
during drafting and are kept here verbatim so the numbers stay reproducible. Run in
this order, after stage 13:

| Script | Writes |
|---|---|
| `s21_vote_self_evaluation.py` | `results/vote_self_evaluation.parquet` |
| `s22a_consensus_scored_cells.py` | `results/consensus_scored_cells.parquet` (E(i,VOTE), E(i,LOO_i)) |
| `s22b_per_plan.py` | `results/sec22_per_plan.parquet` |
| `s22c_row_iov.py` | `results/sec22_row_iov.parquet` |
| `s23_columns.py` | `results/sec23_col_iov.parquet`, `results/sec23_per_column.parquet` |

Then `pipelines/14_patient_level_inference.py`.

They use hard-coded paths and `print` output rather than the provenance writer, and
should become numbered stages.
