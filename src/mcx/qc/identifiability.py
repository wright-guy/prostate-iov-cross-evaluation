"""Verifying that each dose really belongs to the observer its folder claims.

PatientName is '<initials>_Atlas_<patient>_CT' on every RTDOSE and RTPLAN file, so the folder
name is the only thing in the dataset linking a dose to a contour set. That link has to
be established from the dosimetry itself or every downstream result rests on a filename.

The test is column-oriented: fix the evaluation contour j, vary the plan i, and ask
whether the plan optimised on j covers j best. Orienting it the other way -- fix the
plan, vary the contour -- does not work, because D98 is confounded by structure size:
the observer with the smallest PTV scores highest under every plan, so the row maximum
identifies the smallest contour rather than the matching plan.

No threshold is chosen by inspection. Under the null hypothesis that plan labels are
unrelated to contour labels, the diagonal is a random draw from its own column, so the
mean within-column z-score of the diagonal is centred on zero. A permutation test over
plan-label assignments turns that into a p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_PERMUTATIONS = 20000
PERMUTATION_SEED = 20260904


@dataclass(frozen=True)
class IdentifiabilityResult:
    patient: str
    n_sets: int
    statistic: float          # mean within-column z-score of the diagonal
    p_value: float
    n_columns_won: int        # columns where the diagonal is rank 1
    worst_margin_gy: float    # largest (column max - diagonal) over columns
    per_column: pd.DataFrame

    @property
    def fraction_won(self) -> float:
        return self.n_columns_won / self.n_sets if self.n_sets else float("nan")


def _column_z(matrix: np.ndarray) -> np.ndarray:
    """Z-score each column of an (n_plan, n_eval) matrix."""
    mu = matrix.mean(axis=0, keepdims=True)
    sd = matrix.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 0, sd, np.nan)
    return (matrix - mu) / sd


def diagonal_statistic(matrix: np.ndarray) -> float:
    """Mean within-column z-score of the diagonal. Zero under the null."""
    z = _column_z(matrix)
    return float(np.nanmean(np.diag(z)))


def permutation_test(
    matrix: np.ndarray,
    *,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> tuple[float, float]:
    """Return ``(statistic, p_value)`` for the observed plan->contour labelling.

    The null shuffles which plan row is treated as the diagonal for each column,
    i.e. it asks how often a random relabelling of the plans would put the diagonal
    as high in its column as the observed labelling does.
    """
    n = matrix.shape[0]
    observed = diagonal_statistic(matrix)
    z = _column_z(matrix)
    rng = np.random.default_rng(seed)

    null = np.empty(n_permutations)
    cols = np.arange(n)
    for k in range(n_permutations):
        perm = rng.permutation(n)
        null[k] = float(np.nanmean(z[perm, cols]))

    # Add-one correction: a p-value from B permutations can never be exactly zero.
    p = float((np.count_nonzero(null >= observed) + 1) / (n_permutations + 1))
    return observed, p


def assess(
    values: pd.DataFrame,
    *,
    patient: str,
    metric: str = "D98",
    n_permutations: int = N_PERMUTATIONS,
) -> IdentifiabilityResult:
    """Assess one patient from a long frame of (plan_set, eval_set, metric) values."""
    wide = values.pivot(index="plan_set", columns="eval_set", values=metric)
    sets = [s for s in wide.index if s in wide.columns]
    wide = wide.loc[sets, sets]
    matrix = wide.to_numpy(dtype=float)

    statistic, p_value = permutation_test(matrix, n_permutations=n_permutations)

    z = _column_z(matrix)
    rows = []
    for k, eval_set in enumerate(sets):
        column = matrix[:, k]
        order = np.argsort(-column)
        rank = int(np.where(order == k)[0][0]) + 1
        rows.append(
            {
                "patient": patient,
                "eval_set": eval_set,
                "diagonal_value": float(column[k]),
                "column_max": float(column.max()),
                "column_argmax": sets[int(order[0])],
                "margin_gy": float(column.max() - column[k]),
                "diagonal_rank": rank,
                "diagonal_z": float(z[k, k]),
            }
        )
    per_column = pd.DataFrame(rows)

    return IdentifiabilityResult(
        patient=patient,
        n_sets=len(sets),
        statistic=statistic,
        p_value=p_value,
        n_columns_won=int((per_column["diagonal_rank"] == 1).sum()),
        worst_margin_gy=float(per_column["margin_gy"].max()),
        per_column=per_column,
    )
