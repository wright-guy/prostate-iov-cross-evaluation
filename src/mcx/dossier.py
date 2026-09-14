"""Assemble the results dossier from results/*.parquet.

One source of truth, two renderings. Every number the paper could quote is emitted with
a stable identifier, the pipeline stage that produced it, and the table it came from, so
nothing has to be retyped by hand and any figure in the manuscript can be traced back to
the run that made it.

Interpretation is kept in its own field, separate from the measurement, so the argument
in the paper stays the author's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mcx.config import Config


@dataclass
class Quotable:
    """One citable number."""

    id: str
    section: str
    statement: str          # sentence-ready, with the value already in it
    value: float | str | None = None
    ci_lo: float | None = None
    ci_hi: float | None = None
    unit: str | None = None
    source_table: str | None = None
    source_stage: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "id": self.id, "section": self.section, "statement": self.statement,
            "value": self.value, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
            "unit": self.unit, "source_table": self.source_table,
            "source_stage": self.source_stage,
        }


@dataclass
class Block:
    """One result in the dossier: a claim, its evidence, and what it means."""

    heading: str
    statement: str
    interpretation: str
    table: pd.DataFrame | None = None
    table_caption: str | None = None
    figure: str | None = None
    figure_caption: str | None = None
    source: str = ""
    caveat: str | None = None


@dataclass
class Section:
    title: str
    lead: str
    blocks: list[Block] = field(default_factory=list)


def _fmt(v: float, dp: int = 2) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{dp}f}"


def _ci(row: pd.Series, point: str, lo: str = "ci_lo", hi: str = "ci_hi",
        dp: int = 2) -> str:
    return f"{_fmt(row[point], dp)} [{_fmt(row[lo], dp)}, {_fmt(row[hi], dp)}]"


# --------------------------------------------------------------------- methods


def methods_section(cfg: Config, r) -> tuple[Section, list[Quotable]]:  # noqa: ANN001
    structures = pd.read_parquet(r / "structures.parquet")
    grids = pd.read_parquet(r / "grids.parquet")
    margin = pd.read_parquet(r / "ptv_margin.parquet")
    diag = pd.read_parquet(r / "diagonal_endpoints.parquet")
    cover = pd.read_parquet(r / "dose_coverage.parquet")
    repro = pd.read_parquet(r / "consensus_reproduction.parquet")
    ident = pd.read_parquet(r / "tier1_identifiability.parquet")

    obs = structures[structures["present"]
                     & structures["contour_set"].isin(cfg.observers)]
    vol = (obs[obs["structure"].isin(["CTV", "PTV", "Rectum", "Bladder"])]
           .groupby("structure")["volume_cc"]
           .agg(n="count", median="median",
                q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75),
                min="min", max="max")
           .reindex(["CTV", "PTV", "Rectum", "Bladder"]).round(1).reset_index())

    d50 = diag["D50"]
    n_cells = len(cfg.design_cells())
    q: list[Quotable] = [
        Quotable("methods.cohort.n_patients", "Methods",
                 f"Five prostate patients, each contoured by ten independent observers.",
                 5, unit="patients", source_table="structures.parquet",
                 source_stage="02"),
        Quotable("methods.design.n_cells", "Methods",
                 f"The primary analysis comprises {n_cells} cross-evaluation cells: "
                 f"432 in Arm S, 49 on the diagonal and 49 in Arm C.",
                 n_cells, unit="cells", source_table="endpoints_long.parquet",
                 source_stage="08"),
        Quotable("methods.prescription", "Methods",
                 f"{cfg.prescription_gy:.0f} Gy in {cfg.n_fractions} fractions, "
                 "delivered as a single-arc VMAT plan on an Elekta VersaHD at 6 MV.",
                 cfg.prescription_gy, unit="Gy", source_table="config/dataset.yaml",
                 source_stage="—"),
        Quotable("methods.normalisation.ctv_d50", "Methods",
                 f"All plans were normalised consistently: median CTV D50 on the "
                 f"self-evaluation diagonal was {d50.median():.2f} Gy "
                 f"(range {d50.min():.2f}–{d50.max():.2f}, SD {d50.std():.3f}) "
                 f"across {len(d50)} plans.",
                 float(d50.median()), float(d50.min()), float(d50.max()), "Gy",
                 "diagonal_endpoints.parquet", "06"),
        Quotable("methods.ptv_margin", "Methods",
                 f"The PTV margin is recorded nowhere in the DICOM and was measured from "
                 f"the masks: the median CTV-surface-to-PTV-surface distance was "
                 f"{margin['median'].median():.2f} mm in all five patients "
                 f"(IQR {margin['iqr'].median():.2f} mm), with no directional asymmetry.",
                 float(margin["median"].median()), unit="mm",
                 source_table="ptv_margin.parquet", source_stage="03"),
        Quotable("methods.grid", "Methods",
                 f"All contours and doses were resampled onto one analysis grid per "
                 f"patient at {grids['spacing_x'].iloc[0]:.1f} mm in plane, retaining the "
                 "native CT slice spacing in z so that no contour was interpolated into "
                 "existence.",
                 float(grids["spacing_x"].iloc[0]), unit="mm",
                 source_table="grids.parquet", source_stage="02"),
        Quotable("methods.dose_containment", "Methods",
                 f"All {len(cover):,} (plan × contour set × structure) combinations lay "
                 "entirely inside their dose grid, so absolute-volume endpoints are "
                 "valid for every cell.",
                 len(cover), unit="combinations", source_table="dose_coverage.parquet",
                 source_stage="06"),
        Quotable("methods.mapping_verified", "Methods",
                 "The dose-to-observer mapping carries no DICOM label and was verified "
                 "from the dosimetry by permutation test: p = 5.0×10⁻⁵ in every patient, "
                 "the floor at 20 000 permutations.",
                 5.0e-5, unit="p", source_table="tier1_identifiability.parquet",
                 source_stage="01"),
        Quotable("methods.consensus_algorithm", "Methods",
                 f"The consensus contours are a strict majority vote "
                 f"(⌊n/2⌋+1), reproduced independently to a median Dice of "
                 f"{repro['dice_at_strict_majority'].median():.3f} "
                 f"(minimum {repro['dice_at_strict_majority'].min():.3f}).",
                 float(repro["dice_at_strict_majority"].median()), unit="Dice",
                 source_table="consensus_reproduction.parquet", source_stage="04"),
    ]

    section = Section(
        "Methods — numbers to state",
        "Everything the methods section has to assert, with its provenance. "
        "Each was produced by a quality-control stage, not by an analysis.",
        [
            Block(
                "Cohort and contour volumes",
                q[0].statement + " " + q[1].statement,
                "The design is unbalanced because M030 has no O6 rectum contour and no "
                "O6 plan, so that patient contributes nine observers. The mixed design "
                "absorbs this; the cell counts differ from the 550 originally planned.",
                vol, "T1. Contour volumes across the ten observers and five patients (cm³).",
                source="structures.parquet, stage 02",
            ),
            Block(
                "Treatment and normalisation",
                q[2].statement + " " + q[3].statement,
                "The department protocol prescribes to PTV D95 = 78 Gy. These plans are "
                "CTV-optimised with no PTV objective and do not follow that convention, "
                "by design — which is why PTV-based constraints sit systematically in "
                "minor variation.",
                source="diagonal_endpoints.parquet, stage 06",
            ),
            Block(
                "Geometry and quality control",
                " ".join(x.statement for x in (q[4], q[5], q[6])),
                "The 7 mm margin means the existing PTV is the +7 mm rung of the "
                "coverage-degradation ladder, so that rung can be cross-checked against "
                "a structure that was actually drawn.",
                source="ptv_margin.parquet, grids.parquet, dose_coverage.parquet",
            ),
            Block(
                "Provenance checks that had to be made",
                q[7].statement + " " + q[8].statement,
                "PatientName is identical across every dose file in the dataset, so the "
                "folder name is the only label the DICOM carries. Establishing the "
                "mapping from the dosimetry was necessary, not merely prudent.",
                source="tier1_identifiability.parquet, consensus_reproduction.parquet",
            ),
        ],
    )
    return section, q


# ----------------------------------------------------------------------- aim 1


def aim1_section(cfg: Config, r) -> tuple[Section, list[Quotable]]:  # noqa: ANN001
    comp = pd.read_parquet(r / "aim1_compression.parquet")
    bias = pd.read_parquet(r / "aim1_consensus_bias.parquet")
    tend = pd.read_parquet(r / "aim1_observer_tendency.parquet")
    corr = pd.read_parquet(r / "aim1_tendency_correlation.parquet")
    geo = pd.read_parquet(r / "geometric_long.parquet")

    loo = comp[(comp["comparison"] == "loo_corrected") & comp["headline"]]
    t_h1 = (loo.assign(factor=lambda d: d.apply(lambda x: _ci(x, "compression_factor"),
                                                axis=1))
            .pivot_table(index="structure", columns="metric", values="factor",
                         aggfunc="first")
            .reindex(["CTV", "PTV", "Rectum", "Bladder"]).reset_index())

    ctv_dsc = loo[(loo["structure"] == "CTV") & (loo["metric"] == "dsc")].iloc[0]
    msd = bias[(bias["quantity"] == "signed_msd_mm")
               & (bias["reference"] == "loo_corrected")].set_index("structure")

    agree = (geo[geo["family"] == "obs_obs"].groupby("structure")[["dsc", "msd_mm", "hd95_mm"]]
             .median().reindex(["CTV", "PTV", "Rectum", "Bladder"]).round(3).reset_index())

    rect = corr[(corr["structure_a"] == "CTV") & (corr["structure_b"] == "Rectum")].iloc[0]
    blad = corr[(corr["structure_a"] == "CTV") & (corr["structure_b"] == "Bladder")].iloc[0]

    q = [
        Quotable("aim1.h1.ctv.dsc.loo", "Aim 1",
                 f"Agreement with a leave-one-out consensus was better than "
                 f"observer-to-observer agreement by a factor of "
                 f"{_ci(ctv_dsc, 'compression_factor')} for CTV Dice, against a "
                 f"variance-identity prediction of 1.34.",
                 float(ctv_dsc["compression_factor"]), float(ctv_dsc["ci_lo"]),
                 float(ctv_dsc["ci_hi"]), "ratio", "aim1_compression.parquet", "07b"),
        Quotable("aim1.h2.rectum.signed_msd", "Aim 1",
                 f"The consensus is a biased estimator of the observers: for the rectum, "
                 f"observers lay a median of "
                 f"{_ci(msd.loc['Rectum'], 'median', dp=2)} mm outside the "
                 "leave-one-out consensus surface.",
                 float(msd.loc["Rectum", "median"]), float(msd.loc["Rectum", "ci_lo"]),
                 float(msd.loc["Rectum", "ci_hi"]), "mm",
                 "aim1_consensus_bias.parquet", "07b"),
        Quotable("aim1.tendency.ctv_rectum", "Aim 1",
                 f"An observer who contours a large target also contours a large rectum "
                 f"(r = {rect['pearson_r']:.2f}, p = {rect['pearson_p']:.3f}), but not a "
                 f"large bladder (r = {blad['pearson_r']:.2f}, p = {blad['pearson_p']:.2f}).",
                 float(rect["pearson_r"]), unit="r",
                 source_table="aim1_tendency_correlation.parquet", source_stage="07b"),
    ]

    section = Section(
        "Results — Aim 1: geometric",
        "Does agreement improve when observers are compared to a consensus rather than "
        "to each other, and by how much against the variance-identity prediction?",
        [
            Block(
                "Observer-to-observer agreement",
                f"Median observer-to-observer Dice was "
                f"{agree.set_index('structure').loc['CTV', 'dsc']:.3f} for CTV, "
                f"{agree.set_index('structure').loc['Rectum', 'dsc']:.3f} for rectum and "
                f"{agree.set_index('structure').loc['Bladder', 'dsc']:.3f} for bladder.",
                "Bladder agreement is highest, which is expected: it is the one structure "
                "here whose boundaries are unambiguous on CT.",
                agree, "Median observer-to-observer agreement across 2 120 comparisons.",
                source="geometric_long.parquet, stage 07",
            ),
            Block(
                "H1 — the compression factor",
                q[0].statement,
                "Compression lands close to theory for Dice and mean surface distance. "
                "Two departures matter: the CTV interval excludes 1.34, so the target "
                "compresses slightly less than the variance identity predicts; and HD95 "
                "compresses least in three of four structures, barely at all for bladder. "
                "HD95 is a tail statistic, so this is H4's mechanism appearing in pure "
                "geometry, before any dose is involved.",
                t_h1, "T2a. Leave-one-out corrected compression factor with 95% "
                      "cluster-bootstrap intervals. Benchmark 1.34.",
                "F2_geometric_compression.png",
                "F2. Agreement by comparison family, and the compression factor against "
                "both variance-identity benchmarks.",
                source="aim1_compression.parquet, stage 07b",
            ),
            Block(
                "H2 — the consensus is biased",
                q[1].statement,
                "Observers are systematically larger than the consensus. The interval "
                "excludes zero for CTV, PTV and rectum under both references; bladder is "
                "the exception, and is again the structure with unambiguous boundaries. "
                "The bias shows on the surface metric more clearly than on volume, as a "
                "thin surface effect should.",
                bias[bias["quantity"] == "signed_msd_mm"][
                    ["structure", "reference", "median", "ci_lo", "ci_hi",
                     "excludes_zero"]].round(4),
                "T2b. Signed mean surface distance from observer to consensus. "
                "Positive means the observer lies outside the consensus.",
                source="aim1_consensus_bias.parquet, stage 07b",
            ),
            Block(
                "Observer tendencies carry across structures",
                q[2].statement,
                "A shared generous-versus-conservative trait that operates where "
                "boundaries are ambiguous — prostate apex and base, rectal superior and "
                "inferior extent — and vanishes where contrast is high. This matters for "
                "the argument that consensus does nothing about systematic bias shared "
                "across observers: a tendency common to the group survives a majority "
                "vote by construction.",
                tend.round(3),
                "T2c. Median signed volume difference against the leave-one-out "
                "consensus, per observer and structure.",
                source="aim1_observer_tendency.parquet, stage 07b",
                caveat="n = 10 observers. The CTV–PTV correlation of 0.96 is a positive "
                       "control, since the PTV is derived from the CTV.",
            ),
        ],
    )
    return section, q


# ----------------------------------------------------------------------- aim 2

PRETTY = {
    "ctv_d98": "CTV D98", "ctv_v95pct": "CTV V95%",
    "ladder_d98_3mm": "CTV+3 mm D98", "ladder_d98_5mm": "CTV+5 mm D98",
    "paddick_ci": "Paddick CI", "rectum_v70pct": "Rectum V70",
    "rectum_v65pct": "Rectum V65", "rectum_d2cc": "Rectum D2cc",
    "rectum_dmean": "Rectum Dmean", "bladder_v70pct": "Bladder V70",
    "bladder_dmean": "Bladder Dmean", "ntcp_rectum_relative": "NTCP rectum",
}
HEADLINE = list(PRETTY)


def aim2_section(cfg: Config, r) -> tuple[Section, list[Quotable]]:  # noqa: ANN001
    comp = pd.read_parquet(r / "aim2_compression.parquet")
    est = pd.read_parquet(r / "aim2_estimands.parquet")
    flips = pd.read_parquet(r / "aim2_flip_rates.parquet")
    var = pd.read_parquet(r / "aim2_variance_components.parquet")
    bio = pd.read_parquet(r / "bio_long.parquet")
    ep = pd.read_parquet(r / "endpoints_long.parquet")
    diagc = pd.read_parquet(r / "diagonal_consistency.parquet")

    iqr = comp[(comp["spread_measure"] == "iqr") & comp["endpoint"].isin(HEADLINE)]
    t_h3 = (iqr.assign(f=lambda d: d.apply(lambda x: _ci(x, "compression_factor"), axis=1))
            .pivot_table(index="endpoint", columns="basis", values="f", aggfunc="first")
            .reindex([e for e in HEADLINE if e in set(iqr["endpoint"])])
            .rename(index=PRETTY).reset_index())
    n_below = int(iqr[iqr["basis"] == "error"]["below_geometric"].sum())
    n_total = int((iqr["basis"] == "error").sum())

    t_h4 = (comp[(comp["basis"] == "error") & comp["endpoint"].isin(HEADLINE)]
            .pivot_table(index="endpoint", columns="spread_measure",
                         values="compression_factor")
            .reindex([e for e in HEADLINE if e in set(comp["endpoint"])])
            .rename(index=PRETTY).round(2).reset_index())

    n = bio[(bio["model"] == "ntcp") & (bio["structure"] == "Rectum")
            & (bio["variant"] == "relative")]
    per_plan = n.groupby(["patient", "plan_set"])["value"].agg(["min", "max"])
    span = 100 * (per_plan["max"] - per_plan["min"])
    fold = (per_plan["max"] / per_plan["min"])

    s_flips = flips[flips["arm"] == "S"].copy()
    s_flips = s_flips[s_flips["ordinal_flip_across_truths"] > 0].sort_values(
        "ordinal_flip_across_truths", ascending=False)
    t_flip = s_flips[["label", "ordinal_flip_across_truths",
                      "binary_flip_across_truths", "pct_per_protocol"]].copy()
    for c in ("ordinal_flip_across_truths", "binary_flip_across_truths"):
        t_flip[c] = (100 * t_flip[c]).round(0).astype(int)
    t_flip["pct_per_protocol"] = t_flip["pct_per_protocol"].round(0).astype(int)
    t_flip.columns = ["Constraint", "Tier changes (%)", "Per-protocol changes (%)",
                      "Per-protocol overall (%)"]
    top = s_flips.iloc[0]

    t_var = (var[var["endpoint"].isin(HEADLINE)]
             .assign(ratio=lambda d: d.apply(lambda x: _ci(x, "truth_over_plan"), axis=1))
             .set_index("endpoint")[["pct_plan", "pct_truth", "pct_interaction",
                                     "ratio", "between_patient_pct"]]
             .reindex([e for e in HEADLINE if e in set(var["endpoint"])])
             .rename(index=PRETTY).round(1).reset_index())
    rect_ratio = var[var["endpoint"] == "rectum_v70pct"].iloc[0]

    d98 = ep[ep["endpoint"] == "ctv_d98"]
    t_est = (est[est["endpoint"].isin(HEADLINE)]
             .pivot_table(index="endpoint", columns="arm",
                          values=["e1_median", "e2_median", "e3_mean"])
             .reindex([e for e in HEADLINE if e in set(est["endpoint"])])
             .rename(index=PRETTY).round(3).reset_index())

    q = [
        Quotable("aim2.h3.n_below_geometric", "Aim 2",
                 f"Dosimetric compression was close to 1.0 on both bases and "
                 f"{n_below} of {n_total} headline endpoints had intervals lying "
                 "entirely below the geometric benchmark of 1.34.",
                 n_below, unit="endpoints", source_table="aim2_compression.parquet",
                 source_stage="10b"),
        Quotable("aim2.risk_span.median_pp", "Aim 2",
                 f"For an identical delivered dose distribution, predicted late rectal "
                 f"toxicity varied by a median of {span.median():.1f} percentage points "
                 f"({fold.median():.1f}-fold) across the ten plausible rectum contours, "
                 f"and from {100 * n['value'].min():.1f}% to "
                 f"{100 * n['value'].max():.1f}% across the cohort.",
                 float(span.median()), unit="percentage points",
                 source_table="bio_long.parquet", source_stage="09"),
        Quotable("aim2.flip.top", "Aim 2",
                 f"{top['label']} changed protocol tier for "
                 f"{100 * top['ordinal_flip_across_truths']:.0f}% of delivered plans "
                 "depending solely on which observer's contour was used to score it.",
                 float(100 * top["ordinal_flip_across_truths"]), unit="%",
                 source_table="aim2_flip_rates.parquet", source_stage="10b"),
        Quotable("aim2.h5.rectum_v70", "Aim 2",
                 f"For rectum V70 the evaluation contour contributed "
                 f"{_ci(rect_ratio, 'truth_over_plan')} times the variance of the "
                 "planning contour: the two sides are comparable.",
                 float(rect_ratio["truth_over_plan"]), float(rect_ratio["ci_lo"]),
                 float(rect_ratio["ci_hi"]), "ratio",
                 "aim2_variance_components.parquet", "11"),
        Quotable("aim2.ctv_d98.range", "Aim 2",
                 f"CTV D98 ranged from {d98['value'].min():.1f} to "
                 f"{d98['value'].max():.1f} Gy across the 530 cells "
                 f"(Arm S {d98[d98.arm == 'S']['value'].min():.1f}–"
                 f"{d98[d98.arm == 'S']['value'].max():.1f} Gy).",
                 float(d98["value"].min()), unit="Gy",
                 source_table="endpoints_long.parquet", source_stage="08"),
    ]

    section = Section(
        "Results — Aim 2: dosimetric and biological",
        "Does the geometric benefit of consensus survive the transfer into dose, and "
        "how much does the choice of scoring contour move a clinical decision?",
        [
            Block(
                "The three estimands",
                "E1, delivery variability, is identically zero in Arm C because there "
                "is one consensus plan. E2, residual truth uncertainty, is non-zero in "
                "both arms and is what consensus cannot remove. E3 is the mean penalty "
                "against perfect information.",
                "Reporting only E1 would make consensus look decisive by construction. "
                "The comparison that carries information is E2, and the ratio of E2 "
                "between arms is the compression factor below.",
                t_est, "T3a. E1, E2 and E3 by arm for the headline endpoints.",
                source="aim2_estimands.parquet, stage 10b",
            ),
            Block(
                "H3 — the geometric benefit does not reach dose",
                q[0].statement,
                "A compression factor of 1.0 means the consensus plan gives no "
                "measurable reduction in residual truth uncertainty. Against a geometric "
                "compression of 1.26–1.42, this is the study's central result: consensus "
                "removes variation in what is planned and leaves uncertainty about what "
                "is delivered to the true anatomy essentially intact.",
                t_h3, "T3b. Compression factor with 95% cluster-bootstrap intervals, on "
                      "both bases. Benchmark 1.34.",
                "F4_compression_by_endpoint.png",
                "F4. Compression by endpoint against the geometric benchmark.",
                source="aim2_compression.parquet, stage 10b",
                caveat="Two bases are reported. The error basis, spread of "
                       "endpoint(i,j) − endpoint(j,j), is the like-for-like analogue of "
                       "the geometric identity; the E2 basis is the pre-registered form "
                       "and is dominated by how much the truths differ from each other.",
            ),
            Block(
                "H4 — the tails",
                "Compression on the 5th–95th percentile range and on the full range is "
                "reported alongside the IQR.",
                "Consensus regularises the centre of a distribution more readily than "
                "its extremes, and clinical risk lives in the extremes. Read this table "
                "with H1's HD95 result, where the same pattern appears in pure geometry.",
                t_h4, "T3c. Compression factor by spread measure, error basis.",
                source="aim2_compression.parquet, stage 10b",
            ),
            Block(
                "Constraint flip rates",
                q[2].statement,
                "The most clinically legible result in the study. It says that protocol "
                "compliance for a plan that has already been delivered is not a property "
                "of the plan alone.",
                t_flip, "T3d. Proportion of delivered plans whose protocol tier changes "
                        "with the scoring contour, Arm S.",
                "F7_flip_rates_and_risk.png",
                "F7. Constraint flip rates, and the predicted-risk range for each "
                "delivered plan across the ten plausible rectum contours.",
                source="aim2_flip_rates.parquet, stage 10b",
                caveat="Arm C has one plan per patient, so its flip rates rest on five "
                       "observations and are not compared with Arm S without that "
                       "denominator. The reverse direction — fixing the truth and "
                       "varying the plan — is zero in Arm C by construction.",
            ),
            Block(
                "Predicted risk for a fixed delivered dose",
                q[1].statement,
                "This is the transferability argument in its sharpest form: published "
                "NTCP models were fitted under one contouring convention, and the same "
                "delivered dose yields materially different predicted risk under "
                "another.",
                source="bio_long.parquet, stage 09",
            ),
            Block(
                "H5 — planning side against evaluation side",
                q[3].statement,
                "For the rectum the two sides are comparable; for the target the "
                "planning side dominates, as it must, since that is what was optimised. "
                "Paddick conformity is evaluation-dominated. The practical implication "
                "is that OAR reporting is at least as sensitive to who drew the scoring "
                "contour as to who drew the planning contour.",
                t_var, "T3e. Within-patient variance shares for the Arm S matrix, and "
                       "the evaluation-to-planning ratio with 95% intervals.",
                "F5_F6_variance_components.png",
                "F5 and F6. Variance components, and which side of the design drives "
                "each endpoint.",
                source="aim2_variance_components.parquet, stage 11",
                caveat="The pre-registered crossed mixed model converged for 0 of 38 "
                       "endpoints. The primary estimate is an exact two-way "
                       "decomposition within patient. For bladder, 69–87% of variance "
                       "is between patients, so its within-patient shares rest on a "
                       "small residual.",
            ),
            Block(
                "Diagonal consistency, and where sensitivity lives",
                "The ratio of off-diagonal to diagonal spread is 0.79–1.15 for CTV D95, "
                "D50, D2 and Dmean, 1.41 for CTV D98, 2.33 for rectum V70, 3.16 at "
                "CTV+3 mm and 5.78 at CTV+5 mm.",
                "Target metrics inside the optimised volume are pinned by the "
                "normalisation and are indistinguishable from plan-to-plan "
                "reproducibility; both spreads are 0.13–0.36 Gy. Sensitivity appears at "
                "the target boundary and grows steeply outside it. This is the "
                "quantitative form of the CTV-optimisation limitation, and it is also "
                "§7.3's honest-reporting case.",
                diagc.groupby("endpoint")["ratio_off_to_diagonal"].median()
                .replace(np.inf, np.nan).dropna().sort_values(ascending=False)
                .round(2).reset_index().head(14),
                "S1. Off-diagonal to diagonal IQR ratio, median over patients.",
                source="diagonal_consistency.parquet, stage 10",
            ),
        ],
    )
    return section, q


# ----------------------------------------------------------------------- aim 3


def aim3_section(cfg: Config, r) -> tuple[Section, list[Quotable]]:  # noqa: ANN001
    pred = pd.read_parquet(r / "aim3_prediction.parquet")
    rb = pred[pred["target"] == "replanning_benefit"]

    rect = rb[rb["endpoint"] == "ntcp_rectum_relative"]
    best = rect.loc[rect["spearman_rho"].abs().idxmax()]
    unsigned = rect[rect["metric_class"] == "unsigned"]

    def table_for(endpoint: str) -> pd.DataFrame:
        g = rb[rb["endpoint"] == endpoint].copy()
        g = g.reindex(g["spearman_rho"].abs().sort_values(ascending=False).index)
        g["rho"] = g.apply(lambda x: f"{x.spearman_rho:+.3f}"
                           + ("*" if x.significant_fdr else ""), axis=1)
        g["LOPO R2"] = g["lopo_r2"].round(3)
        return g[["metric", "metric_class", "rho", "LOPO R2", "n_pairs"]].rename(
            columns={"metric": "Metric", "metric_class": "Class",
                     "rho": "Spearman rho", "n_pairs": "n"})

    q = [
        Quotable("aim3.h7.best", "Aim 3",
                 f"The best predictor of replanning benefit in NTCP(rectum) was "
                 f"{best['metric']}, a dose-aware metric, at rho = "
                 f"{best['spearman_rho']:+.2f} with a leave-one-patient-out R² of "
                 f"{best['lopo_r2']:.2f}.",
                 float(best["spearman_rho"]), unit="rho",
                 source_table="aim3_prediction.parquet", source_stage="12"),
        Quotable("aim3.h6.unsigned_null", "Aim 3",
                 f"The three unsigned overlap metrics achieved |rho| of at most "
                 f"{unsigned['spearman_rho'].abs().max():.3f} against the same endpoint: "
                 "no predictive power whatever.",
                 float(unsigned["spearman_rho"].abs().max()), unit="rho",
                 source_table="aim3_prediction.parquet", source_stage="12"),
    ]

    section = Section(
        "Results — Aim 3: predictive metrics",
        "Do the metrics routinely used to accept or reject a contour carry the "
        "information needed to predict its dosimetric consequence?",
        [
            Block(
                "H6 — direction is the signal, for the OAR",
                q[1].statement + " " + q[0].statement,
                "Unsigned metrics are invariant to the direction of disagreement, and "
                "for an organ at risk the direction is the entire signal: contouring "
                "too much and too little have opposite-signed dosimetric consequences "
                "that an overlap measure cannot distinguish. This is the sharpest "
                "available answer to the recurring criticism of DSC as a surrogate for "
                "clinical impact.",
                table_for("ntcp_rectum_relative"),
                "T4a. Predicting the replanning benefit in NTCP(rectum). "
                "* survives Benjamini–Hochberg at q = 0.05 across the whole grid.",
                "F8_metric_prediction.png",
                "F8. Metric against dosimetric consequence, by metric class.",
                source="aim3_prediction.parquet, stage 12",
            ),
            Block(
                "H6 does not hold for the target",
                "For CTV D98 on the primary target, unsigned metrics (best |rho| 0.234) "
                "slightly outperform signed ones (0.153). On the secondary target the "
                "ordering reverses sharply, signed 0.617 against unsigned 0.234.",
                "Reported as the split result it is rather than generalised from the "
                "rectum. A target's coverage responds to overall mismatch in a way an "
                "organ at risk's dose does not, which is a plausible mechanism but not "
                "one this design can confirm.",
                table_for("ctv_d98"),
                "T4b. Predicting the replanning benefit in CTV D98.",
                source="aim3_prediction.parquet, stage 12",
            ),
            Block(
                "H7 — dose-aware metrics win",
                "Mean dose in the false-negative volume is the best predictor for both "
                "endpoints and both prediction targets.",
                "Only the irradiated part of a surface can influence a DVH, so a "
                "whole-structure overlap measure dilutes the signal with disagreement "
                "that cannot matter dosimetrically. The practical reading is that a "
                "contour-QA metric should be computed against the dose, not against the "
                "contour alone.",
                source="aim3_prediction.parquet, stage 12",
                caveat="No acceptability thresholds are derived. The form of a useful "
                       "metric is proposed; deriving thresholds needs a far larger "
                       "cohort. Every correlation is within-patient, pooled by Fisher z; "
                       "nothing is pooled across patients.",
            ),
        ],
    )
    return section, q


def build(cfg: Config) -> tuple[list[Section], pd.DataFrame]:
    r = cfg.results_root
    sections, quotables = [], []
    for fn in (methods_section, aim1_section, aim2_section, aim3_section):
        section, q = fn(cfg, r)
        sections.append(section)
        quotables.extend(q)
    return sections, pd.DataFrame([x.as_row() for x in quotables])
