# Analysis protocol — pre-registration

**Study.** Does consensus contouring mitigate the dosimetric impact of interobserver
variation? A cross-evaluation study in prostate VMAT.

**Status.** Fixed before any Aim 1, 2 or 3 result was computed. The git tag
`protocol-v1` marks the commit at which this was frozen; any later change appears as a
subsequent commit and must be declared as a protocol amendment in the paper.

**What has already been run, and why that does not compromise this document.**
Pipeline stages 00–06 (data indexing, QC, mask building, dose resampling) were completed
before this protocol was written. They produce no hypothesis-relevant result: they
establish that the data are what they claim to be, and every number they produced is a
methods number, not a finding. The cross-evaluation itself — stages 07 onward — has not
been run. The one exception is declared in §8 below.

---

## 1. Design

Five prostate patients, each contoured independently by ten observers, plus a consensus
(VOTE) contour set. A VMAT plan was optimised on each contour set. Every plan is scored
against every plausible true anatomy.

- **i** — the contour set the plan was optimised on
- **j** — the contour set used as the assumed true anatomy

| Cell type | Definition | n | Role |
|---|---|---|---|
| Arm S | i ≠ j, both observers | 432 | single-observer status quo |
| Arm C | i = VOTE, j = observer | 49 | consensus mitigation |
| Diagonal | i = j, both observers | 49 | perfect-information bound; planning-consistency control |
| **Primary** | | **530** | |
| Sensitivity | j = VOTE | 54 | conventional but statistically leaky; reported separately |

STAPLE is excluded (plan-quality failure). M030/O6 does not exist — no rectum contour
and no plan — so M030 runs with nine observers and the design is unbalanced. Arm S
excludes the diagonal deliberately: including i = j would inject an artificial
perfect-knowledge case with probability 1/10 and flatter the status quo.

VOTE is excluded from the truth column in the primary analysis because it is a derived
quantity from the ten observers, not an independent plausible anatomy.

---

## 2. Hypotheses

Registered before analysis. Direction and rationale are stated so that a null result is
interpretable rather than merely disappointing.

| # | Hypothesis | Rationale |
|---|---|---|
| **H1** | Geometric agreement to VOTE is better than observer-to-observer agreement by a factor between 1.34 (leave-one-out corrected) and 1.49 (uncorrected) | Variance identity: SD(x_i − x_j) = √2·σ against SD(x_i − x̄) = σ√(1 − 1/N) |
| **H2** | VOTE is not an unbiased estimator of the observer mean surface | Majority vote is a median-like morphological operator, not a mean of signed distance fields |
| **H3** | The compression factor for dosimetric and NTCP endpoints is smaller than the geometric factor | Non-linear, saturating, asymmetric dose response; steep local gradients |
| **H4** | Compression is weakest in the distributional tails (5th/95th percentile, worst case) | Consensus regularises the centre, not the extremes |
| **H5** | Evaluation-side OAR contour choice contributes variance comparable to, or exceeding, planning-side contour choice for rectal and bladder endpoints | DVH scoring is directly sensitive to the scoring volume; the plan sees the OAR only through optimisation weights |
| **H6** | Unsigned overlap metrics (DSC, HD95, MSD, APL) cannot predict signed endpoint change; signed metrics can | Over- and under-contouring have opposite-signed dosimetric consequences; unsigned metrics are invariant to direction |
| **H7** | Dose-aware, regionally restricted metrics outperform whole-structure geometric metrics for OAR endpoints | Only the high-dose portion of an OAR surface influences its DVH |

**Failure to reject H3 is publishable.** If consensus does compress dosimetric
variability efficiently, that is a genuine result bounding where robust optimisation
adds value. The paper is structured so that outcome needs no restructuring.

---

## 3. Endpoints

Fixed in `config/endpoints.yaml`. Prescription 78 Gy in 39 fractions.

- **Target (CTV):** D98, D95, D50, D2, D_mean, V95%(74.1 Gy), homogeneity index
- **Coverage ladder:** D98 and V95% on CTV and CTV+3/+5/+7 mm, plus the boundary dose
  gradient in %/mm. These are expansion shells, **not PTVs** — nothing was optimised
  against them, and they must not be called PTVs in the paper.
- **Rectum:** relative V70, V65, V60, V50, D_mean; absolute D2cc, D1cc, V70cc, V60cc
- **Bladder:** relative V70, V65, D_mean; absolute D2cc, V70cc
- **Conformity:** ICRU 83 treated volume (95% isodose), Paddick CI against each j
- **TCP:** under fixed clonogen density and fixed clonogen number
- **NTCP:** LKB rectum and bladder, under relative and absolute volume normalisation

Primary endpoints for the Aim 3 prediction models, fixed in advance:
**ΔD98(CTV)** and **ΔNTCP(rectum)**.

### Volume-normalisation artefact
For both TCP and NTCP, part of the apparent variability is mechanical dilution from
contour size, not dose. It is quantified explicitly by recomputing with the dose held
fixed and only the scoring volume changed. The difference between the two normalisations
**is** the artefact.

---

## 4. Metrics for Aim 3

Fixed in `config/metrics.yaml`. §7.4 of the planning document says "pre-specify ~8
metrics, no more" and then lists about fourteen. Both cannot hold; the "no more" clause
is the one that protects the paper. **Eight are primary**, spanning the three classes
evenly rather than favouring any hypothesis:

- Unsigned: DSC, HD95, MSD
- Signed: signed volume difference, signed MSD
- Dose-aware: mean dose in the false-negative volume, mean dose in the false-positive
  volume, DSC restricted to the 90% isodose

The remaining metrics (surface DSC at 3 mm, APL, FP/FN volumes, dose-weighted surface
distance, integral dose in FP/FN, posterior interface overlap and its dose-weighted
variant) are **secondary**: computed and reported, excluded from the primary
metric × endpoint grid and its FDR correction, and never promoted after the fact.

**Prediction targets.** Primary is the replanning benefit Δ_B = endpoint(j,j) −
endpoint(i,j), which is non-circular but not computable at decision time. Secondary is
the estimation error Δ_A = endpoint(i,j) − endpoint(i,i), which is computable and
therefore usable only as a pre-plan screening tool — stated as such.

**No acceptability thresholds are derived.** The form of a useful metric may be
proposed; deriving thresholds requires a far larger cohort.

---

## 5. Statistical plan

Fixed in `config/analysis.yaml`.

**Three estimands, all reported for every endpoint.** E1 delivery variability (spread
across plans i at fixed truth j; identically zero in Arm C by construction, surfaced
rather than hidden). E2 residual truth uncertainty (spread across truths j at fixed plan
i; non-zero in both arms). E3 expected systematic loss, E[endpoint(i,j)] −
E[endpoint(j,j)].

**Compression factor** R = spread(Arm S) / spread(Arm C) on E2-comparable terms, against
the 1.34–1.49 geometric benchmarks. Spread is **IQR and 5th–95th percentile range, not
SD** — dose response is asymmetric and saturating, so a symmetric second moment
misrepresents it. Worst case reported separately.

**Uncertainty** by cluster bootstrap over patients, 10 000 resamples, seed 20260904,
percentile intervals at 95%. Resampling patients rather than cells: cells within a
patient share anatomy and a dose grid, and treating 530 as independent would understate
every interval substantially. With five patients the intervals will be wide. That is
honest, and it is why this is a variance-decomposition study.

**Variance decomposition** by linear mixed model with observers crossed with patients:
`Y ~ 1 + (1|Patient) + (1|PlanObs) + (1|EvalObs) + (1|Patient:PlanObs) +
(1|Patient:EvalObs) + (1|PlanObs:EvalObs)`. Percentage of variance per component, per
endpoint. Primary engine R/lme4, cross-checked in Python/statsmodels; disagreement
beyond 2 percentage points on any component is reported.

**Prediction** validated leave-one-patient-out, patient-clustered correlations,
Benjamini–Hochberg FDR at q = 0.05 across the primary metric × endpoint grid. Pooling all
530 pairs into a naive Pearson r is forbidden: inter-patient anatomy would dominate.

**Constraint flip rates** from the department protocol, reported binary (per-protocol
versus not) and ordinal (3×3 tier transitions). Degenerate constraints are reported as
such rather than dropped.

---

## 6. Data and quality control

Established by stages 00–06, all gates passing.

- 5 patients, 59 RTDOSE, one merged RTSTRUCT per patient, 78 Gy in 39 fractions,
  single-arc VMAT, Elekta VersaHD 6 MV.
- **Observer↔dose mapping verified independently** by permutation test on the
  within-column z-score of the diagonal: p = 5.0×10⁻⁵ (the floor at 20 000 permutations)
  for all five patients. `PatientName` is identical across every dose file, so the
  folder name is the only label the data carries, and this had to be established from
  the dosimetry.
- **Plan normalisation:** diagonal CTV D50 = 80.25 Gy, range 79.97–80.55, SD 0.128 across
  54 plans. The clinical protocol prescribes to PTV D95 = 78 Gy; these CTV-optimised
  plans do not follow that convention, by design.
- **PTV margin:** 7 mm isotropic, measured from the masks (median CTV-to-PTV surface
  distance 7.07 mm in all five patients, IQR 1.0 mm, no directional asymmetry). It is
  recorded nowhere in the DICOM.
- **Analysis grid:** 1.0 mm in plane, native CT slice spacing in z, so no contour is ever
  interpolated into existence. Volumes stable to better than 0.1% across a 0.75–2.5 mm
  convergence sweep.
- **Dose-grid containment:** all 2788 (plan × contour set × structure) combinations are
  100% inside their dose grid, so absolute-volume endpoints are valid for every cell.
- **DVH code** cross-validated against an independently implemented histogram-based
  estimator on 810 real cells; worst disagreement 0.0100 Gy, the reference's bin width.

Accepted QC failures and their justifications are in `config/waivers.yaml` and are
reproduced verbatim in the methods.

---

## 7. Limitations, stated in advance

- n = 5. Framed as variance decomposition, not significance testing.
- CTV-optimised plans give an **upper bound** on target sensitivity. Quantified by the
  boundary dose gradient in %/mm rather than hand-waved.
- VOTE-plan leakage: the consensus plan retains planning-side information from all ten
  observers. Bounded geometrically by the leave-one-out analysis, not removable.
- No pathological ground truth.
- Single institution, single protocol, single site.
- Absolute TCP and NTCP depend on model calibration and are not predictions of outcome
  for these patients. The design measures the local slope robustly; the absolute level
  it does not.
- HD95 and surface DSC carry a floor set by the CT slice thickness (2.5 mm; 2.0 mm for
  V027).
- Six protocol constraints cannot be evaluated for want of a structure (anterior
  prostate, urethra PRV, femoral heads, two posterior-rectal-wall constraints) or because
  they belong to the +SV arm of the protocol (SV D90). Named in the methods.

---

## 8. Declared prior observations

Honesty requires listing what was seen before this protocol was frozen. Each was
produced by a QC check, not by an analysis, and none tests a hypothesis above.

1. **A coverage asymmetry appeared in the Tier 1 mapping check.** In 25 of 54 columns the
   plan optimised on the evaluation contour is not the best plan for it, and the plan
   built on the largest CTV (O10) wins 17 of those columns. This is consistent with H6
   but was not derived from an Aim 3 analysis, and the Aim 3 result will be reported
   independently of it.
2. **A constraint flip-rate preview was computed** on the non-derived constraints, to
   check whether the flip-rate analysis would be informative at all before committing to
   it. It showed CTV D99 and PTV D95 changing tier in 49/49 columns, rectum V75 in 44/49
   and rectum V70 in 39/49, with PTV D50 and rectum V40 degenerate. This informed the
   decision to report degeneracy explicitly; it did not shape any hypothesis, all of
   which predate it.
3. **Observer volume tendencies** were flagged by the Tier 2 cohort-outlier check: O5
   contours systematically small, O3's seminal vesicles systematically large, O10's
   colon and bowel bag systematically large. §7.2's per-observer random effects analysis
   will be run on the full data regardless.

---

## 9. Amendment log

Any change after the `protocol-v1` tag is recorded here with its date and reason, and
declared in the paper.

**2026-09-05 - rigid dose-shift sensitivity analysis removed.** The plan listed a
+/-3 and +/-5 mm rigid dose-shift analysis to contextualise contour-driven variability
against setup-driven variability. Removed at the author's direction on the grounds that
it adds a second uncertainty axis to a paper that already carries three aims, and is not
required by any hypothesis. Nothing else depends on it; it remains a natural companion
analysis. Recorded here because it was pre-registered.

**2026-09-05 - compression factor reported on a second basis (addition, not substitution).**
The pre-registered definition computes the compression factor on E2, the spread of the
endpoint across plausible truths at a fixed plan. On analysis E2 proved to be dominated
by how much the ten truths differ from each other, which barely depends on which plan is
scored, so an E2 ratio sits near 1 almost regardless of what consensus does. The
geometric benchmarks it is compared against come from a variance identity on the
distance between a contour and its reference, whose dosimetric analogue is the spread of
the per-cell planning error, endpoint(i, j) - endpoint(j, j). Both are now computed and
reported side by side; the pre-registered E2 form is retained unchanged. No hypothesis
was altered. See docs/DECISIONS.md for the full reasoning, including a pooling confound
found and removed in the process.

**2026-09-11 - inference moved to the patient level (substitution).** Every interval was
previously a 95% percentile interval from a cluster bootstrap over patients. With five
clusters that bootstrap has 126 distinct resamples and under-covers, and several claims
rested on interval limits. All intervals are now t-intervals on per-patient estimates
(log scale for ratios, 4 df). Raised in external review. Stage 14.

**2026-09-11 - benchmarks derived for the voting rule used (substitution).** The 1.34 /
1.49 benchmarks assume the consensus is the mean of the observers; a strict-majority
vote is an order statistic. Simulated for the operator: 1.31 leave-one-out, 1.53 full.
H1 and H2 are unchanged in form; the reference values they are read against are not.

**2026-09-11 - R_2.2 and R_2.3 estimator (post-hoc, declared).** The results planning
document (not registered, and not part of this protocol) specified median(A)/median(B).
The mean was adopted after the median-based values had been computed, because A is
already a mean of nine absolute differences and B is a single one; under the null the
median form returns 1.40 rather than 1.31. Both are reported with equal prominence.

**2026-09-11 - 5-of-10 consensus control (addition).** Built from the same masks in the
four ten-observer patients to test whether the consensus displacement is a property of
the voting threshold. Geometric only; no plans exist for it.

**2026-09-14 - dose-aware predictive metrics not reported (removal).** Three of the eight
pre-specified Aim 3 metrics (mean dose in the missed region, mean dose in the added region,
Dice within the high-dose region) are not reported. They use the delivered dose as well as
the contours, which makes them close to a partial calculation of the dosimetric target, and
they cannot be computed before a plan exists. Declared in Section 4.

**2026-09-14 - PTV coverage constraints not reported (removal).** PTV78 D95, D50 and D2 are
dropped from the flip-rate analysis. The plans were optimised to the CTV and the PTVs were
created afterwards, so PTV coverage describes a structure the plans never saw.

**2026-09-14 - Aim 3 inference (substitution).** Pooled Spearman p-values with
Benjamini-Hochberg correction are replaced by within-patient Spearman correlations
summarised over patients, with leave-one-patient-out R-squared as the measure of predictive
value. The pooled p-values treated non-independent comparisons within a patient as
independent.
