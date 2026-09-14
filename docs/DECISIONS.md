# Decision log

Every judgement call, dated, with its reason. Entries are appended, never rewritten;
a decision that turns out wrong gets a superseding entry rather than an edit.

---

## 2026-09-04 — Plans are CTV-optimised; PTVs are post-hoc

**Decision.** Treat `CTV_CT_<set>` as the optimisation target and `PTV_<set>` as an
evaluation shell only.

**Why.** Confirmed by the user: the plans are CTV-optimised and the PTVs were created
afterwards as isotropic expansions of the CTV. This is consistent with the dosimetry —
diagonal CTV D50 is pinned at 80.25 Gy (range 79.97–80.55, SD 0.128 across all 54
plans) while diagonal PTV D98 floats between 73.7 and 77.4 Gy, which is what incidental
coverage of a structure nothing optimised against looks like.

**Consequence.** The §5 expansion-shell ladder (CTV+3/+5/+7 mm) stands as written, and
the margin-free framing in the introduction is correct. The PTV is still useful as the
evaluation structure for the Tier 1 mapping test because it is more sensitive to plan
identity than the CTV.

---

## 2026-09-04 — VOTE is the consensus arm; STAPLE is excluded

**Decision.** `analysis_sets` = 10 observers + VOTE. `truth_sets` = the 10 observers
only. STAPLE is extracted but excluded from all analysis.

**Why.** User correction: VOTE is the consensus contour and forms Arm C. STAPLE is the
one dropped, for plan quality (planning document §3). VOTE is excluded from the truth
column because it is a derived quantity from the ten observers, not an independent
plausible anatomy (planning document §3); it appears as truth only in the secondary
sensitivity analysis.

**Consequence.** Primary design is 530 cells: Arm S 432 (4×90 + 1×72), diagonal 49,
Arm C 49. Secondary (j = VOTE) is 54.

**Open.** The consensus algorithm is described only as "a Python library". §7.2 needs
leave-one-out VOTE regeneration for H1's 1.34 benchmark. Proposed approach, pending
confirmation: reimplement simple majority vote, verify the reimplementation reproduces
the supplied VOTE to within a stated DSC, and only then generate the LOO variants. If
the reproduction fails, H1 is reported against the 1.49 uncorrected benchmark alone and
the omission is stated.

---

## 2026-09-04 — O6 is dropped from M030

**Decision.** M030 runs with nine observers.

**Why.** `Rectum_O6` is absent from the M030 structure set and no M030/O6 plan was
exported. Confirmed by the user.

**Consequence.** The design is unbalanced. The §7.3 crossed mixed model handles this;
the cell counts in the abstract need updating from 550 to 530.

---

## 2026-09-04 — `DoseSummationType = BEAM` is correct, not a mislabel

**Decision.** Waive the `dose.summation_type_is_plan` check, with the reason recorded
in `config/waivers.yaml` and reproduced in the methods.

**Why.** Inspection of the V027/O9 RTPLAN shows a single DYNAMIC beam of 89 control
points spanning gantry 184°→176° CW, i.e. one 352° VMAT arc. With one beam in the plan,
the beam dose *is* the plan dose and `ReferencedBeamNumber = 1` is literally correct.

**To verify.** Re-confirm across all 59 plans once they are supplied — M030/O3 has an
unusually large dose grid (80×106×93) that may indicate a second arc.

---

## 2026-09-04 — The mapping test is column-oriented and threshold-free

**Decision.** Verify the observer↔dose mapping with a per-patient permutation test on
the within-column z-score of the diagonal, at α = 0.001. Implemented in
`src/mcx/qc/identifiability.py`.

**Why.** `PatientName` is `<initials>_Atlas_<patient>_CT` on every RTDOSE and RTPLAN file, so
the folder name is the only thing linking a dose to a contour set. That has to be
established from the dosimetry or every downstream result rests on a filename.

The first attempt was row-oriented — fix the plan, vary the contour, assert the diagonal
tops its row — and it failed on 23 of 49 plans. The failure was in the test, not the
data: D98 is confounded by structure size, so the observer with the smallest PTV (O5, in
all five patients) scores highest under *every* plan. The row maximum identifies the
smallest contour, not the matching plan. Fixing the evaluation contour and varying the
plan removes that confound entirely.

The threshold was then removed as well. Under the null that plan labels are unrelated to
contour labels, the diagonal is a random draw from its column and the mean within-column
z-score is centred on zero, so a permutation test gives a p-value with nothing chosen by
inspection.

**Result.** All five patients: p = 5.0e-05, the floor at 20 000 permutations.
Diagonal z ranges from +0.69 (M030) to +1.22 (K018). The mapping is confirmed.

---

## 2026-09-04 — Per-column diagonal losses are a finding, not a defect

**Decision.** `map.diagonal_wins_column` warns; it does not fail.

**Why.** In 31 of 54 columns the diagonal is not rank 1, and the pattern is systematic:
**O10's plan wins 17 of those columns, and O10 has the largest CTV in four of five
patients.** A plan optimised on a large target delivers a large high-dose region, which
covers everyone else's smaller PTV well — including their own. Conversely O5, with the
smallest CTV throughout, loses columns it should win.

This is not a labelling error. It is the coverage asymmetry the paper exists to
quantify: over-contouring is dosimetrically forgiving for target coverage and
under-contouring is not. It bears directly on H6, since an unsigned overlap metric
cannot distinguish the two.

**Planned refinement.** D98 rewards large planning targets. A conformity statistic —
Paddick CI of the ICRU 83 treated volume against contour j — penalises over- and
under-coverage symmetrically and should put the diagonal at rank 1 far more often.
Adding it as a second, independent mapping statistic is deferred to stage 07, where the
treated volume is computed anyway.

---

## 2026-09-04 — Even-odd (XOR) polygon fill

**Decision.** Rasterise every ROI with even-odd fill.

**Why.** It is the only rule that handles both cases in this dataset with one code path.
Seminal vesicles routinely have two disjoint loops per slice, which even-odd unions
correctly; bladder and rectum occasionally have a nested loop (gas or lumen), which
even-odd turns into a ring correctly. A union rule would fill the holes; a
largest-loop-wins rule would drop a vesicle.

**Check.** Stage 03 compares the resulting mask volume against an independent shoelace
volume computed from the polygons, so a fill error cannot pass silently.

---

## 2026-09-04 — `derived/` lives outside OneDrive

**Decision.** Resolve the cache to `%LOCALAPPDATA%/MultiContouring/derived` unless
`MCX_DERIVED` or `config/dataset.yaml` says otherwise.

**Why.** A multi-gigabyte mask cache inside a synced folder causes constant re-upload and
file-lock errors. `data/` and `derived/` are both gitignored.

---

## 2026-09-04 — Analysis grid is 1 mm in plane, native CT spacing in z

**Decision.** One grid per patient: 1.0 mm in x and y, z planes coincident with the
planning CT slice positions, extent set by the union of CTV/PTV/rectum/bladder/seminal
vesicles across every contour set plus 20 mm.

**Why not 1 mm isotropic.** Contours exist only at CT slice positions, and 2.5 mm is not
a multiple of 1 mm, so on an isotropic grid most contours would fall between planes and
have to be interpolated — inventing geometry the data does not contain. Native z means
every contour lands exactly on a plane and nothing is fabricated. The cost is that HD95
and surface DSC carry a floor set by the slice thickness (2.5 mm, or 2.0 mm for V027),
which is stated in the methods; surface extraction uses the true anisotropic spacing.

**Evidence for 1 mm in plane.** The stage 03 convergence sweep recomputes volume across
0.75/1.0/1.5/2.0/2.5 mm. Total volumes agree to better than 0.1 % throughout, so 1 mm is
amply converged rather than merely assumed.

**Consequence.** Grids are 1.3–2.6 M voxels; 432 masks build in about 90 s and cache to a
few tens of kilobytes each, bit-packed.

---

## 2026-09-04 — The PTV margin is 7 mm isotropic, measured not assumed

**Finding.** The expansion distance appears nowhere in the DICOM. Measuring the distance
from the CTV surface to the PTV surface across all 59 contour sets gives a median of
**7.07 mm in every one of the five patients**, IQR 1.0 mm. Splitting the PTV surface by
direction shows no anterior/posterior/lateral asymmetry (6.4–7.5 mm in every direction,
the spread being voxel discretisation), so it is isotropic, not a posterior-reduced
clinical margin.

**Consequence.** The §5 expansion-shell ladder (CTV+3/+5/+7 mm) now has a reference
point: the existing PTV is the +7 mm rung, so that rung can be cross-checked against a
structure that was actually drawn.

---

## 2026-09-04 — Tier 2 gates on analysis structures, warns on descriptive ones

**Decision.** Containment, volume round-trip, internal gaps and degenerate loops FAIL for
CTV, PTV, rectum and bladder; the same checks WARN for seminal vesicles, PRS, colon and
bowel bag.

**Why.** The analysis grid extent is set by the analysis structures plus 20 mm, so colon
and bowel bag genuinely extend beyond it — 71 of the containment warnings are that, by
design. Failing on them would be failing on a design choice. Including them in the extent
instead would inflate every grid by hundreds of millimetres for structures that are not
comparable between observers in the first place (5 to 77 slices for the same patient).

**Result.** All 236 analysis-structure masks are within 1.05 % of their independent
shoelace volume, median 0.19 %. Every rasterisation defect in the dataset falls outside
the analysis structures except the two waived stray loops.

---

## 2026-09-04 — Two stray contour loops, waived with measurements

**Decision.** Waive `raster.no_degenerate_loops` for M020/O2 bladder and
`raster.no_subvoxel_slivers` for V027/O6 bladder.

**Why.** The first is a one-point contour of zero area on a slice that also holds the
real 206-point bladder contour of 3607 mm²; the second is a three-point triangle of
0.09 mm² on a slice holding the real 402-point contour of 2723 mm². Both are stray clicks
in the contouring session. The rasteriser now counts these two causes separately and
records the discarded polygon area, so the waivers quote a measured quantity rather than
an assertion.

---

## 2026-09-04 — All 59 doses are complete plan doses; 57 RTPLANs not needed

**Decision.** Close the request for the remaining RTPLAN files.

**Why.** The waiver on `DoseSummationType = BEAM` needed the claim "every dose file is a
complete plan dose", not "every plan is single-arc" — and the first is provable from the
dose data already in hand. A file containing one arc of a multi-arc plan would show
roughly half the intended dose; the diagonal CTV D50 of all 54 analysis plans falls
between 79.97 and 80.55 Gy (median 80.25, SD 0.128). None is a partial delivery.

This also disposes of the M030/O3 concern: its dose grid is larger than any other
(80×106×93), but its maximum is 83.12 Gy and its D50 is in band, so the grid is a
manually enlarged calculation box, not a second arc.

The two RTPLANs supplied (V027/O9, M030/O5) are structurally identical — 39 fractions,
one DYNAMIC beam of 89 control points, 352° CW arc, 6 MV, Elekta VersaHD, collimator 10°,
couch 0°, 550 MU/min — differing only in MU (537, 548), beam dose per fraction (2.05879,
2.05042 Gy) and isocentre.

**Reopens if.** Per-plan MU is wanted as a plan-complexity covariate in Aim 2. That is a
Phase F question and cheap to revisit.

---

## 2026-09-04 — Dose outside the calculation box is NaN, never zero

**Decision.** `resample_to_grid` returns dose with NaN wherever the analysis grid falls
outside the source dose grid, plus a boolean coverage mask alongside it.

**Why.** Filling with zero would read as "no dose here", which is a different and false
claim, and would silently corrupt every absolute-volume endpoint on a structure that
pokes outside the calculation box — D2cc would quietly become the dose to whatever
happened to be inside. Making it NaN forces every consumer to decide explicitly.

**Result.** The decision turned out to be free: **all 2788 (plan × contour set ×
structure) combinations are 100 % inside their dose grid**, so no cell is affected.
That was a live risk before it was measured, and D2cc, V70cc and V60cc are valid
everywhere as a result.

---

## 2026-09-04 — The DVH code is cross-validated against a second estimator, not a library

**Decision.** `mcx.endpoints.dvh_reference` implements a histogram-based DVH estimator,
independent of the sort-based one used in the pipeline. Tier 3 runs both on real cells
and requires agreement.

**Why not dicompyler-core.** It is unusable here: it still imports the pre-2020 `dicom`
module and does not work against pydicom 3. Writing the second estimator has a further
advantage over a third-party one — it can be pinned to closed-form answers. Both
implementations are tested against a linear dose ramp over a cuboid, where every
endpoint is analytic.

**Result.** On 810 real comparisons the two agree to **0.0100 Gy at worst**, which is
exactly the reference implementation's histogram bin width, and to 6e-5 Gy on mean dose.
The only disagreement is the reference's own discretisation.

---

## 2026-09-04 — OAR D50 is the endpoint most sensitive to discretisation

**Finding.** Comparing the native-dose-grid pipeline against the analysis-grid pipeline
end to end, agreement by endpoint is: D98 0.040 Gy median (0.265 worst), Dmean 0.095 Gy
(0.528), D2 0.105 Gy (1.216), D2cc 0.083 Gy (1.984), **D50 0.103 Gy median but 1.619
worst** — and every one of the largest non-V027 disagreements is a rectum or bladder D50.

**Why.** An OAR's D50 sits in the low-to-mid dose region where a large volume fraction
occupies a shallow dose gradient, so a small change in which voxels are included moves
it disproportionately. D98 on a target sits on a steep, well-defined part of the curve
and is correspondingly stable.

**Consequence.** Worth stating in the methods. It does not threaten the §5 endpoint list,
which uses V_x and D_mean rather than D50 for rectum and bladder, but it is a reason to
be wary of OAR median-dose endpoints generally — including in Aim 3, where a metric that
predicts ΔD50 well may be predicting discretisation noise.

---

## 2026-09-04 — Cache keys use a geometry hash, not the full config hash

**Decision.** `Config.geometry_hash` covers only what can move a voxel — patient and
observer lists, the alias map, known-absent pairs, ROI templates and overrides, fill
rule, and the analysis-grid definition. `config_hash` remains the full fingerprint and is
still stamped on every results row.

**Why.** Editing a comment in `structures.yaml` invalidated 432 masks and 59 resampled
doses. Over-invalidation is the safe direction, but not when it discards several hundred
megabytes of correct geometry for a documentation change. Provenance and cache identity
are different questions and should not share a key.

---

## 2026-09-04 — The max-dose check is relative, and is a smoke test

**Decision.** `dose.max_preserved_by_resampling` bounds the loss at 1 % rather than
0.5 Gy.

**Why.** The first version failed on V027/O4 at exactly 0.500 Gy, on a threshold that was
arbitrary in the first place. The mechanism is understood: trilinear interpolation cannot
exceed the source maximum and under-reads a peak whenever the analysis grid does not
sample the voxel holding it. The loss scales with dose, so a relative bound is the
correct form. Observed loss is 0.12 % median for the four patients whose CT and dose
grids share 2.5 mm slices, rising to 0.29 % median and 0.61 % worst for V027, whose
2.0 mm analysis planes essentially never coincide with its 2.5 mm dose planes.

The threshold was not moved to make a failure disappear: no endpoint in §5 uses the
global maximum, which is a single voxel. The check exists to catch a scaling or geometry
error, which would present as a whole-number percentage. The endpoint that carries the
same information, D2 %, is checked directly against the native-grid pipeline and agrees
to 0.105 Gy median. `dose.native_and_analysis_agree` is now reported per endpoint rather
than per structure, so a sensitive endpoint is named instead of averaged into a worst case.

---

## 2026-09-04 — Prescription confirmed: 78 Gy in 39 fractions

**Decision.** `prescription.unconfirmed` is now false. Department protocol "Definitive
Prostate +/- SV: 78 Gy to D95", transcribed into `config/constraints.yaml` from
PMBaaa50c supplementary Table 1.

**Consequence.** EQD2, LKB NTCP and TCP are unblocked. The clinical protocol normalises
to PTV D95 = 78 Gy; these CTV-optimised research plans do not follow that convention
(diagonal PTV D95 76.4-78.0 Gy against a pinned CTV D50 of 80.25 Gy), which is stated in
the methods and is why PTV-based constraints sit systematically in minor variation.

---

## 2026-09-04 — The protocol is three-tier, so flip rates are reported two ways

**Decision.** Report the binary rate §7.3 asks for (per-protocol versus not) *and* the
full 3x3 tier-transition matrix.

**Why.** The department protocol grades per-protocol / minor variation / major variation
rather than pass/fail. Collapsing that to binary would discard the distinction between a
plan that slips one tier and one that fails outright, which is the clinically meaningful
difference. A two-sided constraint that falls below its band is recorded as `under_dose`
rather than forced into a tier the protocol does not define.

---

## 2026-09-04 — Five protocol constraints cannot be computed, and one does not apply

**Decision.** Keep all 24 protocol rows in config with an `applicability` field, rather
than deleting the ones we cannot use. 18 are computable per observer, 1 is
patient-level, 5 are unavailable.

**Unavailable for want of a structure:** anterior prostate D90 (needs CTV_ANTPROST,
clinical set only), rectum posterior wall Dmax and the PRW slice count (need a posterior
rectal wall; deriving one from a solid rectum would invent a convention the observers
never applied, and that convention is exactly what varies between them), urethra PRV V82
and femoral heads V40 (no such contours exist anywhere in this dataset).

**Not applicable — SV D90.** These plans are the prostate-only arm. Measured: the
seminal vesicles overlap their own observer's CTV by a median of **0.3%** of SV volume,
and SV D90 under that observer's own plan is **33 Gy** against a protocol aim of 78.5 Gy.
The protocol row refers to `CTV_SV`, a target sub-volume; `Seminal_Vesc_<observer>` here
is an anatomical contour that was drawn but not treated. Scoring it would have produced
100% major variation across all 530 cells — a confident-looking artefact of applying the
wrong arm of the protocol. Incidental SV dose is real and observer-dependent (median
Dmean 59.8 Gy, V70 43.6%) and is reported descriptively.

**Side effect.** VOTE's missing seminal vesicle contour now costs nothing: no constraint
and no §5 endpoint depends on it.

---

## 2026-09-04 — Composite constraint structures are built from the evaluation side only

**Decision.** `composite_structure_rule: evaluation_side_only`. Rectum-in-PTV,
rectum-outside-PTV, the rectum PRV and peripheral tissue are all built from the assumed
true anatomy j. The plan i enters a cell only through the dose distribution.

**Why.** Mixing j's rectum with i's PTV is a defensible-looking alternative that would
quietly confound plan-side and evaluation-side effects — which is precisely the
separation H5 exists to test.

---

## 2026-09-04 — Preview: the flip-rate analysis will work, and which constraints carry it

Computed on all 584 (plan, contour, constraint) combinations for the non-derived
constraints, asking how often the tier changes across plans for a **fixed** evaluation
contour — the E1 quantity:

| Constraint | Columns where the tier flips |
|---|---|
| CTV D99, PTV D95 | 49/49 (100%) |
| Rectum V75 | 44/49 (90%) |
| Rectum V70 | 39/49 (80%) |
| Bladder V50, V60 | 29/49 (59%) |
| Rectum V65 | 27/49 (55%) |
| Rectum V60 | 9/49 (18%) |
| Rectum V50 | 2/49 (4%) |
| PTV D2 | 1/49 (2%) |
| PTV D50, Rectum V40 | 0/49 — degenerate |

The informative constraints are the high-dose ones, which is mechanistically right, and
CTV D99 ranges from 64.3 to 79.2 Gy across the design. PTV D50 and rectum V40 are
degenerate here and contribute nothing to a flip rate; that is reported rather than
hidden, since a constraint that never changes tier is a real (if null) result about which
parts of the protocol are robust to contouring variation.

---

## 2026-09-04 — QUANTEC NTCP, and a calibrated TCP with the published value as sensitivity

**Decision.** LKB rectum from QUANTEC (Michalski 2010: n = 0.09, m = 0.13,
TD50 = 76.9 Gy). LKB bladder from Emami/Burman (n = 0.5, m = 0.11, TD50 = 80 Gy) —
which is the QUANTEC-endorsed set rather than a QUANTEC fit, since QUANTEC's bladder
review concluded the data were inadequate to derive new parameters. That distinction
goes in the methods. alpha/beta 3 Gy for both late endpoints, 1.5 Gy for prostate.

**TCP needed a judgement call.** With the published clonogen density of 1e7 per cm³ the
Poisson model returns **TCP = 0.975 at 78 Gy** to a 50.6 cm³ target, with a local slope
under 2 % per Gy — the flat top of the sigmoid, where every difference between contour
sets is compressed toward zero. Reporting that would be a finding about where the model
sits on its own curve, not about contouring.

Solving for the density giving 85 % control at the prescription (rho = 2.44e8 per cm³)
puts it on the steep part: local slope 3.5 % per Gy at 78 Gy, maximum slope 6.6 % per Gy.
**Every TCP result is reported at both densities.** If they disagree qualitatively, the
TCP conclusions are calibration-dependent and the paper must say so.

**Also confirmed by test:** Gaussian heterogeneity in alpha is doing real work — it cuts
the maximum dose-response slope from 18.6 % to 6.6 % per Gy. A single-alpha model would
have exaggerated every contour effect roughly threefold.

---

## 2026-09-04 — Phase H added: the results dossier is the deliverable

**Decision.** The plan ended at parquet tables and figures, which is pipeline output, not
something a paper can be written from. Phase H produces a **results dossier** in two
synchronised renderings — a sectioned `.docx` with paper-ready tables and a published
HTML artifact — both generated from the same `results/*.parquet` so they cannot diverge.

Organised by paper section rather than by pipeline stage. Every result carries a
sentence-ready statement with effect size and uncertainty, the table, the figure, one
line of interpretation kept visibly separate from the measurement, and the stage and file
that produced it. A `results/quotable.csv` keys every citable number to a stable
identifier so none is retyped by hand.

**Why it was missing.** The planning document specifies figures and tables (§8) but not
the artefact that makes them usable while writing. Adding it late would have meant
retrofitting provenance onto numbers already generated; adding it now means every stage
from here on emits what the dossier needs.

---

## 2026-09-05 — The consensus algorithm is confirmed as strict majority vote

**Finding.** Sweeping every possible vote threshold against the supplied VOTE contours
gives **strict majority (⌊n/2⌋+1) as the best match in 19 of 20 patient × structure
cases** — 6 of 10 observers, and 5 of 9 for M030. Dice against the supplied contours is
0.913–0.983, median 0.963.

**The residual gap is discretisation, not a different algorithm.** Every disagreeing
voxel lies within 3.91 mm of the VOTE surface and 100 % (median) within two voxels: a
thin surface skin, not blobs. That is the signature of the mask → contour → mask round
trip the supplied contours went through and a reimplementation does not.

**Consequence.** The leave-one-out consensus contours are legitimate, so H1 can be
tested against both the 1.34 and 1.49 benchmarks as §7.2 intends.

**Also established:** `PTV_VOTE` is a majority vote of the observers' PTVs, not a 7 mm
expansion of `CTV_VOTE` (Dice 0.96 against 0.93, and 132 cc against 115 cc). Vote-then-
expand and expand-then-vote do not commute, which is itself a small illustration of H2's
claim that majority vote is a non-linear morphological operator.

---

## 2026-09-05 — M030's consensus uses a 5-of-9 threshold, not 6-of-10

**Caveat, stated because it is a real confound in Arm C.** M030 has nine observers, so
its strict-majority threshold is 5 of 9 (55.6 %) against 6 of 10 (60 %) elsewhere. A
lower effective threshold produces a slightly larger consensus, and that is visible:
M030 is the only patient whose VOTE is *larger* than its mean observer contour
(volume ratio 1.01–1.04 against 0.89–1.00 for the other four).

It is not correctable — using 6 of 9 would be a different rule from the one the other
patients used — so it is reported. It affects Arm C for M030 only.

---

## 2026-09-05 — Surface DSC is reported but excluded from the headline compression

**Decision.** The compression factor is computed on DSC, MSD and HD95. Surface DSC is
computed and reported but flagged `headline = False`.

**Why.** For an overlap metric the compression factor must be computed on the
*disagreement*, (1 − metric), since agreement is bounded above by 1 and a ratio of
agreements is meaningless. That transform becomes unstable as the metric saturates:
bladder surface DSC is 0.969 between observers and 0.992 against the consensus, so the
disagreements are 0.031 and 0.008 and their ratio is 3.8–5.3. That number is driven by
the last decimal place of a saturated metric, not by anatomy.

The guard is automatic — any overlap metric whose disagreement falls below 0.05 is
flagged — rather than a one-off exclusion of a metric that happened to misbehave.

---

## 2026-09-05 — A surface-distance bug, found by a unit test

**What was wrong.** The first implementation measured surface distance by evaluating a
signed distance field of the *whole region* at the other contour's surface voxels. A
surface voxel lies inside its own mask, so two identical contours reported a mean
surface distance of about half a voxel instead of zero, and a one-slice shift on a
2.5 mm grid reported 5.0 mm instead of 2.5 mm.

**The fix.** Distances are measured surface to surface — from each surface voxel of one
contour to the nearest surface voxel of the other — which is zero on contact. Sign is
applied separately from containment. Four tests pin the behaviour, including one that
checks anisotropic spacing is respected, which is the classic way to get plausible but
wrong surface distances on a grid whose z spacing differs from its in-plane spacing.

**Also corrected: one of those tests was itself wrong.** It asserted that signed MSD
flips sign when the arguments are swapped. That holds only for nested contours; where
neither contains the other, both surfaces lie partly outside the other and both means
can be positive. The nested case is tested separately.

---

## 2026-09-05 - The compression factor is reported on two bases, and why

**Problem.** The pre-registered definition is spread(Arm S) / spread(Arm C) "on
E2-comparable terms". E2 is the spread of the endpoint across plausible truths at a
fixed plan. But E2 is dominated by how much the ten truths differ from each other, and
that barely depends on which plan is being scored - so an E2 ratio sits near 1 almost
regardless of what consensus does, and cannot on its own support or refute H3.

**The geometric benchmark is a different shape.** The 1.34 and 1.49 figures come from a
variance identity on the *distance between a contour and its reference*: SD(x_i - x_j)
against SD(x_i - x_bar). The dosimetric analogue is therefore the spread of the per-cell
**planning error**, endpoint(i, j) - endpoint(j, j), not the spread of the endpoint.

**Decision.** Report both, labelled. The error basis is the like-for-like comparison
with 1.34; the E2 basis is the pre-registered form and answers a real but different
question. This is an addition to the protocol, not a substitution, and is recorded as
such in the amendment log.

**A confound removed along the way.** Pooling all of Arm S mixes two sources of
variation: how the error moves as the assumed truth changes, and how plan quality
differs between observers. Arm C has one plan and carries only the first, so pooling
inflates Arm S and manufactures compression out of plan quality - something the
geometric benchmark has no analogue for, since a contour has no equivalent of a good or
bad plan. Taking one spread per plan and comparing medians removes it.

The correction was large and in the direction that matters: error-basis compression fell
from **1.9-2.9 (pooled, wrong) to 0.8-1.4 (per plan, right)**, which is the difference
between apparently refuting H3 in the wrong direction and supporting it. The two bases
now agree with each other, which they did not before.

---

## 2026-09-05 - Flip rates are reported in the truth direction

**Decision.** The headline flip rate fixes the delivered plan and varies whose contour
scores it. The other direction - fix the truth, vary the plan - is also reported but
labelled as **zero in Arm C by construction**: one consensus plan cannot disagree with
itself, so that direction measures nothing about consensus quality and is an E1-style
definitional artefact.

**Caveat that must travel with the Arm C numbers.** Arm C has one plan per patient, so
its flip rates are estimated from five observations. A rate of 100% means five of five.
They are reported with the denominator visible and are not compared to Arm S without it.

---

## 2026-09-05 - Two model endpoints are uninformative in this cohort, and that is a result

**Bladder NTCP is pinned at zero.** gEUD spans 16.7-57.0 Gy against a TD50 of 80 Gy, so
NTCP ranges from 3e-13 to 4.4e-3 across all 530 cells. With the QUANTEC-endorsed
Emami/Burman parameters the bladder never approaches tolerance in these plans. Detected
automatically by the Tier 4 informativeness check, and reported as a null: bladder must
be assessed on its DVH endpoints, not on NTCP. It is also a concrete instance of the
NTCP-transferability argument in section 9.

**TCP is hypersensitive to cold spots.** Twelve of 530 cells fall below TCP 0.5 while the
median is 0.908; the 1st percentile is 0.094 and the 5th is 0.849. The lowest-TCP cells
have CTV D98 of 70-77 Gy and V95% still above 95%. This is the Poisson model behaving as
designed - control requires killing every clonogen, so a small cold volume dominates -
but it means TCP variability is driven by a handful of cells and any TCP conclusion must
be reported with the tail explicit. Modelled TCP collapses once CTV D98 falls below about
76 Gy: median TCP is 0.003 for D98 of 70-74 Gy against 0.911 for D98 above 78 Gy.

---

## 2026-09-05 - Target dose metrics inside the optimised volume are insensitive by design

**Finding, from the Tier 4 diagonal-consistency check.** The ratio of off-diagonal to
diagonal IQR - contour effect against plan-to-plan reproducibility - is:

| Endpoint | Ratio | Reading |
|---|---|---|
| CTV D95, D50, D2, D_mean | 0.79-1.15 | indistinguishable from plan noise |
| CTV D98 | 1.41 | contour effect emerges |
| Rectum V70, D2cc | 2.33, 2.82 | contour effect dominates |
| CTV+3 mm D98 | 3.16 | |
| CTV+5 mm D98 | 5.78 | |

Both spreads for the central CTV metrics are 0.13-0.36 Gy, which is the plan
reproducibility floor. The mechanism is not subtle: every plan is normalised to CTV
D50 = 80.25 Gy, so metrics inside the optimised volume are pinned by construction.
Sensitivity appears at the target boundary and grows steeply outside it.

This is section 7.3's honest-reporting case - "if comparable, report it" - and it is also
the quantitative form of the CTV-optimisation limitation.

---

## 2026-09-05 - Rigid dose-shift sensitivity dropped

**Decision.** The pre-registered +/-3 and +/-5 mm rigid dose-shift analysis is not run.
Author's direction: it adds a second uncertainty axis to a paper already carrying three
aims, and no hypothesis depends on it.

Recorded in the protocol amendment log and marked `status: removed_2026_09_05` in
`config/analysis.yaml` rather than deleted, so the paper can state what was planned and
not done. Stage 11 was renumbered to the variance decomposition, which the plan had
under-specified.

---

## 2026-09-05 - H5: the variance decomposition is a two-way layout, not a mixed model

**Decision.** Decompose each patient's Arm S matrix into plan-side, evaluation-side and
interaction sums of squares, then combine patients by the cluster bootstrap used
elsewhere. The pre-registered crossed mixed model is attempted as a cross-check.

**Why.** The protocol names R/lme4 as primary with statsmodels as cross-check. R is not
available in this environment, and statsmodels fits crossed effects only through
variance-component formulas that are fragile on unbalanced data - the design is
unbalanced, since M030 has nine observers and the diagonal is excluded from Arm S.
**The cross-check converged for 0 of 38 endpoints**, which is exactly the failure mode
the protocol anticipated.

Within a patient the Arm S cells form a near-complete two-way layout: every plan against
every truth but its own. A classical decomposition of that layout is exact, needs no
optimiser, and answers H5 directly, since H5 is a comparison of two effects measured
within the same patient. Between-patient variance is reported alongside rather than
partitioned in; for bladder it is 69-87% of the total, which would otherwise swamp the
comparison.

**Result.** H5 is supported for the rectum: evaluation-side and planning-side variance
are comparable (ratios 0.87-1.29, every interval spanning 1). The target is
planning-dominated (CTV D98, 0.55 [0.33, 0.63]) as it should be, since that is what was
optimised. Paddick conformity is evaluation-dominated, 1.44 [1.22, 1.71].

---

## 2026-09-05 - Aim 3: H7 strongly supported, H6 supported for the OAR only

**H6 for the rectum is as clean as it gets.** Predicting the replanning benefit in
NTCP(rectum), the three unsigned overlap metrics achieve |rho| of 0.001, 0.002 and 0.006
- no predictive power whatever - while signed metrics reach 0.32 and dose-aware metrics
reach 0.73. Unsigned metrics are invariant to the direction of disagreement, and for an
OAR the direction is the entire signal.

**H6 for the target is not supported on the primary target.** Predicting the replanning
benefit in CTV D98, unsigned metrics (best |rho| 0.234) slightly outperform signed ones
(0.153). On the secondary target, estimation error, the ordering reverses sharply
(signed 0.617 against unsigned 0.234). Reported as the split result it is, not
generalised from the rectum.

**H7 is strongly supported.** Mean dose in the false-negative volume is the best
predictor for both endpoints and both targets: |rho| 0.73 and 0.88, with
leave-one-patient-out R-squared of 0.31 and 0.63 against essentially zero for
whole-structure geometric metrics.

Every correlation is within-patient, pooled by Fisher z; nothing is pooled across
patients, and the grid is FDR-corrected at q = 0.05.

---

## 2026-09-10 — Planning provenance, and what the timestamps show

**Fact.** One planner was given every contour set and produced all 59 plans; the
observers contoured only (user, 2026-09-10).

**What this settles.** The plan-quality component isolated in results Table 7 — roughly
half the diagonal spread for the rectum, three quarters for the bladder — is
within-planner variation, not differences in skill between planners. It is a random
nuisance rather than an attribute attached to an observer, which is what the column-wise
analysis of Section 2.3 needs it to be.

**What the RTDOSE creation timestamps then show.**

1. **The planning order is identical in every patient**: STAPLE, O4, O10, O8, O2, O7,
   O6, O9, O1, O5, O3, VOTE. Order within a patient is therefore perfectly aliased with
   observer identity, and no within-session learning effect can be separated from an
   observer effect. A Spearman test of endpoint against order returns rho = -0.39
   (p = 0.006) for Paddick CI, but that is a restatement of the observer effect already
   reported in Section 1, not evidence of drift. Do not report it as drift.

2. **VOTE was planned last in all five patients.** The consensus plan was the eleventh
   plan produced on that anatomy. This is a second mechanism, independent of contour
   self-inclusion, by which the consensus plan is flattered, and it strengthens rather
   than weakens the Section 2.3 result: despite both advantages the consensus compressed
   the penalty by less than the variance identity predicts.

3. **K018 was planned first, on a separate day, and took twice as long** — 134 minutes
   against 42–72 for the other four, with a 35-minute gap mid-sequence. It also has the
   largest plan effect of the five patients for rectum V70 (6.42 pp against 1.84–3.09),
   rectum mean dose and Paddick CI. Recorded as a watch-list item; not excluded, since
   its contour-driven results are consistent with the rest of the cohort.

**Not done.** No attempt to correct for planning order, because it cannot be identified
separately from observer identity in this design.

---

## 2026-09-11 - Revision after external review

Every testable claim in the review was checked against the data before anything was
changed. Stage `pipelines/14_patient_level_inference.py` produces every revised number.

**Confirmed and acted on.**

1. *Plan-to-plan noise attenuates R_2.3 toward 1.* Subtracting E(j,j) removes a
   per-anatomy offset, not the random part of each plan, so numerator and denominator
   carry the same added variance. The noise share needed to reconcile each observed
   R_2.3 with 1.53 is 6-10% (rectum V70, mean dose), 27% (D2cc), 32-35% (bladder), 38%
   (bleeding NTCP), all well under the plan shares in Table 7. The claim that the
   consensus plan compresses less than predicted is withdrawn. Section 2.2 is immune,
   because the dose is identical across a row.
2. *Five-cluster bootstrap.* Replaced by patient-level t-intervals. Consequences: the CTV
   no longer compresses less than predicted (1.30 [1.26, 1.36]); the consensus bleeding
   penalty no longer excludes zero (+0.81 [-0.36, +1.98]); the consensus D2cc penalty
   survives (+0.36 [+0.05, +0.68], positive in all five patients).
3. *The consensus rectal penalty is over-claimed.* Tested directly: regressing the
   observer plans' penalty on the rectum volume deficit predicts -0.12 Gy for the
   consensus plan's deficit of 2.4 cc, against +0.36 Gy observed. The penalty is not
   what a smaller planning rectum does, so "the inward bias of majority voting,
   arriving in dose" is withdrawn. Now an exploratory observation, consistent with a
   planning trade-off and confounded with planning order and the lack of blinding.
4. *Mean-based benchmark for an order-statistic operator.* Half right: leave-one-out
   falls 1.34 -> 1.31 as predicted, but the full value rises 1.49 -> 1.53, the opposite
   of the reviewer's direction. The idealised model also gives a 0.12 SD inward
   displacement at 6/10 for unbiased observers.
5. *Estimator deviation.* Declared as post hoc; both estimators reported equally.
6. *Tendency correlation rests on one observer.* r = 0.74 -> 0.36 (p = 0.35) without O5;
   no other single observer moves it outside 0.73-0.82. Withdrawn as a finding.

**Added.** A 5-of-10 consensus control: the displacement vanishes (CTV +0.03 mm, rectum
+0.17 mm, intervals spanning zero) against +0.53 and +0.62 mm at 6/10 in the same
patients. Every threshold above one half displaced the consensus inward; exactly one
half did not.

**Not done.** Repeat planning, which is the only way to measure plan-to-plan variation
and so to test R_2.3 against its benchmark. Recorded as a limitation.

---

## 2026-09-14 - The paper is reframed; the consensus analysis becomes secondary

**Decision (author).** The design cannot answer the original title question, whether
consensus contouring mitigates the dosimetric impact of interobserver variation. The paper
is now about the dosimetric and protocol-compliance consequences of interobserver
variation and the failure of geometric metrics to predict them.

**Structure of `paper/results_sections.tex`.**

| Now | Content | Was |
|---|---|---|
| 1 | Geometric interobserver variation (volumes, agreement) | 1.1-1.2 |
| 2.1 | Self-evaluated results, plan vs contour decomposition | 2.1 |
| **2.2** | **Dosimetric variation with the dose fixed (primary)** | 2.2.1 |
| 2.3 | Variation with the anatomy fixed (context) | 2.3.1 |
| **3** | **Protocol compliance (primary)** | 3 |
| **4** | **Geometry does not predict dose (primary)** | 4 |
| 5 | Consensus contouring, secondary and largely inconclusive | 1.3-1.6, 2.2.2, 2.3.2-2.3.3 |
| S1-S7 | Supplementary tables and Figure S1 | Tables 3-6, 10, 12, 13; Figure 2 |

**Why the apparatus was cut.** The benchmark simulation, the estimator comparison and the
plan-noise reconciliation all concluded that the consensus behaves like a median observer,
and that where it seems not to, the design cannot say why. That conclusion is now stated
in a sentence, with the working in a short supplementary note. The consensus displacement
of 0.3-0.6 mm is sub-voxel on a 1 mm grid with 2.5 mm slices, so it gets a sentence and
Tables S2-S3 rather than a subsection and the M030 natural experiment.

**Removed at the author's direction.** All mention of planner blinding: the planning
process was identical for every contour set.

**Retained, for transparency.** The post-hoc estimator change is still declared, in the
supplementary note, and both estimators remain in Tables S5-S6.

---

## 2026-09-14 - Five corrections from the author's review

1. **The varied-plan column of the decomposition table is not plan quality.** Fixing the
   evaluation contour and varying the plan changes both the contour the plan was optimised
   on and the planner's non-deterministic behaviour. The column is relabelled, "plan
   quality contributes roughly half" is removed, and so is the reconciliation of the
   consensus-plan factors with plan shares, which was circular. Plan-to-plan variation is
   stated as unknown throughout.
2. **"Planning on the correct anatomy is often not best" was the null result.** With ten
   interchangeable plans the correct plan is beaten in 9/10 columns on average, 44.1 of 49;
   44 were observed. Replaced by the correct plan's rank within its column: 0.49-0.52 for
   rectum V70 in every patient (interchangeable), 0.26-0.43 for CTV D98 in every patient
   (detectably better). Planning on the right contour helps coverage of it, not rectal dose.
3. **The plans are not clinical plans.** PTV coverage metrics are removed. Representativeness
   is argued from volumes instead: the 95% isodose volume exceeds the post-hoc PTV volume in
   every plan (median ratio 1.23, range 1.17-1.32). Section 3 now states that rectal flip
   rates are unlikely to be overstated for plans optimised to meet the constraints, while
   the CTV D99 rate probably is for plans that cover a PTV.
4. **Section 4 had p-values on non-independent pairs.** Replaced by within-patient Spearman
   (median and range over patients) with leave-one-patient-out R-squared; the "432 pairs"
   framing is gone. Dice, HD95 and MSD against rectal NTCP estimation error stay below 0.09
   in magnitude in every patient.
5. **The dose-aware metrics are removed from Section 4.** They are close to a partial
   calculation of the target and unavailable before planning. The replanning-benefit target
   is now described as containing plan-to-plan variation, with an unknown predictability
   ceiling; the estimation error is the clean test.

Numbers from `analysis/revision_checks.py`.
