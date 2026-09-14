# Data issue register

Live. Each issue is resolved, waived or open. Waived issues carry their justification in
`config/waivers.yaml` and appear in the methods.

## Resolved

| Issue | Resolution |
|---|---|
| Fractionation unknown | **39 fractions**, read from `FractionGroupSequence.NumberOfFractionsPlanned` in the V027/O9 RTPLAN. |
| `DoseSummationType = BEAM` on all 59 doses | Not a mislabel. Two RTPLANs (V027/O9, M030/O5) show **single-arc VMAT** — one DYNAMIC beam, 89 control points, 352° CW. More decisively, no dose file is a partial delivery: the diagonal CTV D50 of all 54 analysis plans falls in 79.97–80.55 Gy, where one arc of a multi-arc plan would read about half. |
| M030/O3's oversized dose grid (80×106×93) | Not a second arc. Its maximum is 83.12 Gy and its D50 is in band, so the grid is a manually enlarged calculation box. |
| CTV- or PTV-optimised? | **CTV-optimised**; PTVs are post-hoc isotropic expansions (user, 2026-09-04). Consistent with diagonal CTV D50 pinned at 80.25 Gy (SD 0.128) against diagonal PTV D98 floating 73.7–77.4 Gy. |
| Is the observer↔dose folder mapping real? | **Yes.** Permutation test on the within-column z-score of the diagonal: p = 5.0e-05 (the floor at 20 000 permutations) for all five patients, diagonal z +0.69 to +1.22. |
| VOTE's role | Consensus contour, Arm C. STAPLE is the excluded set (user, 2026-09-04). |
| PTV margin, undocumented | **7 mm isotropic**, measured back out of the masks: median CTV-to-PTV surface distance 7.07 mm in all five patients, IQR 1.0 mm, no directional asymmetry. |
| Are absolute-volume endpoints valid? | **Yes, everywhere.** All 2788 (plan x contour set x structure) combinations are 100 % inside their dose grid, so D2cc, V70cc and V60cc are computable for every cell. |
| Is the DVH code right? | Cross-validated against an independently implemented histogram-based estimator on 810 real cells: worst disagreement 0.0100 Gy, exactly the reference's bin width. Both are pinned to closed-form answers on a linear dose ramp. |
| Does resampling change the answer? | No. Native-grid and analysis-grid pipelines agree to 0.080 Gy median, 0.921 Gy worst, excluding V027 where the native path is known to drop a quarter of each structure. |
| Observer anonymisation (Q13) | **O1-O10**, a seeded random permutation; key held privately, applied throughout the results document 2026-09-11. |
| Was the planner blinded to the consensus? | Not blinded, but the planning process was identical for every contour set, so knowing which set was the consensus could not change how it was planned (user, 2026-09-14). **Not a limitation; not raised in the paper.** |
| Consensus algorithm (Q6) | Majority vote in **MILXView** for the parent study (Roach 2018); a strict majority reproduces it (median Dice 0.963). Source: earlier draft, 2026-09-14. |
| What is `PRS`? (Q9) | **Peri-rectal space**, contoured by all observers and excluded from analysis, as were colon and bowel bag. Source: earlier draft. |
| `Rectum2_O7` in K042 (Q11) | A second, **revised** rectum contour from that observer; the original (`Rectum_O7`) is correct, which is what the pipeline used. Source: earlier draft. |
| Observer backgrounds (Q16) | Ten observers from four centres: six radiation oncologists, two medical physicists, one radiation therapist, one radiographer; trainee to highly experienced; five contoured in Eclipse, five in Pinnacle3. Source: earlier draft. |
| Dataset source and planning (Q1-adjacent) | Patients from TROG 03.04 RADAR, selected by image-similarity clustering (Kennedy 2016). Plans: Pinnacle3 Auto-planning, objectives accepted on the consensus plan by a radiation therapist and propagated unchanged to all observer contour sets; VersaHD 6 MV, 80-leaf-pair MLC. Source: earlier draft. |
| Who produced the 59 plans? | **One planner, given every contour set** (user, 2026-09-10). Observers contoured; they did not plan. Planner skill is therefore not confounded with observer identity. The size of plan-to-plan variation is **unknown**: it would need replicate plans on the same contour, and the varied-plan column of the decomposition table mixes it with the planning-contour effect. |

## Waived

| Issue | Scope | Where |
|---|---|---|
| `DoseSummationType = BEAM` | all doses | `waivers.yaml` |
| Plan-referenced structure sets absent | all plans | `waivers.yaml` |
| VOTE/STAPLE carry only 4 of 8 structures | VOTE, STAPLE | `waivers.yaml` |
| `Rectum_O6` missing, no M030/O6 plan | M030/O6 | `waivers.yaml` |
| ~25 % volume lost rasterising onto the native dose grid | V027 | `waivers.yaml` |

### V027 slice mismatch — the one worth understanding

V027's planning CT is **2.0 mm** while every dose grid in the dataset is **2.5 mm**. On
the native dose grid, 8 of 20 CTV contours, 18 of 45 rectum contours and 16 of 42
bladder contours fall between planes and are dropped, costing roughly a quarter of each
structure's volume. The other four patients agree with the independent shoelace volume
to within 0.1–1.0 %.

This affects the Tier 1 mapping test only, which runs on the native grid for speed; the
mapping conclusion is unchanged. **From stage 05 onward everything is resampled onto a
common CT-derived analysis grid**, where no contour is dropped, and the Tier 1 statistic
is recomputed there as confirmation.

## Open

| Issue | Blocks | Question |
|---|---|---|
| Total prescribed dose | EQD2, LKB NTCP, TCP | 78 Gy inferred from legacy ROI names; the RTPLAN gives 39 fractions and 2.05879 Gy/fx at the isocentre (80.29 Gy there), which does not settle it. **Q1** |
| Clinical constraint set | flip-rate analysis | **Q4** |
| NTCP/TCP parameter sets | biological endpoints | **Q5** |
| STAPLE's plan-quality failure | methods statement | nothing anomalous in the dose headers. **Q8** |
| Rectum definition, gas, superior/inferior limits | D2cc, V70 | observers followed a standardised protocol with expert reference contours (Roach 2019, per the earlier draft), but its rectal definition is not stated there; a `RECTUM-GAS` ROI exists in 4/5 patients, clinical set only. **Q10** |
| Dose calculation algorithm | methods | Pinnacle3 16.2/16.2.1 Auto-planning, 2.5 mm grid, heterogeneity correction with density overrides; algorithm name not recorded. **Q17** |
| Code and data availability statement | methods | **Q18** |
| Colon / Bowel_Bag exclusion | scope | 5 to 77 slices for the same patient. **Q12** |
| REC/IRB reference | methods | **Q15** |

## Watch list

| Item | Why |
|---|---|
| `TissueHeterogeneityCorrection` differs between patients | `IMAGE + ROI_OVERRIDE` for K018/M020/M030, `IMAGE` alone for K042/V027 — consistent within patient |
| `PTV_STAPLE` and `PTV78_STAPLE` both exist | STAPLE is excluded, so cosmetic; recorded as a warning |
| `ApprovalStatus = UNAPPROVED` on the RTPLAN | research plans, not clinically approved — state in methods |
| Planning order is identical in every patient | STAPLE, O4, O10, O8, O2, O7, O6, O9, O1, O5, O3, VOTE, from the RTDOSE creation timestamps. Order within a patient is therefore perfectly aliased with observer identity and a within-session learning effect cannot be separated from an observer effect. |
| The consensus plan was always planned last | VOTE is the 12th and final dose file in all five patients, after ten observer plans on the same anatomy. The consensus plans were nonetheless typical in quality (20th-80th percentile on every self-evaluated endpoint), so this is noted rather than interpreted. |
| K018 was planned first and took twice as long | 134 min on 22 Jan against 42-72 min for the other four on 24 Jan, with a 35-minute mid-sequence gap. It also has the largest varied-plan spread for rectum V70 (6.42 pp against 1.84-3.09), rectum mean dose and Paddick CI. That spread mixes planning-contour effects with plan-to-plan variation, so it cannot be attributed to the technique settling. |
| OAR D50 is discretisation-sensitive | worst native-vs-analysis disagreements are all rectum/bladder D50 (up to 1.62 Gy) because an OAR median sits in a shallow dose gradient; relevant to Aim 3, where a metric predicting ΔD50 may be predicting noise |
