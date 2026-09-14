"""Tests for config resolution, the design matrix, the QC gate and the mapping test."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import load_config  # noqa: E402
from mcx.qc.checks import Finding, GateFailure, QCLog, Status, enforce  # noqa: E402
from mcx.qc.identifiability import assess, diagonal_statistic, permutation_test  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# -------------------------------------------------------------------- config


def test_observer_key_maps_labels_to_raw_names(cfg):
    """With a key, labels are used everywhere and raw names only at the DICOM boundary."""
    keyed = replace(cfg, observer_key_raw={"observers": {"O2": "XyZ", "O9": "Qp"},
                                           "folder_alias": {"XYZ": "XyZ"}})
    assert keyed.raw_set("O2") == "XyZ"
    assert keyed.raw_set("VOTE") == "VOTE"
    assert keyed.canonical_set("XYZ") == "O2"  # aliased folder -> raw name -> label
    assert keyed.canonical_set("Qp") == "O9"
    assert keyed.canonical_set("xyz") == "xyz"  # still not case-folded
    assert keyed.dose_folders("O2") == ["XyZ", "XYZ"]
    assert keyed.structures["Rectum"].roi_template.format(set="Qp") == keyed.roi_name(
        "K018", "O9", "Rectum")


def test_vote_is_a_plan_but_never_a_truth(cfg):
    assert cfg.consensus_set == "VOTE"
    assert "VOTE" in cfg.analysis_sets
    assert "VOTE" not in cfg.truth_sets
    assert "STAPLE" not in cfg.analysis_sets


def test_arm_assignment(cfg):
    assert cfg.arm("O4", "O4") == "diagonal"
    assert cfg.arm("O4", "O10") == "S"
    assert cfg.arm("VOTE", "O10") == "C"


def test_design_matrix_is_530_cells(cfg):
    """The corrected primary design: 432 Arm S + 49 diagonal + 49 Arm C."""
    cells = cfg.design_cells()
    counts = pd.Series([c[3] for c in cells]).value_counts()
    assert counts["S"] == 432
    assert counts["diagonal"] == 49
    assert counts["C"] == 49
    assert len(cells) == 530


def test_m030_mae_excluded_everywhere(cfg):
    assert cfg.is_known_absent("M030", "O6") is not None
    assert cfg.is_known_absent("K018", "O6") is None
    assert all(not (p == "M030" and "O6" in (i, j)) for p, i, j, _ in cfg.design_cells())


def test_roi_name_honours_overrides_and_exceptions(cfg):
    assert cfg.roi_name("K018", "O9", "Rectum") == "Rectum_O9"
    assert cfg.roi_name("K042", "O7", "Rectum") == "Rectum_O7"  # not Rectum2_O7
    assert cfg.roi_name("K018", "STAPLE", "PTV") == "PTV78_STAPLE"


def test_derived_cache_is_not_inside_the_repo_data_dir(cfg):
    assert "data" not in cfg.derived_root.parts[-3:]


# ----------------------------------------------------------------- QC gate


def test_gate_raises_on_unwaived_failure(cfg):
    log = QCLog(cfg, "test")
    log.check("made.up.check", False, "boom")
    with pytest.raises(GateFailure, match="made.up.check"):
        enforce(log)


def test_waiver_downgrades_failure_to_waived(cfg):
    log = QCLog(cfg, "test")
    f = log.check(
        "dose.summation_type_is_plan", False, "BEAM",
        patient="K018", contour_set="O5",
    )
    assert f.status is Status.WAIVED
    assert "single-arc" in (f.waiver_reason or "")
    enforce(log)  # must not raise


def test_scoped_waiver_does_not_leak_to_other_scopes(cfg):
    log = QCLog(cfg, "test")
    covered = log.add(
        Finding("struct.roi_present", Status.FAIL, "missing", 0,
                patient="M030", contour_set="O6", structure="Rectum")
    )
    other = log.add(
        Finding("struct.roi_present", Status.FAIL, "missing", 0,
                patient="K018", contour_set="O6", structure="Rectum")
    )
    assert covered.status is Status.WAIVED
    assert other.status is Status.FAIL


def test_warn_does_not_trip_the_gate(cfg):
    log = QCLog(cfg, "test")
    log.check("some.check", False, "cosmetic", warn_only=True)
    enforce(log)


# --------------------------------------------------- mapping identifiability


def test_diagonal_statistic_is_zero_for_a_flat_matrix():
    m = np.tile(np.arange(5.0), (5, 1))  # every column constant across plans
    assert np.isnan(diagonal_statistic(m))


def test_permutation_test_detects_a_strong_diagonal():
    rng = np.random.default_rng(0)
    m = rng.normal(70, 2, size=(10, 10))
    np.fill_diagonal(m, 90.0)
    stat, p = permutation_test(m, n_permutations=2000)
    assert stat > 2.0
    assert p < 0.001


def test_permutation_test_does_not_fire_on_noise():
    rng = np.random.default_rng(1)
    m = rng.normal(70, 2, size=(10, 10))
    _, p = permutation_test(m, n_permutations=2000)
    assert p > 0.01


def test_permutation_test_ignores_a_dominant_row():
    """A uniformly good plan must not be mistaken for a correct labelling.

    This is the failure mode that broke the first, row-oriented version of the test:
    the observer with the smallest PTV scored highest under every plan.
    """
    rng = np.random.default_rng(2)
    m = rng.normal(70, 2, size=(10, 10))
    m[3, :] += 20.0  # one plan beats everything, unrelated to the diagonal
    _, p = permutation_test(m, n_permutations=2000)
    assert p > 0.01


def test_assess_reports_ranks_and_margins():
    sets = ["A", "B", "C"]
    values = [
        {"plan_set": i, "eval_set": j, "D98": 90.0 if i == j else 70.0}
        for i in sets
        for j in sets
    ]
    res = assess(pd.DataFrame(values), patient="P", metric="D98", n_permutations=500)
    assert res.n_columns_won == 3
    assert res.worst_margin_gy == 0.0
    assert set(res.per_column["diagonal_rank"]) == {1}


# ------------------------------------------------------ protocol constraints


def test_prescription_is_confirmed(cfg):
    assert cfg.prescription_gy == 78.0
    assert cfg.n_fractions == 39
    assert cfg.dataset["prescription"]["unconfirmed"] is False


def test_unavailable_constraints_are_kept_with_a_reason(cfg):
    """They are excluded from analysis but must still be nameable in the methods."""
    unavailable = [c for c in cfg.constraints if c["applicability"] == "unavailable"]
    assert {c["id"] for c in unavailable} == {
        "ant_prostate_d90", "rectum_post_wall_dmax", "rectum_prw_slices",
        "urethra_prv_v82", "femoral_heads_v40", "sv_d90",
    }
    assert all(c.get("reason") for c in unavailable)
    assert all(c not in cfg.computable_constraints for c in unavailable)


def test_lower_is_better_tiering(cfg):
    c = next(x for x in cfg.constraints if x["id"] == "rectum_v70")
    assert cfg.constraint_tier(c, 9.99) == "per_protocol"
    assert cfg.constraint_tier(c, 10.0) == "minor"
    assert cfg.constraint_tier(c, 19.99) == "minor"
    assert cfg.constraint_tier(c, 20.0) == "major"


def test_higher_is_better_tiering(cfg):
    c = next(x for x in cfg.constraints if x["id"] == "ptv_d95")
    assert cfg.constraint_tier(c, 78.0) == "per_protocol"
    assert cfg.constraint_tier(c, 77.0) == "minor"
    assert cfg.constraint_tier(c, 76.2) == "minor"
    assert cfg.constraint_tier(c, 76.19) == "major"


def test_two_sided_constraint_names_under_dosing(cfg):
    """PTV D50's protocol band is 78-81.5 Gy; below 78 is not tiered by the protocol."""
    c = next(x for x in cfg.constraints if x["id"] == "ptv_d50")
    assert cfg.constraint_tier(c, 77.9) == "under_dose"
    assert cfg.constraint_tier(c, 80.0) == "per_protocol"
    assert cfg.constraint_tier(c, 81.6) == "minor"
    assert cfg.constraint_tier(c, 82.1) == "major"


def test_missing_value_is_undefined_not_a_pass(cfg):
    c = next(x for x in cfg.constraints if x["id"] == "rectum_v70")
    assert cfg.constraint_tier(c, float("nan")) == "undefined"


def test_composite_structures_are_evaluation_side_only(cfg):
    """Mixing j's rectum with i's PTV would confound the H5 separation."""
    assert cfg.constraints_raw["composite_structure_rule"] == "evaluation_side_only"


# ------------------------------------------------------- pre-registration guards


def test_every_config_file_parses(cfg):
    """A YAML syntax error in a config must fail here, not halfway through a run."""
    import yaml

    from mcx.config import CONFIG_DIR

    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        with path.open(encoding="utf-8") as fh:
            assert yaml.safe_load(fh) is not None, path.name


def test_exactly_eight_primary_metrics():
    """Section 7.4 says 'about 8, no more'. The primary grid is what FDR corrects."""
    import yaml

    from mcx.config import CONFIG_DIR

    with (CONFIG_DIR / "metrics.yaml").open(encoding="utf-8") as fh:
        m = yaml.safe_load(fh)
    assert len(m["primary"]) == 8
    classes = [x["class"] for x in m["primary"]]
    assert classes.count("unsigned") == 3
    assert classes.count("signed") == 2
    assert classes.count("dose_aware") == 3
    # Every metric must be in exactly one list.
    ids = [x["id"] for x in m["primary"]] + [x["id"] for x in m["secondary"]]
    assert len(ids) == len(set(ids))
    assert m["derive_thresholds"] is False


def test_analysis_plan_cell_counts_match_the_design(cfg):
    """The protocol quotes these numbers; the code must agree with them."""
    import yaml

    from mcx.config import CONFIG_DIR

    with (CONFIG_DIR / "analysis.yaml").open(encoding="utf-8") as fh:
        a = yaml.safe_load(fh)
    arms = a["design"]["arms"]
    realised = pd.Series([c[3] for c in cfg.design_cells()]).value_counts()
    assert arms["S"]["n_cells"] == realised["S"] == 432
    assert arms["C"]["n_cells"] == realised["C"] == 49
    assert arms["diagonal"]["n_cells"] == realised["diagonal"] == 49
    assert a["design"]["primary_total"] == len(cfg.design_cells()) == 530


def test_spread_measure_is_not_sd():
    """Dose response is asymmetric and saturating; SD would misrepresent it."""
    import yaml

    from mcx.config import CONFIG_DIR

    with (CONFIG_DIR / "analysis.yaml").open(encoding="utf-8") as fh:
        a = yaml.safe_load(fh)
    assert a["compression"]["spread_measures"]["primary"] == "iqr"
    assert "sd" not in a["compression"]["spread_measures"]["secondary"]


def test_bootstrap_clusters_on_patients():
    """Treating 530 cells as independent would understate every interval."""
    import yaml

    from mcx.config import CONFIG_DIR

    with (CONFIG_DIR / "analysis.yaml").open(encoding="utf-8") as fh:
        a = yaml.safe_load(fh)
    assert a["uncertainty"]["cluster_unit"] == "patient"
    assert a["uncertainty"]["n_resamples"] >= 10000
    assert isinstance(a["uncertainty"]["seed"], int)


def test_endpoint_ids_are_unique():
    import yaml

    from mcx.config import CONFIG_DIR

    with (CONFIG_DIR / "endpoints.yaml").open(encoding="utf-8") as fh:
        e = yaml.safe_load(fh)
    ids = [x["id"] for x in e["target"]["endpoints"]]
    for oar in e["oars"]:
        ids += [x["id"] for x in oar["relative"]] + [x["id"] for x in oar["absolute"]]
    ids += [x["id"] for x in e["biological"]]
    assert len(ids) == len(set(ids))
    assert set(e["primary_for_prediction"]) <= set(ids)
