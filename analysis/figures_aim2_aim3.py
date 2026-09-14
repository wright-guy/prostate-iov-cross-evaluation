"""Figures F3-F8. Reads results/ only.

    python analysis/figures_aim2_aim3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcx.config import load_config  # noqa: E402

TEAL, GOLD, GREY, RED = "#0c666d", "#b8860b", "#8c8c8c", "#a3312a"
OUT = Path(__file__).resolve().parents[1] / "figures"

HEADLINE = ["ctv_d98", "ladder_d98_3mm", "ladder_d98_5mm", "paddick_ci",
            "rectum_v70pct", "rectum_v65pct", "rectum_d2cc", "rectum_dmean",
            "bladder_v70pct", "bladder_dmean"]
PRETTY = {
    "ctv_d98": "CTV D98", "ladder_d98_3mm": "CTV+3mm D98",
    "ladder_d98_5mm": "CTV+5mm D98", "paddick_ci": "Paddick CI",
    "rectum_v70pct": "Rectum V70", "rectum_v65pct": "Rectum V65",
    "rectum_d2cc": "Rectum D2cc", "rectum_dmean": "Rectum Dmean",
    "bladder_v70pct": "Bladder V70", "bladder_dmean": "Bladder Dmean",
    "ntcp_rectum_relative": "NTCP rectum",
}


def f3_heatmaps(ep: pd.DataFrame) -> Path:
    """Cross-evaluation matrices for three endpoints, diagonal highlighted."""
    cfg = load_config()
    patient = "K018"
    endpoints = ["ctv_d98", "rectum_v70pct", "paddick_ci"]
    order = [*cfg.observers, "VOTE"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    for ax, endpoint in zip(axes, endpoints, strict=True):
        sub = ep[(ep["patient"] == patient) & (ep["endpoint"] == endpoint)]
        wide = sub.pivot_table(index="plan_set", columns="truth_set", values="value")
        rows = [o for o in order if o in wide.index]
        cols = [o for o in cfg.observers if o in wide.columns]
        m = wide.loc[rows, cols].to_numpy(dtype=float)
        im = ax.imshow(m, cmap="viridis", aspect="auto")
        for r, plan in enumerate(rows):
            for c, truth in enumerate(cols):
                if plan == truth:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                               edgecolor="white", linewidth=2.0))
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=7)
        ax.set_xlabel("evaluation contour  j", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("plan optimised on  i", fontsize=8)
        ax.set_title(PRETTY.get(endpoint, endpoint), fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"F3  Cross-evaluation matrices, {patient}. "
                 "White cells are the perfect-information diagonal; the bottom row is "
                 "the consensus plan.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = OUT / "F3_cross_evaluation_heatmaps.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p


def f4_compression(comp: pd.DataFrame, geo_benchmark: float = 1.34) -> Path:
    sub = comp[(comp["spread_measure"] == "iqr") & comp["endpoint"].isin(HEADLINE)]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    endpoints = [e for e in HEADLINE if e in set(sub["endpoint"])]
    x = np.arange(len(endpoints))
    for offset, basis, colour, label in ((-0.18, "error", TEAL, "error basis"),
                                         (0.18, "e2", GOLD, "E2 basis")):
        rows = sub[sub["basis"] == basis].set_index("endpoint").reindex(endpoints)
        pt = rows["compression_factor"].to_numpy(dtype=float)
        lo = rows["ci_lo"].to_numpy(dtype=float)
        hi = rows["ci_hi"].to_numpy(dtype=float)
        ax.errorbar(x + offset, pt,
                    yerr=np.vstack([np.clip(pt - lo, 0, None), np.clip(hi - pt, 0, None)]),
                    fmt="o", color=colour, capsize=3, markersize=6, linewidth=1.5,
                    label=label)
    ax.axhline(geo_benchmark, color=RED, linestyle="--", linewidth=1.3,
               label=f"geometric benchmark {geo_benchmark}")
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.6)
    ax.text(len(endpoints) - 0.4, 1.02, "no compression", fontsize=8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(e, e) for e in endpoints], rotation=30, ha="right",
                       fontsize=9)
    ax.set_ylabel("compression factor  spread(Arm S) / spread(Arm C)", fontsize=9)
    ax.set_ylim(0, max(3.0, float(np.nanmax(sub["ci_hi"])) * 1.05))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("F4  Does the geometric benefit of consensus reach dose?\n"
                 "95% cluster-bootstrap intervals over patients", fontsize=11)
    fig.tight_layout()
    p = OUT / "F4_compression_by_endpoint.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p


def f5_f6_variance(var: pd.DataFrame) -> Path:
    sub = var[var["endpoint"].isin(HEADLINE)].set_index("endpoint").reindex(
        [e for e in HEADLINE if e in set(var["endpoint"])])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    y = np.arange(len(sub))
    axes[0].barh(y, sub["pct_plan"], color=TEAL, label="planning contour  i")
    axes[0].barh(y, sub["pct_truth"], left=sub["pct_plan"], color=GOLD,
                 label="evaluation contour  j")
    axes[0].barh(y, sub["pct_interaction"],
                 left=sub["pct_plan"] + sub["pct_truth"], color=GREY,
                 label="interaction")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([PRETTY.get(e, e) for e in sub.index], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("share of within-patient variance (%)", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title("F5  Variance components", fontsize=11, fontweight="bold")

    pt = sub["truth_over_plan"].to_numpy(dtype=float)
    lo = sub["ci_lo"].to_numpy(dtype=float)
    hi = sub["ci_hi"].to_numpy(dtype=float)
    colours = [GOLD if l > 1 else TEAL for l in lo]
    axes[1].errorbar(pt, y, xerr=np.vstack([np.clip(pt - lo, 0, None),
                                            np.clip(hi - pt, 0, None)]),
                     fmt="none", ecolor="black", capsize=3, linewidth=1.2)
    axes[1].scatter(pt, y, c=colours, s=55, zorder=3)
    axes[1].axvline(1.0, color=RED, linestyle="--", linewidth=1.3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xscale("log")
    axes[1].set_xlabel("evaluation-side / planning-side variance", fontsize=9)
    axes[1].set_title("F6  Which side of the design drives it?", fontsize=11,
                      fontweight="bold")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].set_axisbelow(True)
    fig.suptitle("Arm S cross-evaluation matrices, decomposed within patient. "
                 "Right of the dashed line, whose contour scores the plan matters more "
                 "than which plan was made.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = OUT / "F5_F6_variance_components.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p


def f7_flip_rates(flips: pd.DataFrame, bio: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    s = flips[flips["arm"] == "S"].copy()
    s = s[s["ordinal_flip_across_truths"] > 0].sort_values("ordinal_flip_across_truths")
    y = np.arange(len(s))
    axes[0].barh(y, 100 * s["ordinal_flip_across_truths"], color=TEAL,
                 label="tier changes")
    axes[0].barh(y, 100 * s["binary_flip_across_truths"], color=GOLD, height=0.45,
                 label="per-protocol status changes")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(s["label"], fontsize=8)
    axes[0].set_xlabel("% of delivered plans whose compliance depends on the "
                       "scoring contour", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title("F7a  Constraint flip rates, Arm S", fontsize=11, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].set_axisbelow(True)

    n = bio[(bio["model"] == "ntcp") & (bio["structure"] == "Rectum")
            & (bio["variant"] == "relative")]
    per_plan = n.groupby(["patient", "plan_set"])["value"].agg(["min", "max", "median"])
    per_plan = per_plan.sort_values("median").reset_index()
    x = np.arange(len(per_plan))
    axes[1].vlines(x, 100 * per_plan["min"], 100 * per_plan["max"],
                   color=GREY, linewidth=1.6)
    axes[1].scatter(x, 100 * per_plan["median"], color=TEAL, s=18, zorder=3)
    axes[1].set_xlabel("individual plans, ordered by median predicted risk", fontsize=9)
    axes[1].set_ylabel("predicted late rectal toxicity (%)", fontsize=9)
    axes[1].set_title("F7b  Predicted risk for one delivered dose,\n"
                      "across the ten plausible rectum contours", fontsize=11,
                      fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)
    fig.tight_layout()
    p = OUT / "F7_flip_rates_and_risk.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p


def f8_prediction(pred: pd.DataFrame) -> Path:
    sub = pred[pred["target"] == "replanning_benefit"]
    endpoints = sorted(sub["endpoint"].unique())
    fig, axes = plt.subplots(1, len(endpoints), figsize=(6.4 * len(endpoints), 5.4))
    axes = np.atleast_1d(axes)
    colour = {"unsigned": GREY, "signed": TEAL, "dose_aware": GOLD}
    for ax, endpoint in zip(axes, endpoints, strict=True):
        g = sub[sub["endpoint"] == endpoint].copy()
        g["abs_rho"] = g["spearman_rho"].abs()
        g = g.sort_values("abs_rho")
        y = np.arange(len(g))
        ax.barh(y, g["abs_rho"], color=[colour.get(c, GREY) for c in g["metric_class"]])
        for i, (_, row) in enumerate(g.iterrows()):
            if row["significant_fdr"]:
                ax.text(row["abs_rho"] + 0.012, i, "*", va="center", fontsize=11)
        ax.set_yticks(y)
        ax.set_yticklabels(g["metric"], fontsize=8)
        ax.set_xlabel("|Spearman rho| against the replanning benefit", fontsize=9)
        ax.set_xlim(0, max(0.85, float(g["abs_rho"].max()) * 1.18))
        ax.set_title(PRETTY.get(endpoint, endpoint), fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[c])
               for c in ("unsigned", "signed", "dose_aware")]
    axes[0].legend(handles, ["unsigned", "signed", "dose-aware"], fontsize=8,
                   loc="lower right")
    fig.suptitle("F8  Can a contour metric predict dosimetric consequence?  "
                 "Within-patient rank correlation; * survives FDR at q = 0.05",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = OUT / "F8_metric_prediction.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p


def main() -> int:
    cfg = load_config()
    r = cfg.results_root
    OUT.mkdir(parents=True, exist_ok=True)
    made = [
        f3_heatmaps(pd.read_parquet(r / "endpoints_long.parquet")),
        f4_compression(pd.read_parquet(r / "aim2_compression.parquet")),
        f5_f6_variance(pd.read_parquet(r / "aim2_variance_components.parquet")),
        f7_flip_rates(pd.read_parquet(r / "aim2_flip_rates.parquet"),
                      pd.read_parquet(r / "bio_long.parquet")),
        f8_prediction(pd.read_parquet(r / "aim3_prediction.parquet")),
    ]
    for p in made:
        print(f"wrote {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
