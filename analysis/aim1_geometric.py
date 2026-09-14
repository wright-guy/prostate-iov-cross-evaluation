"""Aim 1 figures and tables. Reads results/ only; never touches DICOM.

    python analysis/aim1_geometric.py

Produces F2 (geometric agreement and the compression factor against the
operator-specific benchmarks from stage 14) and T1 (contour volumes and geometric summary).
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

STRUCTURE_ORDER = ["CTV", "PTV", "Rectum", "Bladder"]
FAMILY_LABEL = {
    "obs_obs": "observer\nvs observer",
    "obs_vote": "observer\nvs VOTE",
    "obs_loo": "observer\nvs LOO consensus",
}
FAMILY_COLOUR = {"obs_obs": "#8c8c8c", "obs_vote": "#0c666d", "obs_loo": "#b8860b"}


def figure_f2(geo: pd.DataFrame, comp: pd.DataFrame, out: Path,
              bench_full: float, bench_loo: float) -> Path:
    structures = [s for s in STRUCTURE_ORDER if s in geo["structure"].unique()]
    fig, axes = plt.subplots(2, len(structures), figsize=(3.6 * len(structures), 7.6))
    if len(structures) == 1:
        axes = axes.reshape(2, 1)

    # --- top row: the agreement distributions themselves
    for col, structure in enumerate(structures):
        ax = axes[0, col]
        sub = geo[geo["structure"] == structure]
        data, colours, labels = [], [], []
        for family in ("obs_obs", "obs_vote", "obs_loo"):
            vals = sub.loc[sub["family"] == family, "dsc"].dropna()
            if vals.empty:
                continue
            data.append(vals.to_numpy())
            colours.append(FAMILY_COLOUR[family])
            labels.append(FAMILY_LABEL[family])
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, colour in zip(bp["boxes"], colours, strict=True):
            patch.set_facecolor(colour)
            patch.set_alpha(0.55)
            patch.set_edgecolor(colour)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(structure, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("Dice similarity coefficient", fontsize=9)

    # --- bottom row: compression factor with patient-level t intervals
    headline = comp[comp["headline"]]
    for col, structure in enumerate(structures):
        ax = axes[1, col]
        sub = headline[headline["structure"] == structure]
        metrics = sorted(sub["metric"].unique())
        x = np.arange(len(metrics))
        width = 0.36
        for offset, (label, colour, bench) in zip(
            (-width / 2, width / 2),
            (("uncorrected", "#0c666d", bench_full), ("loo_corrected", "#b8860b", bench_loo)),
            strict=True,
        ):
            rows = sub[sub["comparison"] == label].set_index("metric").reindex(metrics)
            point = rows["compression_factor"].to_numpy(dtype=float)
            lo = rows["ci_lo"].to_numpy(dtype=float)
            hi = rows["ci_hi"].to_numpy(dtype=float)
            ax.errorbar(
                x + offset, point,
                yerr=np.vstack([np.clip(point - lo, 0, None), np.clip(hi - point, 0, None)]),
                fmt="o", color=colour, capsize=3, markersize=5, linewidth=1.4,
                label=f"vs {'VOTE' if label == 'uncorrected' else 'LOO consensus'}",
            )
            ax.axhline(bench, color=colour, linestyle="--", linewidth=1.0, alpha=0.7)

        ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_mm", "").upper() for m in metrics], fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0.9, max(2.1, float(np.nanmax(headline["ci_hi"])) + 0.1))
        if col == 0:
            ax.set_ylabel("Compression factor", fontsize=9)
            ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
        if col == len(structures) - 1:
            ax.text(0.98, bench_full, f"{bench_full:.2f}", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7, color="#0c666d")
            ax.text(0.98, bench_loo, f"{bench_loo:.2f}", transform=ax.get_yaxis_transform(),
                    ha="right", va="top", fontsize=7, color="#b8860b")

    fig.suptitle(
        "Geometric agreement and its compression toward a consensus contour\n"
        "dashed lines: benchmarks for the majority-vote operator; bars are 95% t-intervals "
        "on per-patient estimates (4 df)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def table_t1(structures: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """Cohort contour volumes and observer agreement, per structure."""
    obs = structures[
        structures["present"]
        & structures["contour_set"].isin(load_config().observers)
        & structures["structure"].isin(STRUCTURE_ORDER)
    ]
    vol = obs.groupby("structure")["volume_cc"].agg(
        n="count", median="median", q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75), min="min", max="max",
    )
    pair = geo[geo["family"] == "obs_obs"].groupby("structure").agg(
        dsc_median=("dsc", "median"),
        dsc_q1=("dsc", lambda s: s.quantile(0.25)),
        dsc_q3=("dsc", lambda s: s.quantile(0.75)),
        msd_median=("msd_mm", "median"),
        hd95_median=("hd95_mm", "median"),
    )
    return vol.join(pair).reindex([s for s in STRUCTURE_ORDER if s in vol.index])


def main() -> int:
    cfg = load_config()
    r = cfg.results_root
    geo = pd.read_parquet(r / "geometric_long.parquet")
    # Patient-level estimates and operator-specific benchmarks from stage 14.
    inf = pd.read_parquet(r / "inference_s1_compression.parquet")
    comp = inf.rename(columns={"estimate": "compression_factor", "lo": "ci_lo",
                               "hi": "ci_hi"})
    comp["comparison"] = comp["reference"].map({"full": "uncorrected",
                                                "loo": "loo_corrected"})
    comp["headline"] = True
    bench = pd.read_parquet(r / "inference_benchmarks.parquet").set_index("estimator")
    structures = pd.read_parquet(r / "structures.parquet")

    out = Path(__file__).resolve().parents[1] / "figures"
    f2 = figure_f2(geo, comp, out / "F2_geometric_compression.png",
                   bench_full=float(bench.loc["population_full", "benchmark"]),
                   bench_loo=float(bench.loc["population_loo", "benchmark"]))
    print(f"wrote {f2}")

    t1 = table_t1(structures, geo)
    t1.to_csv(r / "T1_cohort_geometry.csv")
    pd.set_option("display.width", 220)
    print("\n=== T1: cohort contour volumes and observer agreement ===")
    print(t1.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
