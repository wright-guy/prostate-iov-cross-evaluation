"""Figures 1-5 of the manuscript, written to paper/figures/.

    python analysis/manuscript_figures.py

Figure 1 is a schematic. Figures 2, 4 and 5 read results/ only. Figure 3 also reads the planning CT, the cached
analysis-grid masks and the cached resampled dose for one patient, so it needs data/ and
the derived cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mcx.config import load_config  # noqa: E402
from mcx.dose.store import DoseStore  # noqa: E402
from mcx.geom.analysis_grid import load_grid  # noqa: E402
from mcx.geom.masks import MaskStore  # noqa: E402
from mcx.io.ct import load_ct_series  # noqa: E402

OUT = REPO / "paper" / "figures"
cfg = load_config()
RES = cfg.results_root

FULL_WIDTH = 6.9  # inches, two-column page width
TIER_COLOURS = {"per_protocol": "#dcefe0", "minor": "#fbeccd", "major": "#f6d5d2"}
PATIENT_COLOURS = dict(zip(cfg.patients, ["#0c666d", "#b8860b", "#7b3294", "#a3312a", "#4d4d4d"]))

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def observer_labels() -> dict[str, str]:
    """Contour-set id -> anonymised label (O1-O10).

    Where the contour sets are already labelled O1-O10 (the public repository), each id
    is its own label; otherwise the labels come from the private observer key.
    """
    if all(s[0] == "O" and s[1:].isdigit() for s in cfg.observers):
        return {s: s for s in cfg.observers}
    labels = {}
    for line in (REPO / "docs" / "OBSERVER_KEY.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 2 and cells[0].startswith("O") and cells[0][1:].isdigit():
            labels[cells[1]] = cells[0]
    assert sorted(labels) == sorted(cfg.observers), "observer key does not match config"
    return labels


LABEL = observer_labels()
BY_LABEL = sorted(cfg.observers, key=lambda s: int(LABEL[s][1:]))  # ids in O1..O10 order


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def observer_cells(endpoints: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Observer plans evaluated on observer contours (diagonal and off-diagonal)."""
    e = endpoints[(endpoints.endpoint == endpoint) & endpoints.plan_set.isin(cfg.observers)
                  & endpoints.truth_set.isin(cfg.observers)]
    return e[["patient", "plan_set", "truth_set", "value"]]


# ============================================================================ Figure 1
def figure_schematic() -> None:
    """Data-free schematic of the cross-evaluation design (Methods)."""
    n, i, j = 5, 1, 3  # grid size and the example plan i and contour j (0-based)
    row_c, col_c, cell_c = "#f4c7c1", "#c6dbef", "#0c666d"
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.1))

    def cell(r, c, face, text="", tcolor="black", hatch=None):
        ax.add_patch(Rectangle((c, -r - 1), 1, 1, fc=face, ec="0.35", lw=0.8, hatch=hatch))
        if text:
            ax.text(c + 0.5, -r - 0.5, text, ha="center", va="center", fontsize=7.5,
                    color=tcolor)

    for r in range(n):
        for c in range(n):
            face = "white"
            if r == i:
                face = row_c
            if c == j:
                face = col_c if r != i else face
            cell(r, c, face)
    for k in range(n):
        cell(k, k, "0.82")
    cell(i, i, "0.82", r"$E_{i,i}$")
    cell(j, j, "0.82", r"$E_{j,j}$")
    cell(i, j, cell_c, r"$E_{i,j}$", tcolor="white")
    # consensus plan row, set apart below the observer plans
    for c in range(n):
        cell(n + 0.35, c, "#fbeccd")
    ax.text(-0.2, -n - 0.85, "C", ha="right", va="center", fontsize=8)

    labels = ["1", r"$i$", r"$\cdots$", r"$j$", r"$n$"]
    for k, lab in enumerate(labels):
        ax.text(k + 0.5, 0.25, lab, ha="center", va="bottom", fontsize=8)
        ax.text(-0.2, -k - 0.5, lab, ha="right", va="center", fontsize=8)
    ax.text(n / 2, 0.95, "Observer contour set used for evaluation", ha="center", fontsize=8)
    ax.text(-0.95, -n / 2, "Plan, by the contour set\nit was optimised on", ha="center",
            va="center", rotation=90, fontsize=8)

    x0 = n + 0.9
    notes = [
        (row_c, "Row $i$: one delivered plan evaluated on every observer's contours. "
                "The dose is fixed and only the contour varies.\n"
                r"Estimation error: $E_{i,j} - E_{i,i}$"),
        (col_c, "Column $j$: every observer plan evaluated on one contour set. "
                "The anatomy is fixed and the plan varies.\n"
                r"Replanning benefit: $E_{j,j} - E_{i,j}$"),
        ("0.82", "Diagonal: each plan evaluated on the contour it was optimised on "
                 "(self-evaluation)."),
        ("#fbeccd", "C: the consensus plan, evaluated on every observer's contours."),
    ]
    y = -0.1
    import textwrap
    for face, text in notes:
        ax.add_patch(Rectangle((x0, y - 0.55), 0.45, 0.45, fc=face, ec="0.35", lw=0.8))
        wrapped = "\n".join(textwrap.fill(part, 62) for part in text.split("\n"))
        ax.text(x0 + 0.65, y - 0.1, wrapped, va="top", fontsize=7.5)
        y -= 0.45 + 0.42 * (wrapped.count("\n") + 1)
    ax.set_xlim(-1.4, x0 + 7.2)
    ax.set_ylim(-n - 1.55, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "fig1_design_schematic")


# ============================================================================ Figure 2
def figure_matrices(ep: pd.DataFrame, patient: str = "K018") -> None:
    """Cross-evaluation matrices for one patient: CTV D98 and rectum V70 (Results)."""
    panels = [("ctv_d98", r"CTV $D_{98}$ (Gy)"), ("rectum_v70pct", r"Rectum $V_{70}$ (%)")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.2))
    for ax, (endpoint, title), letter in zip(axes, panels, "ab", strict=True):
        cells = observer_cells(ep, endpoint)
        wide = cells[cells.patient == patient].pivot(index="plan_set", columns="truth_set",
                                                     values="value")
        ids = [s for s in BY_LABEL if s in wide.index]
        m = wide.loc[ids, ids].to_numpy(float)
        im = ax.imshow(m, cmap="viridis", aspect="equal")
        for k in range(len(ids)):
            ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec="white", lw=1.4))
        ticks = [LABEL[s] for s in ids]
        ax.set_xticks(range(len(ids)), ticks, rotation=90)
        ax.set_yticks(range(len(ids)), ticks)
        ax.set_xlabel("Contour set used for evaluation ($j$)")
        ax.set_ylabel("Contour set the plan was optimised on ($i$)")
        ax.set_title(f"({letter}) {title}", loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "fig2_cross_evaluation_matrices")


# ============================================================================ Figure 2
def ct_on_grid(patient: str, grid) -> np.ndarray:  # noqa: ANN001
    """Planning CT (HU) interpolated onto the analysis grid, shape (nz, ny, nx)."""
    ct = load_ct_series(cfg, patient)
    xs = grid.origin[0] + np.arange(grid.shape[2]) * grid.spacing[0]
    ys = grid.origin[1] + np.arange(grid.shape[1]) * grid.spacing[1]
    cols = (xs - ct.origin_xy[0]) / ct.pixel_spacing[0]
    rows = (ys - ct.origin_xy[1]) / ct.pixel_spacing[1]
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    by_z = {round(z, 2): p for z, p in zip(ct.z, ct.paths, strict=True)}
    out = np.full(grid.shape, -1000.0, dtype=np.float32)
    for k, z in enumerate(grid.z):
        path = by_z.get(round(float(z), 2))
        if path is None:
            continue
        ds = pydicom.dcmread(path)
        hu = ds.pixel_array * float(getattr(ds, "RescaleSlope", 1)) + float(
            getattr(ds, "RescaleIntercept", 0))
        out[k] = ndimage.map_coordinates(hu.astype(np.float32), [rr, cc], order=1, cval=-1000)
    return out


def figure2(ep: pd.DataFrame, patient: str = "K018", plan: str = "O10") -> None:
    """One delivered plan, ten rectum contours: anatomy and the resulting DVHs."""
    grid = load_grid(cfg, patient)
    masks, doses = MaskStore(cfg), DoseStore(cfg)
    ids = [s for s in BY_LABEL if masks.has(patient, s, "Rectum")]
    rect = {s: masks.get(patient, s, "Rectum") for s in ids}
    ctv = masks.get(patient, "VOTE", "CTV")
    dose, _ = doses.get(patient, plan)
    ct = ct_on_grid(patient, grid)
    colours = dict(zip(ids, plt.cm.tab10(np.linspace(0, 1, 10))))

    # Slices through the consensus CTV centroid.
    kz, ky, kx = (int(round(v)) for v in ndimage.center_of_mass(ctv))
    union = np.any(np.stack(list(rect.values())), axis=0) | ctv
    zz, yy, xx = np.nonzero(union)
    pad = 12
    y0, y1 = max(yy.min() - pad, 0), min(yy.max() + pad, grid.shape[1] - 1)
    x0, x1 = max(xx.min() - pad, 0), min(xx.max() + pad, grid.shape[2] - 1)
    z0, z1 = max(zz.min() - 3, 0), min(zz.max() + 3, grid.shape[0] - 1)
    dx, dy, dz = grid.spacing

    # Width ratios give the axial and sagittal images the same physical height on the page.
    aspect_ax = (x1 - x0) * dx / ((y1 - y0) * dy)
    aspect_sa = (y1 - y0) * dy / ((z1 - z0) * dz)
    fig = plt.figure(figsize=(FULL_WIDTH, 2.75))
    gs = fig.add_gridspec(1, 4, width_ratios=[aspect_ax, aspect_sa, 0.22, 1.3], wspace=0.08)
    ax_a, ax_s, ax_d = (fig.add_subplot(gs[0, k]) for k in (0, 1, 3))

    def draw(ax, img, extent, pieces, iso, title):  # noqa: ANN001
        ax.imshow(img, cmap="gray", vmin=-160, vmax=240, extent=extent, origin="upper",
                  interpolation="bilinear")
        for s, piece in pieces.items():
            if piece.any():
                ax.contour(piece.astype(float), levels=[0.5], colors=[colours[s]],
                           linewidths=1.6 if s == plan else 0.8, extent=extent, origin="upper")
        ax.contour(iso[0], levels=[70.0], colors=["#ff2a2a"], linewidths=1.0,
                   linestyles="--", extent=extent, origin="upper")
        ax.contour(iso[1].astype(float), levels=[0.5], colors=["white"], linewidths=0.9,
                   extent=extent, origin="upper")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left")

    ext_ax = (0, (x1 - x0) * dx, (y1 - y0) * dy, 0)
    draw(ax_a, ct[kz, y0:y1, x0:x1], ext_ax,
         {s: m[kz, y0:y1, x0:x1] for s, m in rect.items()},
         (dose[kz, y0:y1, x0:x1], ctv[kz, y0:y1, x0:x1]), "(a) Axial")
    # Sagittal: superior at the top, anterior on the left.
    sl = (slice(z1, z0, -1), slice(y0, y1), kx)
    ext_sa = (0, (y1 - y0) * dy, (z1 - z0) * dz, 0)
    draw(ax_s, ct[sl], ext_sa, {s: m[sl] for s, m in rect.items()}, (dose[sl], ctv[sl]),
         "(b) Sagittal")
    ax_s.set_aspect("equal")
    ax_a.set_aspect("equal")

    # DVHs of the same plan on every rectum contour.
    for s in ids:
        d = np.sort(dose[rect[s]])[::-1]
        v = 100.0 * np.arange(1, d.size + 1) / d.size
        ax_d.plot(d, v, color=colours[s], lw=1.6 if s == plan else 0.9)
    ax_d.axvline(70, color="0.6", lw=0.6, ls=":")
    ax_d.axhspan(0, 10, xmin=0, xmax=1, color=TIER_COLOURS["per_protocol"], zorder=0)
    ax_d.axhspan(10, 20, color=TIER_COLOURS["minor"], zorder=0)
    ax_d.set_xlim(30, 84)
    ax_d.set_ylim(0, 60)
    ax_d.set_xlabel("Dose (Gy)")
    ax_d.set_ylabel("Rectum volume (%)")
    ax_d.set_title(f"(c) Rectum DVH, plan {LABEL[plan]}", loc="left")
    v70 = observer_cells(ep, "rectum_v70pct")
    v70 = v70[(v70.patient == patient) & (v70.plan_set == plan)].value
    ax_d.text(71, 56, f"$V_{{70}}$: {v70.min():.1f}–{v70.max():.1f}%", fontsize=7, va="top")

    handles = [Line2D([], [], color=colours[s], lw=1.6 if s == plan else 1.0, label=LABEL[s])
               for s in ids]
    handles += [Line2D([], [], color="white", lw=1.0, label="Consensus CTV"),
                Line2D([], [], color="#ff2a2a", lw=1.0, ls="--", label="70 Gy isodose")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=12, frameon=False,
                     bbox_to_anchor=(0.5, -0.1), handlelength=1.4, columnspacing=0.9)
    for text in leg.get_texts():
        if text.get_text() == "Consensus CTV":
            leg.legend_handles[len(ids)].set_color("0.35")
    save(fig, "fig3_example_patient")


# ============================================================================ Figure 3
def figure3(con: pd.DataFrame) -> None:
    """Rectum V70 of every observer plan across the contours it was evaluated on."""
    c = con[(con.constraint == "rectum_v70") & con.plan_set.isin(cfg.observers)
            & con.truth_set.isin(cfg.observers)]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.8))
    ax.axhspan(0, 10, color=TIER_COLOURS["per_protocol"], zorder=0, lw=0)
    ax.axhspan(10, 20, color=TIER_COLOURS["minor"], zorder=0, lw=0)
    ax.axhspan(20, 40, color=TIER_COLOURS["major"], zorder=0, lw=0)
    x, centres, gap = 0, [], 1.6
    for patient in cfg.patients:
        p = c[c.patient == patient]
        own = p[p.plan_set == p.truth_set].set_index("plan_set").value.sort_values()
        start = x
        for plan in own.index:
            vals = p[p.plan_set == plan].value.to_numpy()
            ax.plot([x, x], [vals.min(), vals.max()], color="0.25", lw=0.8, zorder=2)
            ax.scatter(np.full(vals.size, x), vals, s=5, color=PATIENT_COLOURS[patient],
                       alpha=0.75, lw=0, zorder=3)
            ax.scatter([x], [own[plan]], s=16, marker="D", color="black", zorder=4, lw=0)
            x += 1
        centres.append((start + x - 1) / 2)
        if patient != cfg.patients[-1]:
            ax.axvline(x - 1 + gap / 2, color="white", lw=2)
        x += gap - 1
    ax.set_xticks(centres, [f"Patient {p}" for p in cfg.patients])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-1, x - gap + 1)
    ax.set_ylim(0, 36)
    ax.set_ylabel(r"Rectum $V_{70}$ (%)")
    handles = [Patch(color=TIER_COLOURS["per_protocol"], label=r"Per protocol ($\leq$10%)"),
               Patch(color=TIER_COLOURS["minor"], label="Minor variation (10–20%)"),
               Patch(color=TIER_COLOURS["major"], label="Major variation (>20%)"),
               Line2D([], [], marker="D", color="black", ls="", ms=4,
                      label="Evaluated on its own contour"),
               Line2D([], [], marker="o", color="0.4", ls="", ms=3,
                      label="Evaluated on another observer's contour")]
    ax.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
              bbox_to_anchor=(0.5, 1.13), handlelength=1.2, columnspacing=1.0)
    save(fig, "fig4_rectum_v70_compliance")


# ============================================================================ Figure 4
def signed(x: float) -> str:
    """Two decimals with an explicit sign, and a plain 0.00 rather than a signed zero."""
    t = f"{x:+.2f}"
    return "0.00" if t in ("+0.00", "-0.00") else t.replace("-", "\N{MINUS SIGN}")


def figure4() -> None:
    """Estimation error against unsigned and signed geometric agreement."""
    f = pd.read_parquet(RES / "aim3_features.parquet")
    rho = pd.read_parquet(RES / "section4_per_patient_rho.parquet")
    rows = [("ctv_d98", r"CTV $D_{98}$ estimation error (Gy)", 1.0),
            ("ntcp_rectum_relative", "Rectal NTCP estimation error (pp)", 100.0)]
    cols = [("dsc", "Dice"), ("signed_msd_mm", "Signed MSD (mm)")]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.9))
    letters = iter("abcd")
    for r, (endpoint, ylab, scale) in enumerate(rows):
        g = f[f.endpoint == endpoint]
        y = g.estimation_error * scale
        lo, hi = np.nanpercentile(y, [0.5, 99.5])
        span = hi - lo
        ylim = (lo - 0.08 * span, hi + 0.08 * span)
        for c, (metric, xlab) in enumerate(cols):
            ax = axes[r, c]
            for patient in cfg.patients:
                gp = g[g.patient == patient]
                ax.scatter(gp[metric], gp.estimation_error * scale, s=7, lw=0, alpha=0.7,
                           color=PATIENT_COLOURS[patient], label=f"Patient {patient}")
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_ylim(*ylim)
            outside = int(((y < ylim[0]) | (y > ylim[1])).sum())
            s = rho[(rho.endpoint == endpoint) & (rho.target == "estimation_error")
                    & (rho.metric == metric)].iloc[0]
            note = (rf"$\rho$ = {signed(s.rho_median)} ({signed(s.rho_min)} to "
                    rf"{signed(s.rho_max)})" "\n"
                    rf"$R^2_{{\mathrm{{LOPO}}}}$ = {signed(s.lopo_r2)}")
            if outside:
                note += f"\n{outside} point{'s' if outside > 1 else ''} off scale"
            ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.5))
            ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel(ylab)
            ax.set_title(f"({next(letters)})", loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), markerscale=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig5_geometry_vs_dose")


if __name__ == "__main__":
    endpoints = pd.read_parquet(RES / "endpoints_long.parquet")
    constraints = pd.read_parquet(RES / "constraints_long.parquet")
    figure_schematic()
    figure_matrices(endpoints)
    figure3(constraints)
    figure4()
    if cfg.dicom_root.exists():
        figure2(endpoints)
    else:
        print("Figure 3 skipped: it needs the planning CT, masks and dose (data/ not found)")
