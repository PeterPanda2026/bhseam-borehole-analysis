"""
Plan of every hole - production, perimeter, test and test-grout holes - in the
local x, y grid, with the outline and name of each 60 m plot and the
cross-section lines of plot_section.py.

Run:  python plot_all_holes.py          (section lines near y = 180 m)
      python plot_all_holes.py 250      (section lines near y = 250 m)

Output: output/all_holes_layout.png,
        output/section_index_map.png and output/section_locations_y<y>.png
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

from site_data import (GRID, INK, INK2, OUT, PLOT_OFFSET, SURFACE, fit_helmert, helmert_fwd, load_holes,
                       local_xy, rc, style_axes)

# hole_type -> (legend label, colour, marker, size)
STYLE = {
    "production": ("Production holes", "#2a78d6", "o", 3),
    "perimeter":  ("Perimeter holes (P / PP)", "#1baf7a", "s", 6),
    "test":       ("Test holes (-T)", "#e87ba4", "^", 40),
    "test grout": ("Test-grout holes (TG)", "#eb6834", "D", 30),
}


def plot_boundaries(t):
    """Outline of each plot: the 60 m strip between its plot divisions, from its
    first to its last column of holes (+1.5 m). Returns {plot: (corners as a 5x2
    array of x, y, centre x, y)}."""
    g = t[(t.hole_type == "production") & t.diag.notna()]
    loc = np.array([local_xy(p, r, c) for p, r, c in zip(g["plot"], g["row"], g["col"])])
    p = fit_helmert(loc, g[["x_m", "y_m"]].to_numpy())     # setting-out grid -> local x, y
    out = {}
    for plot, d in g.groupby("plot"):
        v = -3.0 * d.col
        v0, v1 = np.percentile(v, [0.5, 99.5])  # ignore a stray mistyped hole or two
        u0 = PLOT_OFFSET[plot]
        corners = [(u0, v0 - 1.5), (u0 + 60, v0 - 1.5), (u0 + 60, v1 + 1.5), (u0, v1 + 1.5), (u0, v0 - 1.5)]
        out[plot] = (helmert_fwd(p, np.array(corners)), helmert_fwd(p, [u0 + 30, (v0 + v1) / 2])[0])
    return out


def draw_section_line(ax, y=180.0, name="A", shift=0, first=True, lw=1.2, label_gap=7.0, fontsize=12):
    """Mark a cross-section of plot_section.py on a plan, labelled e.g. A - A'."""
    from plot_section import grid_line_holes
    line, _ = grid_line_holes(y, shift)
    (x0, y0), (x1, y1) = line.iloc[0][["x_m", "y_m"]], line.iloc[-1][["x_m", "y_m"]]
    ax.plot([x0, x1], [y0, y1], color=INK, lw=lw, zorder=5.5, solid_capstyle="butt",
            label="Cross-sections (grid lines)" if first else None)
    dy = np.sign(shift) * label_gap
    for x, yy, lab, ha in ((x0, y0, name, "right"), (x1, y1, name + "′", "left")):
        ax.annotate(lab, (x, yy), xytext=(x + (-6 if ha == "right" else 6), yy + dy), ha=ha, va="center",
                    fontsize=fontsize, fontweight="bold", color=INK, zorder=11,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6) if dy else None,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground=SURFACE)])


def plot_line_index(min_holes=30, out=OUT):
    """Index map of every section drawn by plot_section.plot_all_lines."""
    from plot_section import grid_lines
    lines = grid_lines(min_holes)
    t = load_holes()
    rc()
    fig, ax = plt.subplots(figsize=(16, 14), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(t.x_m, t.y_m, s=1, color="#c3c2b7", linewidths=0, zorder=2, label=f"{len(t):,} holes")
    for k, r in enumerate(lines.itertuples()):
        ax.plot([r.x_min, r.x_max], [r.y, r.y], color=INK, lw=0.7, zorder=3,
                label="Section lines (one figure each)" if k == 0 else None)
        if k % 4 == 0:
            ax.annotate(f"y {r.y:,.0f}", (r.x_min, r.y), xytext=(-6, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=7, color=INK2)
    ax.set_aspect("equal")
    style_axes(ax)
    ax.set_xlabel("x (m, local grid; increases east)", color=INK2)
    ax.set_ylabel("y (m, local grid; increases north)", color=INK2)
    ax.set_title(f"Cross-section index: {len(lines)} grid lines with ≥{min_holes} holes, south to north",
                 loc="left", fontsize=14, color=INK, pad=18)
    ax.text(0, 1.012, "Each line has a figure in output/sections (python plot_section.py all), "
            "named by its y (e.g. section_y180_level.png)", transform=ax.transAxes, fontsize=9, color=INK2)
    ax.legend(loc="lower left", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9, labelcolor=INK)
    fig.savefig(out / "section_index_map.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    lines.to_csv(out / "section_index.csv", index=False)
    return out / "section_index_map.png"


def plot_section_location(y=180.0, out=OUT, half_width=14.0):
    """Zoomed plan of the cross-section corridor: every hole within half_width m,
    the section lines and the holes on them."""
    from plot_section import SECTIONS, grid_line_holes
    t = load_holes()
    near = t[(t.y_m - y).abs() <= half_width]
    rc()
    fig, ax = plt.subplots(figsize=(18, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(near.x_m, near.y_m, s=5, color="#c3c2b7", linewidths=0, zorder=2, label="Other holes")
    marks = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, (name, shift) in enumerate(SECTIONS):
        line, _ = grid_line_holes(y, shift)
        ax.scatter(line.x_m, line.y_m, s=14, color=marks[k], zorder=4, label=f"Holes on {name}–{name}′ ({len(line)})")
        draw_section_line(ax, y, name, shift, first=k == 0, lw=0.8, label_gap=0, fontsize=11)
    ax.axhline(y, color=INK2, lw=0.6, ls=":", zorder=1)
    ax.text(near.x_m.max() + 22, y, f"y = {y:,.0f}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(near.x_m.min() - 15, near.x_m.max() + 35)
    style_axes(ax)
    ax.set_xlabel("x (m)", color=INK2)
    ax.set_ylabel("y (m)", color=INK2)
    ax.set_title(f"Cross-section locations near y = {y:,.0f} m (y exaggerated)", loc="left", fontsize=13, color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9, labelcolor=INK)
    fig.savefig(out / f"section_locations_y{y:.0f}.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out / f"section_locations_y{y:.0f}.png"


def plot_all(out=OUT, section_y=180.0):
    t = load_holes()
    rc()
    span = t[["x_m", "y_m"]].max() - t[["x_m", "y_m"]].min()
    fig, ax = plt.subplots(figsize=(17, 17 * span.y_m / span.x_m * 0.8 + 1.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for key, (label, colour, marker, size) in STYLE.items():
        d = t[t.hole_type == key]
        if len(d):
            ax.scatter(d.x_m, d.y_m, s=size, marker=marker, color=colour, linewidths=0.4 if size > 10 else 0,
                       edgecolors=SURFACE if size > 10 else "none", label=f"{label} ({len(d):,})",
                       zorder=3 if size < 10 else 4)
    for k, (plot, (corners, centre)) in enumerate(plot_boundaries(t).items()):
        ax.plot(corners[:, 0], corners[:, 1], color=INK, lw=1.2, zorder=5,
                label="Plot boundary (60 m strips)" if k == 0 else None)
        ax.text(*centre, f"Plot {plot}", ha="center", va="center", fontsize=11, rotation=-45,
                fontweight="bold", color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.85))
    if section_y is not None:
        from plot_section import SECTIONS
        for k, (name, shift) in enumerate(SECTIONS):
            draw_section_line(ax, section_y, name, shift, first=k == 0)
    ax.set_aspect("equal")
    style_axes(ax)
    ax.set_xlabel("x (m, local grid; increases east)", color=INK2)
    ax.set_ylabel("y (m, local grid; increases north)", color=INK2)
    ax.set_title(f"All boreholes ({len(t):,})", loc="left", fontsize=15, color=INK, pad=24)
    ax.legend(loc="lower left", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=10,
              markerscale=1.5, labelcolor=INK, borderpad=0.8)
    fig.savefig(out / "all_holes_layout.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out / "all_holes_layout.png"


if __name__ == "__main__":
    y = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
    print(plot_all(section_y=y))
    print(plot_section_location(y))
    print(plot_line_index())
