"""
Contour maps of the worked seam horizon across the site.

The horizon is picked from the drill logs as the top of each hole's first
coal / soft / broken / void / grout-met layer - i.e. the first sign of the
worked seam below the drift. Picks are gridded (linear interpolation on the
Delaunay triangulation of the holes) and contoured, twice:

    worked_seam_depth_contours.png   depth below ground level (m)
    worked_seam_level_contours.png   level (m above site datum), from each hole's ground level

Run:  python plot_seam_contours.py
Output: those two figures and output/worked_seam_picks.csv
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from site_data import GRID, INK, INK2, OUT, SURFACE, load_holes, rc, strata, style_axes

SEAM_CLASSES = ("Coal", "Soft / broken ground", "Void", "Grout met")
DEPTH_LIMITS = (5.0, 40.0)  # plausible depth of the shallow worked seam; beyond = log typo


def seam_picks():
    """One row per production hole: position, ground level, depth and level of the horizon."""
    s = strata()
    s = s[s["class"].isin(SEAM_CLASSES)]
    first = (s.sort_values(["hole_id", "From (m)"]).groupby("hole_id")
             .agg(depth=("From (m)", "first"), description=("Description", "first"),
                  seam_class=("class", "first")))
    t = load_holes().set_index("hole_id")
    t = t[t.hole_type == "production"]
    p = first.join(t[["plot", "x_m", "y_m", "ground_level_m"]], how="inner").dropna()
    p = p.rename(columns={"x_m": "x", "y_m": "y", "ground_level_m": "ground"})
    p = p[p.depth.between(*DEPTH_LIMITS)]
    p["level"] = p.ground - p.depth
    return p.reset_index()


def _grid(p, value, step=5.0, max_gap=12.0):
    """Interpolate picks onto a regular grid; blank cells more than `max_gap` from a hole."""
    gx = np.arange(p.x.min() - step, p.x.max() + step, step)
    gy = np.arange(p.y.min() - step, p.y.max() + step, step)
    XX, YY = np.meshgrid(gx, gy)
    Z = griddata(p[["x", "y"]].to_numpy(), p[value].to_numpy(), (XX, YY), method="linear")
    far, _ = cKDTree(p[["x", "y"]].to_numpy()).query(np.c_[XX.ravel(), YY.ravel()])
    return XX, YY, np.where(far.reshape(XX.shape) > max_gap, np.nan, Z)


def plot_contours(p, value, title, label, cmap, out, name, step=2.0):
    XX, YY, Z = _grid(p, value)
    levels = np.arange(np.floor(np.nanmin(Z) / step) * step, np.nanmax(Z) + step, step)
    rc()
    span = (p.x.max() - p.x.min(), p.y.max() - p.y.min())
    fig = plt.figure(figsize=(16, 16 * span[1] / span[0] + 2.2), facecolor=SURFACE)
    ax = fig.add_axes([0.06, 0.06, 0.80, 0.84])
    ax.set_facecolor(SURFACE)
    fill = ax.contourf(XX, YY, Z, levels=levels, cmap=cmap, extend="both", zorder=2)
    lines = ax.contour(XX, YY, Z, levels=levels, colors=INK, linewidths=0.5, alpha=0.55, zorder=3)
    ax.clabel(lines, levels[::2], fmt="%g", fontsize=8, colors=INK)
    ax.scatter(p.x, p.y, s=0.7, color=INK2, alpha=0.35, linewidths=0, zorder=4,
               label=f"{len(p):,} holes with a seam pick")
    ax.set_aspect("equal")
    style_axes(ax, lw=0.5, zorder=1)
    ax.set_xlabel("x (m, local grid; increases east)", color=INK2)
    ax.set_ylabel("y (m, local grid; increases north)", color=INK2)
    ax.legend(loc="lower left", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9, labelcolor=INK)
    fig.text(0.06, 0.975, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.945, "Horizon picked from the drill logs as the top of each hole's first coal / soft / "
             "broken / void / grout-met layer   ·   linear interpolation between holes, blank where no "
             "hole within 12 m", fontsize=9.5, color=INK2, va="top")
    cax = fig.add_axes([0.88, 0.06, 0.02, 0.6])
    cb = fig.colorbar(fill, cax=cax)
    cb.set_label(label, color=INK2)
    cb.ax.tick_params(colors=INK2)
    fig.savefig(out / f"{name}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out / f"{name}.png"


def main(out=OUT):
    p = seam_picks()
    p.to_csv(out / "worked_seam_picks.csv", index=False)
    print(f"{len(p):,} holes with a seam pick; depth {p.depth.min():.1f}-{p.depth.max():.1f} m, "
          f"level {p.level.min():.1f} to {p.level.max():.1f} m above site datum")
    a = plot_contours(p, "depth", "Worked seam - depth below ground level", "Depth below ground level (m)",
                      "YlOrBr", out, "worked_seam_depth_contours")
    b = plot_contours(p, "level", "Worked seam - level above site datum", "Level (m above site datum)",
                      "YlGnBu", out, "worked_seam_level_contours")
    return a, b


if __name__ == "__main__":
    for path in main():
        print(path)
