"""
Geological cross-section: drilled strata of the boreholes along a line of
roughly constant y (local north coordinate).

The setting-out grid runs at about 45 degrees, so its diagonals (holes whose
line position minus column number is constant) run almost east-west and every
hole on one is exactly collinear; diagonals are 2.12 m apart and a full line of
holes occurs on every second one. A section is taken along the diagonal whose
holes lie closest to the requested y.

Run:  python plot_section.py            (sections A-A', B-B', C-C' near y = 180 m: the grid
                                         line closest to it and the full lines 4.24 m north
                                         and south; depth below ground and level)
      python plot_section.py 250        (the same three sections near y = 250 m)
      python plot_section.py 250 2      (all holes within +/- 2 m of y = 250 m instead)
      python plot_section.py all        (one section along every grid line, output/sections)

Output: output/section_y<y>[...].png
"""

import sys
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.patches import Patch

from site_data import (CLASSES, EDGE, HATCH, INK, INK2, MUTED, OUT, SURFACE, all_layers, classify,
                       load_holes, rc, strata, style_axes)

DATUM_NOTE = {False: "depths as logged below ground level",
              True: "logged depths hung from each hole's ground level (m above site datum)"}
# cross-sections along grid lines: (name, shift in diagonals from the line nearest the y value)
SECTIONS = [("A", 0), ("B", 2), ("C", -2)]  # A-A' nearest; B-B' 4.24 m north; C-C' 4.24 m south


def classification_table(out=OUT):
    """Every distinct description (lower-cased, trimmed) with its class and layer
    count -> output/strata_classification.xlsx."""
    s = all_layers()
    desc = s.Description.astype(str).str.strip().str.lower()
    t = desc.value_counts().rename_axis("description").reset_index(name="layers")
    t["class"] = t.description.map(classify).fillna("(excluded - not a stratum)")
    order = {c: i for i, c in enumerate(list(CLASSES) + ["(excluded - not a stratum)"])}
    t = t.sort_values(["class", "layers"], key=lambda c: c.map(order) if c.name == "class" else -c)
    summary = (t.groupby("class", sort=False)
               .agg(descriptions=("description", "size"), layers=("layers", "sum")).reset_index())
    with pd.ExcelWriter(out / "strata_classification.xlsx") as xw:
        summary.to_excel(xw, sheet_name="class_summary", index=False)
        t[["class", "description", "layers"]].to_excel(xw, sheet_name="description_to_class", index=False)
    return summary, t


@lru_cache(maxsize=1)
def _grid_holes():
    """Production holes with their grid-diagonal index."""
    t = load_holes()
    return t[(t.hole_type == "production") & t.diag.notna()].copy()


def grid_line_holes(y, shift=0):
    """Holes on the grid diagonal closest to y, or `shift` diagonals north (+) /
    south (-) of it (shift = +/-2 -> the next full line, 4.24 m away)."""
    t = _grid_holes()
    k = (t.groupby("diag").y_m.mean() - y).abs().idxmin() + shift
    line = t[t.diag == k].sort_values("x_m")
    slope, intercept = np.polyfit(line.x_m, line.y_m, 1)
    off = np.abs(line.y_m - (slope * line.x_m + intercept)).max()
    info = dict(diag=int(k), slope=slope, intercept=intercept, y_west=slope * line.x_m.min() + intercept,
                y_east=slope * line.x_m.max() + intercept, max_offset=off)
    return line, info


def section_holes(y, half_width):
    """Every hole (any type) within +/- half_width m of y."""
    t = load_holes()
    return t[(t.y_m - y).abs() <= half_width].sort_values("x_m")


def plot_section(y=180.0, half_width=None, out=OUT, shift=0, name="A", depth_max=None, xlim=None,
                 elevation=False, ylim=None, ground_top=None, suffix="", faults=None):
    """half_width None: holes on one grid diagonal, the one nearest y moved `shift`
    diagonals north/south. A number: all holes within +/- half_width m of y.
    elevation=True plots level above the site datum using each hole's ground
    level, instead of depth below ground.
    faults: optional list of dicts (x, label, colour, lw) - where a mapped fault
    crosses the section, drawn as a dashed vertical line (plot_fault_section.py)."""
    if half_width is None:
        holes, info = grid_line_holes(y, shift)
    else:
        holes = section_holes(y, half_width)
    holes = holes.copy()
    holes["ground"] = holes.ground_level_m if elevation else 0.0
    s = strata()
    s = s[s.hole_id.isin(holes.hole_id)].merge(holes[["hole_id", "x_m", "ground"]], on="hole_id")
    logged = holes[holes.hole_id.isin(s.hole_id)].drop_duplicates("hole_id")
    if logged.empty:
        print(f"no logged holes {'on that grid line' if half_width is None else f'within ±{half_width:g} m'} "
              f"of y = {y:,.0f} m - nothing drawn")
        return None
    up = -1 if elevation else 1  # depth axis points down, level axis points up
    gaps = np.diff(np.sort(logged.x_m.to_numpy()))
    bar = 0.8 * gaps[gaps > 0.5].min() if (gaps > 0.5).any() else 3.0

    rc()
    fig = plt.figure(figsize=(18, 8), facecolor=SURFACE)
    ax = fig.add_axes([0.045, 0.09, 0.80, 0.76])
    ax.set_facecolor(SURFACE)
    ok = s["To base (m)"] > s["From (m)"]
    for cls, d in s[ok].groupby("class"):
        ax.bar(d.x_m, d["To base (m)"] - d["From (m)"], bottom=d.ground + up * d["To base (m)"], width=bar,
               color=CLASSES[cls], edgecolor=EDGE.get(cls, "none"), hatch=HATCH.get(cls),
               linewidth=0.6 if cls == "Void" else 0, zorder=3)
    for k, f in enumerate(faults or []):
        ax.axvline(f["x"], color=f["colour"], lw=f["lw"], ls=(0, (6, 3)), zorder=9,
                   label="Fault crossing (mine plan)" if k == 0 else None)
        ax.text(f["x"] + 0.8, 0.015, f["label"], transform=ax.get_xaxis_transform(), rotation=90,
                ha="left", va="bottom", fontsize=8.5, fontweight="bold", color=f["colour"], zorder=10,
                path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)])

    depth_max = depth_max or s.loc[s["To base (m)"] < 60, "To base (m)"].max()
    ground = logged.set_index("hole_id").ground
    top_of = (ground_top or ground.max()) if elevation else 0.0
    # plot names over each plot's run of holes
    y_names = top_of - up * 3.0
    for plot, d in logged.groupby("plot"):
        ax.annotate("", xy=(d.x_m.min() - bar / 2, y_names), xytext=(d.x_m.max() + bar / 2, y_names),
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2))
        ax.text(d.x_m.mean(), y_names - up * 0.6, f"Plot {plot}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
    # every hole's name directly under its stick
    bottom = s.groupby("hole_id")["To base (m)"].max()
    for r in logged.itertuples():
        ax.text(r.x_m, ground[r.hole_id] + up * (bottom[r.hole_id] + 0.4), r.hole_id, rotation=90,
                ha="center", va="top", fontsize=6 if len(logged) > 60 else 8, color=INK2)

    if elevation:
        lo = (ylim or (None, None))[0] or ground.min() - depth_max - 7
        ax.set_ylim(lo, top_of + 6)
        ax.set_yticks(np.arange(np.ceil(lo / 5) * 5, top_of + 5, 5))
    else:
        ax.set_ylim(depth_max + 5, -6)
        ax.set_yticks(np.arange(0, depth_max + 5, 5))
    ax.set_xlim(*(xlim or (holes.x_m.min() - 5, holes.x_m.max() + 5)))
    style_axes(ax, axis="y")
    ax.set_xlabel("x (m, local grid; increases east)", color=INK2)
    ax.set_ylabel("Level (m above site datum)" if elevation else "Depth below ground level (m)", color=INK2)
    head = f"Section {name}–{name}′" if half_width is None and name else f"Section along y = {y:,.0f} m"
    fig.text(0.045, 0.965, f"{head}  –  borehole strata", fontsize=15, fontweight="bold", color=INK, va="top")
    if half_width is None:
        rel = ("Grid line nearest" if shift == 0 else
               f"Grid line {abs(shift) * 2.12:.2f} m {'north' if shift > 0 else 'south'} of the line nearest")
        sub = (f"{rel} y = {y:,.0f} m   ·   y {info['y_west']:,.1f} (west) to {info['y_east']:,.1f} (east), "
               f"all holes on one straight line (max offset {info['max_offset'] * 1000:.0f} mm)"
               f"   ·   {len(logged)} holes   ·   {DATUM_NOTE[elevation]}")
    else:
        sub = f"All holes within ±{half_width:g} m   ·   {len(logged)} holes   ·   {DATUM_NOTE[elevation]}"
    fig.text(0.045, 0.925, sub, fontsize=10, color=INK2, va="top")
    handles = [Patch(facecolor=CLASSES[c], edgecolor=EDGE.get(c, MUTED), hatch=HATCH.get(c), lw=0.6,
                     label=c) for c in CLASSES if c != "Other" or c in set(s["class"])]
    handles += ax.get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.855, 0.85), frameon=False,
               fontsize=9.5, labelcolor=INK, title="Stratum (driller's description)", title_fontsize=10)
    stem = (f"section_y{y:.0f}" + (f"_band{half_width:g}m" if half_width is not None
                                   else "" if shift == 0 or not name else f"_{name}")
            + ("_level" if elevation else "") + suffix)
    fig.savefig(out / f"{stem}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"{len(holes)} holes on the line, {len(logged)} with strata -> {stem}.png")
    return out / f"{stem}.png"


def grid_lines(min_holes=30):
    """Every grid diagonal with at least `min_holes` holes, south to north."""
    t = _grid_holes()
    return (t.groupby("diag").agg(holes=("hole_id", "size"), y=("y_m", "mean"),
                                  x_min=("x_m", "min"), x_max=("x_m", "max"))
            .query("holes >= @min_holes").sort_values("y").reset_index())


@lru_cache(maxsize=2)
def common_limits(elevation=True):
    """One x range, depth and level range for every grid-line section, so all the
    figures share an identical layout."""
    s = strata()
    t = _grid_holes().set_index("hole_id")
    ids = list(set(s.hole_id) & set(t.index))
    base = s[s.hole_id.isin(ids) & (s["To base (m)"] < 60)].groupby("hole_id")["To base (m)"].max()
    xlim = (t.loc[ids, "x_m"].min() - 5, t.loc[ids, "x_m"].max() + 5)
    ground = t.loc[base.index, "ground_level_m"]
    ylim = ((ground - base).min() - 7, None) if elevation else None
    return base.max(), xlim, ylim, float(ground.max())


def plot_all_lines(min_holes=30, elevation=True, out=OUT / "sections"):
    """A section along every grid line with at least `min_holes` holes, all on the
    same scales, named by y so they sort south to north."""
    out.mkdir(exist_ok=True)
    lines = grid_lines(min_holes)
    depth, xlim, ylim, top = common_limits(elevation)
    return [plot_section(r.y, None, out, 0, None, depth, xlim, elevation, ylim, top)
            for r in lines.itertuples()]


def plot_sections(y=180.0, out=OUT, elevation=False):
    """Sections A-A', B-B', C-C' with identical layout and scales, so they can be
    compared side by side."""
    depth, xlim, ylim, top = common_limits(elevation)
    order = sorted(SECTIONS, key=lambda t: -t[1])  # north to south: B, A, C
    return [plot_section(y, None, out, k, n, depth, xlim, elevation, ylim, top) for n, k in order]


if __name__ == "__main__":
    if sys.argv[1:2] == ["all"]:
        plot_all_lines()
    elif len(sys.argv) >= 3:
        plot_section(float(sys.argv[1]), float(sys.argv[2]))
    else:
        y = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
        classification_table()
        for elev in (False, True):
            for p in plot_sections(y, elevation=elev):
                print(p)
