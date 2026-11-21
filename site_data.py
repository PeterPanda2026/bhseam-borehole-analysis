"""
Shared data access for the plotting scripts in this folder.

Everything is read from the project files next to this script and is in the
LOCAL coordinates of the data set (x east, y north, metres from the local
origin; ground level in metres above the site datum = the lowest hole):

    borehole_locations.csv   hole positions and ground levels
    drilling_logs.xlsx       the logs (read with read_drilling_logs.py)
    faults.csv, fault_throws.csv

Figures and tables are written to the folder  output/  (created on first use).

    from site_data import load_holes, strata, load_faults
"""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from read_drilling_logs import read_drill_logs

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
LOGS = HERE / "drilling_logs.xlsx"
STRATA_CSV = HERE / "strata_layers.csv"   # written by read_drilling_logs.py; rebuilt if the workbook is newer

# figure colours (ink, secondary ink, muted, grid lines, background)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#fcfcfb"

# --- the local setting-out grid (project_notes.txt, section 4) ---------------
# plots are 60 m strips numbered 1, 2, 3 ... towards the south west and C, B, A
# towards the north east of plot 1; lines A-Y (no I, O, P) at 3 m across a plot;
# columns at 3 m along it (columns NW of column 1 are 0, 999, 998 ...)
ROWS = "ABCDEFGHJKLMNQRSTUVWXY"
ROW_OFFSET = {r: 3 * i for i, r in enumerate(ROWS)}
PLOT_OFFSET = {"A": 180, "B": 120, "C": 60, **{str(n): -60 * (n - 1) for n in range(1, 11)}}
HOLE_RE = re.compile(r"^([A-HJ-NQ-Y])(\d+)")


def local_xy(plot, row, col):
    """Position of a grid hole in setting-out grid metres (across, along the plot)."""
    return PLOT_OFFSET[plot] + ROW_OFFSET[row], -3.0 * col


def fit_helmert(local, xy):
    """Least-squares rotation + translation (scale 1) from setting-out grid to x, y:
    x = a*u - b*v + x0,  y = b*u + a*v + y0."""
    lc, ec = local.mean(0), xy.mean(0)
    l0, e0 = local - lc, xy - ec
    theta = np.arctan2((l0[:, 0] * e0[:, 1] - l0[:, 1] * e0[:, 0]).sum(), (l0 * e0).sum())
    a, b = np.cos(theta), np.sin(theta)
    return np.array([a, b, ec[0] - (a * lc[0] - b * lc[1]), ec[1] - (b * lc[0] + a * lc[1])])


def helmert_fwd(p, local):
    a, b, x0, y0 = p
    u, v = np.atleast_2d(local).T
    return np.c_[a * u - b * v + x0, b * u + a * v + y0]


@lru_cache(maxsize=1)
def load_holes():
    """borehole_locations.csv with, for grid holes (production and test), the line
    letter, column number and grid-diagonal index used to pick section lines."""
    t = pd.read_csv(HERE / "borehole_locations.csv", dtype={"plot": str, "hole_no": str})
    m = t.hole_no.str.extract(HOLE_RE)
    grid = t.hole_type.isin(["production", "test", "test grout"]) & m[0].notna()
    t["row"] = m[0].where(grid)
    n = pd.to_numeric(m[1], errors="coerce").where(grid)
    t["col"] = np.where(n >= 900, n - 1000, n)
    ok = grid & t["plot"].isin(PLOT_OFFSET)
    t["diag"] = np.nan
    t.loc[ok, "diag"] = [round((PLOT_OFFSET[p] + ROW_OFFSET[r]) / 3 - c)
                         for p, r, c in zip(t.loc[ok, "plot"], t.loc[ok, "row"], t.loc[ok, "col"])]
    return t


# --- strata ------------------------------------------------------------------
# stratum class -> colour (conventional geological tones), in legend order
CLASSES = {
    "Overburden / boulder clay": "#a6764d",
    "Sandstone": "#f2d15c",
    "Mudstone": "#8f96a0",
    "Hard strata (unspecified)": "#1f4e9c",
    "Hard with soft bands (floor)": "#1f4e9c",  # hard colour + soft-colour hatching
    "Firm": "#3fa34d",
    "Soft / broken ground": "#c2338f",
    "Mine waste / worked ground": "#e07b39",
    "Coal": "#111111",
    "Grout met": "#1fb5c9",
    "Void": "#ffffff",
    "Other": "#f0c9c9",
}
HATCH = {"Hard with soft bands (floor)": "////"}
EDGE = {"Void": "#e34948", "Hard with soft bands (floor)": CLASSES["Soft / broken ground"]}

# text that is not a stratum (form headings, plant notes) and common typos
NOT_STRATA = re.compile(r"description of strata|burst hoses|swap over|rotation handle|bulkhead|"
                        r"tighten up|re-drill|fill up hydraulic|^g/l$|^tbc$")
TYPOS = [(r"\+d\d+", ""),  # spreadsheet artefacts, e.g. "soft/gro+d53295ut"
         (r"borken", "broken"), (r"caol", "coal"), (r"sofft", "soft"), (r"\bsft\b", "soft"),
         (r"grput", "grout"), (r"^o/b$", "overburden"), (r"^s/st$", "sandstone")]


def classify(desc):
    """Free-text description -> stratum class (first matching rule wins).
    None for text that is not a stratum; those layers are left out."""
    d = str(desc).strip().lower()
    if NOT_STRATA.search(d):
        return None
    for typo, word in TYPOS:
        d = re.sub(typo, word, d)
    if "void" in d:
        return "Void"
    if "grout" in d:
        return "Grout met"
    if re.search(r"wast|worked|goaf|gob", d):
        return "Mine waste / worked ground"
    if "coal" in d:
        return "Coal"
    if "hard" in d and "broken" not in d:
        if "soft" in d:
            return "Hard with soft bands (floor)"
        if "firm" in d:
            return "Hard strata (unspecified)"
    if re.search(r"f(?:ir|ri)m", d) and "broken" not in d:
        return "Firm"
    if any(w in d for w in ("soft", "broken", "firm", "frac")):
        return "Soft / broken ground"
    sst = re.search(r"sa(?:nd|n)\s*sto(?:ne|me)", d)
    mud = re.search(r"mu(?:d|da|)\s*sto(?:ne|me)", d)
    if sst and (not mud or sst.start() < mud.start()):
        return "Sandstone"
    if mud:
        return "Mudstone"
    if re.fullmatch(r"mud\s*traces?", d):
        return "Other"
    if re.search(r"silt\s*stone|shale", d):
        return "Other"
    if any(w in d for w in ("overbur", "boulder", "clay", "drift", "fill", "topsoil", "sand", "mud", "silt")):
        return "Overburden / boulder clay"
    if any(w in d for w in ("hard", "rock", "strata")):
        return "Hard strata (unspecified)"
    return "Other"


@lru_cache(maxsize=1)
def all_layers():
    """Every logged layer (read_drilling_logs.py), keyed by hole_id = plot + BH No."""
    if STRATA_CSV.exists() and STRATA_CSV.stat().st_mtime > LOGS.stat().st_mtime:
        s = pd.read_csv(STRATA_CSV, dtype={"plot_no": str, "borehole": str}, low_memory=False)
    else:
        s, _ = read_drill_logs(LOGS)
        s.to_csv(STRATA_CSV, index=False)   # the same file read_drilling_logs.py writes
    s["hole_id"] = s.plot_no.astype(str) + s.borehole.astype(str)
    s = s.rename(columns={"description": "Description", "depth_from": "From (m)", "depth_to": "To base (m)",
                          "layer_no": "Layer no.", "plot": "line_letter", "plot_no": "plot"})
    s["class"] = s.Description.map(classify)
    return s


@lru_cache(maxsize=1)
def strata():
    """Layers that are strata (headings and plant notes dropped)."""
    s = all_layers()
    return s[s["class"].notna()].copy()


# --- faults ------------------------------------------------------------------
THROW_CLASSES = [  # throw (m), light to dark
    (0.0, 1.0, "< 1 m", "#f2a0b8", 1.0),
    (1.0, 3.0, "1 – 3 m", "#d9467a", 1.6),
    (3.0, 10.0, "3 – 10 m", "#a3104f", 2.3),
    (10.0, 999.0, "≥ 10 m", "#5c0030", 3.2),
]
UNKNOWN = ("Throw not known", "#9a9892", 1.0)


def load_faults():
    """Fault vertices (fault_id, vertex, x_m, y_m) and one row per fault with the
    throw in metres added (throw_m)."""
    v = pd.read_csv(HERE / "faults.csv").sort_values(["fault_id", "vertex"], kind="stable")
    s = pd.read_csv(HERE / "fault_throws.csv", dtype={"throw_as_marked": str}).set_index("fault_id")
    s["throw_m"] = pd.to_numeric(s.throw_ft, errors="coerce") * 0.3048
    return v, s


def throw_style(m):
    """(legend label, colour, line width) for a throw in metres."""
    if pd.isna(m):
        return UNKNOWN
    for lo, hi, lab, col, lw in THROW_CLASSES:
        if lo <= m < hi:
            return lab, col, lw
    return UNKNOWN


def style_axes(ax, axis="both", lw=0.6, zorder=0):
    """Light grid, no top/right frame - the same look on every figure."""
    ax.grid(axis=axis, color=GRID, lw=lw, zorder=zorder)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def rc():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                         "xtick.color": INK2, "ytick.color": INK2})
