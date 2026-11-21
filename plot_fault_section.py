"""
Level cross-section along the grid line nearest a y value, with the faults
(faults.csv) marked where they cross it. A step in the drilled coal at a fault
crossing shows the throw, and which side is down.

Run:  python plot_fault_section.py            (grid line nearest y = 230 m)
      python plot_fault_section.py 260
Output: output/section_y<y>_level_faults.png and a CSV of the crossings.
"""

import sys

import numpy as np
import pandas as pd

from plot_section import grid_line_holes, plot_section
from site_data import OUT, load_faults, throw_style


def _cross(p1, p2, q1, q2):
    """Intersection point of segments p1-p2 and q1-q2, or None."""
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return None
    t = ((q1 - p1)[0] * d2[1] - (q1 - p1)[1] * d2[0]) / den
    u = ((q1 - p1)[0] * d1[1] - (q1 - p1)[1] * d1[0]) / den
    return p1 + t * d1 if 0 <= t <= 1 and 0 <= u <= 1 else None


def fault_crossings(a, b):
    """Every place a mapped fault crosses the section a-b (x, y arrays)."""
    v, s = load_faults()
    sec_dir = (b - a) / np.hypot(*(b - a))
    rows = []
    for fid, d in v.groupby("fault_id", sort=False):
        P = d[["x_m", "y_m"]].to_numpy()
        for p1, p2 in zip(P[:-1], P[1:]):
            x = _cross(a, b, p1, p2)
            if x is None:
                continue
            f_dir = (p2 - p1) / np.hypot(*(p2 - p1))
            ang = np.degrees(np.arccos(min(1.0, abs(float(sec_dir @ f_dir)))))
            r = s.loc[fid]
            rows.append(dict(fault_id=fid, x=round(x[0], 1), y=round(x[1], 1), angle_to_section_deg=round(ang),
                             throw_as_marked=r.throw_as_marked, throw_m=r.throw_m,
                             figure_uncertain=r.figure_uncertain, remarks=r.remarks))
    return pd.DataFrame(rows).sort_values("x") if rows else pd.DataFrame()


def main(y=230.0):
    line, info = grid_line_holes(y)
    x0, x1 = line.x_m.min() - 5, line.x_m.max() + 5
    a = np.array([x0, info["slope"] * x0 + info["intercept"]])
    b = np.array([x1, info["slope"] * x1 + info["intercept"]])
    xs = fault_crossings(a, b)
    faults = []
    for r in xs.itertuples():
        lab, col, lw = throw_style(r.throw_m)
        throw = "" if pd.isna(r.throw_as_marked) else f"  {r.throw_as_marked}"
        faults.append(dict(x=r.x, label=f"{r.fault_id}{throw}  ({r.angle_to_section_deg}°)", colour=col,
                           lw=max(lw, 1.2)))
    name = f"section_y{y:.0f}_level_faults"
    xs.to_csv(OUT / f"{name}_crossings.csv", index=False)
    print(xs.to_string(index=False) if len(xs) else "no fault crosses this line")
    return plot_section(y, elevation=True, faults=faults, suffix="_faults", name="")


if __name__ == "__main__":
    print(main(float(sys.argv[1]) if len(sys.argv) > 1 else 230.0))
