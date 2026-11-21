"""
Plan of the boreholes in this project, coloured by plot, with the fault traces
and the throw written beside each fault on the colliery plan.

Reads (from this folder):
    borehole_locations.csv   hole positions, local x/y in metres
    faults.csv               fault traces, same local x/y
    fault_throws.csv         throw figure of each fault (feet/inches)
Writes:
    borehole_layout.png      the plan

Requires: pandas, matplotlib   ->   pip install pandas matplotlib
Run:      python plot_borehole_layout.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
PLOT_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7d5ba6", "#c23b22", "#00a0b0", "#6b4226", "#52514e"]
FAULT_COLOUR = "#a3104f"


def main():
    holes = pd.read_csv(HERE / "borehole_locations.csv", dtype={"plot": str})
    faults = pd.read_csv(HERE / "faults.csv")
    throws = pd.read_csv(HERE / "fault_throws.csv", dtype={"throw_as_marked": str}).set_index("fault_id")

    span_x = holes.x_m.max() - holes.x_m.min()
    span_y = holes.y_m.max() - holes.y_m.min()
    fig, ax = plt.subplots(figsize=(18, max(4.0, 18 * span_y / span_x + 2.5)))

    # boreholes, one colour per plot (west to east)
    order = holes.groupby("plot").x_m.mean().sort_values().index
    for k, plot in enumerate(order):
        d = holes[holes["plot"] == plot]
        ax.scatter(d.x_m, d.y_m, s=6, color=PLOT_COLOURS[k % len(PLOT_COLOURS)], linewidths=0, zorder=3,
                   label=f"Plot {plot} ({len(d)} holes)")

    # faults, with their throw (as written on the plan) beside them
    for k, (fid, d) in enumerate(faults.groupby("fault_id", sort=False)):
        d = d.sort_values("vertex")
        ax.plot(d.x_m, d.y_m, color=FAULT_COLOUR, lw=2, zorder=4, label="Fault trace" if k == 0 else None)
        throw = throws.loc[fid, "throw_as_marked"]
        text = fid if pd.isna(throw) or not str(throw).strip() else f"{fid}  {throw}"
        if throws.loc[fid, "figure_uncertain"]:
            text += " ?"
        mid = d.iloc[len(d) // 2]
        ax.annotate(text, (mid.x_m, mid.y_m), xytext=(4, 4), textcoords="offset points", fontsize=9,
                    fontweight="bold", color=FAULT_COLOUR, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_aspect("equal")
    ax.grid(color="#e4e3df", lw=0.6, zorder=0)
    ax.set_xlabel("x (m, local grid; increases east)")
    ax.set_ylabel("y (m, local grid; increases north)")
    ax.set_title(f"Borehole layout: {len(holes):,} holes, {faults.fault_id.nunique()} fault traces "
                 "(throw in feet/inches as written on the plan, e.g. 1/6 = 1 ft 6 in; ? = figure uncertain)",
                 loc="left", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, markerscale=2.5, fontsize=9)
    fig.savefig(HERE / "borehole_layout.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {HERE / 'borehole_layout.png'}")


if __name__ == "__main__":
    main()
