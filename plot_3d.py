"""
Interactive 3D view of borehole strata around a line of constant y.

Each borehole is a vertical column, coloured by stratum class (same rules and
colours as plot_section.py). Open the HTML in a browser: drag to rotate,
scroll to zoom, click legend entries to hide/show a class, hover for details.

Needs plotly  ->  pip install plotly
Run:  python plot_3d.py              (y = 180 m, holes within +/- 10 m)
      python plot_3d.py 250 15       (y, half-width in m)
Output: output/strata_3d_y<y>.html
"""

import sys

import numpy as np
import plotly.graph_objects as go

from plot_section import section_holes
from site_data import CLASSES, OUT, strata

# classes whose section styling (hatch / outline) can't be drawn as a 3D line
COLOURS_3D = dict(CLASSES, **{"Hard with soft bands (floor)": "#7f9fd9", "Void": "#e34948"})


def plot_3d(y=180.0, half_width=10.0, vertical_exaggeration=4.0, out=OUT):
    holes = section_holes(y, half_width)
    s = strata()
    s = s[s.hole_id.isin(holes.hole_id)].merge(holes[["hole_id", "x_m", "y_m"]], on="hole_id")
    s = s[s["To base (m)"] > s["From (m)"]]

    fig = go.Figure()
    for cls, colour in COLOURS_3D.items():
        d = s[s["class"] == cls]
        if d.empty:
            continue
        n = len(d)  # one trace per class: vertical segments separated by gaps (None)
        x = np.column_stack([d.x_m, d.x_m, np.full(n, None)]).ravel()
        yy = np.column_stack([d.y_m, d.y_m, np.full(n, None)]).ravel()
        z = np.column_stack([-d["From (m)"], -d["To base (m)"], np.full(n, None)]).ravel()
        tip = [f"{h}<br>{desc} ({cls})<br>{top:g}-{base:g} m" for h, desc, top, base in
               zip(d.hole_id, d.Description, d["From (m)"], d["To base (m)"])]
        text = np.column_stack([tip, tip, np.full(n, None)]).ravel()
        fig.add_trace(go.Scatter3d(x=x, y=yy, z=z, mode="lines", name=f"{cls} ({n})",
                                   line=dict(color=colour, width=7), text=text, hoverinfo="text"))

    fig.add_trace(go.Scatter3d(x=holes.x_m, y=holes.y_m, z=np.zeros(len(holes)), mode="markers",
                               marker=dict(size=1.5, color="#52514e"), name="Hole collars",
                               text=holes.hole_id, hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=[holes.x_m.min(), holes.x_m.max()], y=[y, y], z=[0, 0],
                               mode="lines", line=dict(color="#0b0b0b", width=3, dash="dash"),
                               name=f"y = {y:,.0f} m", hoverinfo="skip"))

    span_x = holes.x_m.max() - holes.x_m.min()
    span_y = holes.y_m.max() - holes.y_m.min()
    depth = s["To base (m)"].quantile(0.995)
    y_stretch = max(1.0, 0.3 / (span_y / span_x))  # a narrow band is a thin slab: stretch y
    fig.update_layout(
        title=dict(text=f"Borehole strata within ±{half_width:g} m of y = {y:,.0f} m "
                        f"({holes.hole_id.nunique()} holes)<br>"
                        f"<sup>Depth below ground level; vertical exaggeration ×{vertical_exaggeration:g}, "
                        f"y stretched ×{y_stretch:.0f}. Drag to rotate, scroll to zoom, hover for layer details."
                        f"</sup>", x=0.01, y=0.985),
        scene=dict(
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="Depth (m)", aspectmode="manual",
            aspectratio=dict(x=1, y=y_stretch * span_y / span_x, z=vertical_exaggeration * depth / span_x),
            xaxis=dict(tickformat=".0f", range=[holes.x_m.min() - 5, holes.x_m.max() + 5]),
            yaxis=dict(tickformat=".0f", nticks=5, range=[holes.y_m.min() - 1, holes.y_m.max() + 1]),
            zaxis=dict(range=[-depth - 2, 1]),
            camera=dict(eye=dict(x=0.55, y=-1.25, z=0.75)),
        ),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.01, y=1.0, xanchor="left", yanchor="bottom",
            buttons=[
                dict(label="Show all", method="restyle", args=[{"visible": [True] * len(fig.data)}]),
                dict(label="Hide overburden & mudstone", method="restyle",
                     args=[{"visible": ["legendonly" if t.name.startswith(("Overburden", "Mudstone")) else True
                                        for t in fig.data]}]),
            ])],
        legend=dict(title="Stratum (click to hide / show)", itemsizing="constant"),
        paper_bgcolor="#fcfcfb", margin=dict(l=0, r=0, t=110, b=0), height=850,
    )
    path = out / f"strata_3d_y{y:.0f}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"{holes.hole_id.nunique()} holes, {len(s)} layers -> {path}")
    return path


if __name__ == "__main__":
    plot_3d(*[float(a) for a in sys.argv[1:3]])
