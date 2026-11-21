"""
Assign every logged coal layer to a seam, site-wide.

Each seam is a gently dipping surface, so after removing a regional plane the
layers separate into distinct levels - one per seam. The steps are:

  1. robust plane fit (iteratively reweighted least squares) to all coal layers,
  2. histogram of the detrended levels -> one peak per seam,
  3. assign each layer to the nearest seam, then refit that seam's own plane and
     reassign, repeating until the assignment settles (a k-planes / EM loop).

Layers further than `tol` from every seam surface stay unassigned.

Seam names follow project_notes.txt: upper, worked and lower seam. Levels are
in metres above the site datum (ground level at the hole minus depth).

Run:  python seam_model.py
Output: output/coal_seam_assignment.csv, output/seam_assignment_doubtful.csv,
        output/seam_extent.png
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from site_data import GRID, INK, INK2, OUT, SURFACE, load_holes, rc, strata, style_axes

E0, N0 = 220.0, 180.0                  # plane fits are centred here (local x, y)


def coal_points():
    """Every coal layer with its position, ground level and mid-point elevation."""
    s = strata()
    coal = s[(s["class"] == "Coal") & (s["To base (m)"] > s["From (m)"])].copy()
    # E, N = local x, y (renamed to x_m, y_m in the CSV files written)
    t = load_holes().set_index("hole_id").rename(columns={"x_m": "E", "y_m": "N",
                                                          "hole_type": "kind"})
    d = coal.join(t[["E", "N", "ground_level_m", "kind"]], on="hole_id").dropna(
        subset=["E", "N", "ground_level_m"])
    d["mid"] = d.ground_level_m - (d["From (m)"] + d["To base (m)"]) / 2
    d["thickness"] = d["To base (m)"] - d["From (m)"]
    d["depth"] = (d["From (m)"] + d["To base (m)"]) / 2
    return d.reset_index(drop=True)


def robust_plane(E, N, Z, iters=8, c=2.0):
    """Plane z = a*(E-E0) + b*(N-N0) + c0 by iteratively reweighted least squares."""
    A = np.c_[E - E0, N - N0, np.ones(len(E))]
    w = np.ones(len(Z))
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A * w[:, None], Z * w, rcond=None)
        r = Z - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9      # robust sigma
        w = 1 / np.sqrt(1 + (r / (c * s)) ** 2)                      # soft (Cauchy) weights
    return coef


def seam_levels(d, bin_width=0.5, min_sep=4.0, prominence=0.02):
    """Peaks in the histogram of detrended levels: one per seam."""
    coef = robust_plane(d.E.values, d.N.values, d["mid"].values)
    res = d["mid"].values - np.c_[d.E - E0, d.N - N0, np.ones(len(d))] @ coef
    bins = np.arange(res.min() - bin_width, res.max() + bin_width, bin_width)
    hist, edges = np.histogram(res, bins=bins, density=True)
    idx, props = find_peaks(hist, distance=int(min_sep / bin_width), prominence=prominence)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return coef, res, centres[idx], hist, centres, props


def fit_seams(d, dip_east=-0.0365, tol=5.0, min_layers=150, iters=12, bin_width=0.5, min_sep=8.0):
    """Assign coal layers to seams by a k-planes loop.

    Started from peaks in the levels detrended with a known regional dip, each
    seam then gets its own robust plane; layers are reassigned to the nearest
    surface within `tol` m and the planes refitted until the assignment settles.
    Seams with fewer than `min_layers` layers are dropped.

    `min_sep` (8 m) keeps the seeds apart: successive coal layers in the same
    borehole are 8-16 m apart (median 12 m), so surfaces closer than that would
    be splitting one seam rather than separating two.
    """
    E, N, Z = d.E.values, d.N.values, d["mid"].values
    res = Z - (dip_east * (E - E0))
    bins = np.arange(res.min() - bin_width, res.max() + bin_width, bin_width)
    hist, edges = np.histogram(res, bins=bins, density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    idx, _ = find_peaks(hist, distance=int(min_sep / bin_width), prominence=0.01)
    planes = [np.array([dip_east, 0.0, c]) for c in centres[idx]]

    A = np.c_[E - E0, N - N0, np.ones(len(E))]
    label = np.full(len(d), -1)
    for _ in range(iters):
        dist = np.abs(np.column_stack([Z - A @ p for p in planes]))
        nearest = dist.argmin(1)
        label = np.where(dist[np.arange(len(d)), nearest] <= tol, nearest, -1)
        keep, planes2 = [], []
        for k in range(len(planes)):
            m = label == k
            if m.sum() >= min_layers:
                planes2.append(robust_plane(E[m], N[m], Z[m]))
                keep.append(k)
        if len(planes2) == len(planes) and all(np.allclose(planes[k], p, atol=1e-6)
                                               for k, p in zip(keep, planes2)):
            planes = planes2
            break
        planes = planes2
    # relabel deepest-first and order by mean level (upper seam = higher level)
    dist = np.abs(np.column_stack([Z - A @ p for p in planes]))
    nearest = dist.argmin(1)
    label = np.where(dist[np.arange(len(d)), nearest] <= tol, nearest, -1)
    order = np.argsort([-(A @ p).mean() for p in planes])          # highest surface first
    remap = {old: new for new, old in enumerate(order)}
    d = d.assign(seam=[remap.get(k, np.nan) if k >= 0 else np.nan for k in label])
    planes = [planes[k] for k in order]
    return d, planes


def seam_table(d, planes):
    rows = []
    for k, p in enumerate(planes):
        m = d.seam == k
        dd = d[m]
        rows.append(dict(seam=k + 1, layers=int(m.sum()), holes=dd.hole_id.nunique(),
                         dip_east_m_per_100m=round(100 * p[0], 2), dip_north_m_per_100m=round(100 * p[1], 2),
                         dip_deg=round(np.degrees(np.arctan(np.hypot(p[0], p[1]))), 2),
                         level_at_centre=round(p[2], 2),
                         depth_median=round(dd.depth.median(), 1),
                         depth_5_95=f"{dd.depth.quantile(0.05):.1f}-{dd.depth.quantile(0.95):.1f}",
                         level_median=round(dd["mid"].median(), 1),
                         thickness_median=round(dd.thickness.median(), 2),
                         e_range=f"{dd.E.min():,.0f}-{dd.E.max():,.0f}",
                         n_range=f"{dd.N.min():,.0f}-{dd.N.max():,.0f}",
                         residual_sd=round(float(np.std(dd["mid"] - (np.c_[dd.E - E0, dd.N - N0,
                                                                          np.ones(len(dd))] @ p))), 2)))
    return pd.DataFrame(rows)


SEAM_NAMES = ["Upper", "Worked", "Lower"]

# Decisions made by hand, keyed by (hole_id, depth to the top of the layer in m).
# These win over whatever the rules work out, and are the place to record a call
# taken on a hole that rule R6 flagged. Two layers of one hole may share a seam:
# that is a seam split into leaves, and the fits then use one point per hole, the
# mid-point of the whole seam from the top of the first leaf to the base of the last.
OVERRIDES = {
    # 6G78 (x 277, y 31): one coal at 18.0-18.5 m and two thin "coal trace" layers at
    # 23.5-23.7 and 26.5-26.7 m with 2.8 m of mudstone between them. Read as the worked
    # seam over the lower seam split into two leaves.
    ("6G78", 18.0): "Worked",
    ("6G78", 23.5): "Lower",
    ("6G78", 26.5): "Lower",
    # 6J80 (x 285, y 31): no coal at worked-seam level at all - overburden to 18.0 m
    # then mudstone to a single coal trace at 23.5-24.0 m, the same depth as the lower
    # seam in 6C74, 6E76 and 6G78 either side.
    ("6J80", 23.5): "Lower",
}


def apply_overrides(d, overrides=OVERRIDES):
    """Force the hand-made calls in `overrides`, and say which layers they touched."""
    key = list(zip(d.hole_id, d["From (m)"].round(2)))
    hit = [overrides.get(k) for k in key]
    d = d.copy()
    d["overridden"] = [h is not None for h in hit]
    d["seam_name"] = [h if h is not None else s for h, s in zip(hit, d.seam_name)]
    d["seam"] = [SEAM_NAMES.index(s) if isinstance(s, str) else np.nan for s in d.seam_name]
    d.loc[d.overridden, "margin"] = np.inf          # no longer in doubt
    missing = set(overrides) - set(key)
    if missing:
        print(f"  warning: {len(missing)} override(s) match no logged layer: {sorted(missing)}")
    return d


def _surfaces(d, planes):
    """Three reference surfaces: the two fitted seams plus the deeper coal.

    `fit_seams` finds the upper seam (east only) and the worked seam (site-wide).
    The layers it leaves more than 5 m below the worked seam are a third population,
    seen only in the holes that drilled deep enough, and get their own plane here.
    """
    p_worked = planes[1]
    below = (p_worked[0] * (d.E - E0) + p_worked[1] * (d.N - N0) + p_worked[2]) - d["mid"]
    deep = d[d.seam.isna() & (below > 5)]
    return [planes[0], planes[1], robust_plane(deep.E.values, deep.N.values, deep["mid"].values)]


def _level(plane, E, N):
    return plane[0] * (np.asarray(E) - E0) + plane[1] * (np.asarray(N) - N0) + plane[2]


def _offsets(d, surfaces, label, radius, k=8):
    """Rule R5: how far each surface sits above or below its plane, locally.

    For every seam, the residuals of the layers already assigned to it are averaged
    over the nearest `k` assigned layers within `radius` metres - measured in plan,
    so holes on neighbouring cross-sections count exactly like holes along the line.
    Returns one callable per seam giving the local level at (E, N).
    """
    from scipy.spatial import cKDTree
    out = []
    for s, plane in enumerate(surfaces):
        m = label == s
        if m.sum() < 5:
            out.append(lambda E, N, p=plane: _level(p, E, N))
            continue
        P = np.c_[d.E.values[m], d.N.values[m]]
        res = d["mid"].values[m] - _level(plane, P[:, 0], P[:, 1])
        tree = cKDTree(P)

        def local(E, N, tree=tree, res=res, plane=plane):
            Q = np.c_[np.atleast_1d(E), np.atleast_1d(N)]
            dist, idx = tree.query(Q, k=min(k, len(res)), distance_upper_bound=radius)
            dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
            ok = np.isfinite(dist)
            w = np.where(ok, 1.0 / np.maximum(dist, 0.5), 0.0)
            r = np.where(ok, res[np.minimum(idx, len(res) - 1)], 0.0)
            tot = w.sum(1)
            shift = np.where(tot > 0, (r * w).sum(1) / np.where(tot > 0, tot, 1), 0.0)
            return _level(plane, Q[:, 0], Q[:, 1]) + shift
        out.append(local)
    return out


def site_assignment(d=None, planes=None, radius=25.0, rounds=6, tol=6.0, min_gap=5.0):
    """Assign every coal layer to a named seam, site-wide.

    Two rules, applied together to all 6,000-odd layers at once rather than line by
    line, so a seam keeps one identity everywhere:

      R1  within a hole the layers keep their order - the shallower layer cannot be
          put in a deeper seam than the one below it. Each hole's layers are matched
          to the seams as an ordered set, choosing the arrangement with the smallest
          total misfit.
      R5  each seam's level at a hole is its regional plane plus the local residual
          of the nearest assigned layers within `radius` m, measured in plan. Those
          neighbours include holes on the cross-sections either side, so a hole whose
          own line is ambiguous takes its answer from the ground around it.
      R11 a layer cannot be given to a seam unless it lies at least `min_gap` m
          below the seam above it. Without this there is nothing to stop a single
          coal at worked-seam level being called the lower seam, and because R5 then
          takes its neighbours as evidence, one such hole recruits the next (a knot of
          holes can end up "lower seam" sitting 1 m under the worked seam). 5 m is well
          clear of the 8-9 m the real lower seam sits at.

    Layers further than `tol` m from their matched surface are left unassigned.
    """
    from itertools import combinations
    if d is None or planes is None:
        d, planes = fit_seams(coal_points())
    surfaces = _surfaces(d, planes)
    d = d.sort_values(["hole_id", "From (m)"]).reset_index(drop=True)
    # seed: fit_seams' own two seams, plus the deeper population
    label = np.where(d.seam.notna(), d.seam.fillna(-1).to_numpy(), -1).astype(int)
    below = _level(planes[1], d.E, d.N) - d["mid"]
    label[(label < 0) & (below > 5).to_numpy()] = 2

    from scipy.spatial import cKDTree
    rows = d.groupby("hole_id").indices
    margin = np.zeros(len(d))
    for _ in range(rounds):
        level = _offsets(d, surfaces, label, radius)
        # R11 only bites against a seam that is actually proved near the hole. In the
        # west the upper seam is absent, so there is nothing there to be too close to,
        # and constraining against its extrapolated plane would push worked-seam coal
        # up into it.
        trees = [cKDTree(np.c_[d.E.values[label == s], d.N.values[label == s]])
                 if (label == s).sum() else None for s in range(len(surfaces))]
        new, marg = label.copy(), margin.copy()
        for idx in rows.values():
            z = d["mid"].values[idx]                      # already deepest-last
            e, n = d.E.values[idx[0]], d.N.values[idx[0]]
            want = [float(f(e, n)[0]) for f in level]
            proved = [t is not None and len(t.query_ball_point((e, n), radius)) > 0
                      for t in trees]
            deepest = len(surfaces) - 1

            def feasible(combo):
                """R11: the deepest seam may not sit within `min_gap` of the one above.

                Only the deepest seam is constrained. The evidence for the rule is
                specific to it - the lower-seam separation is cleanly bimodal, 8-9 m
                for the real thing against 0-1.5 m for the mislabelled - and applying
                it higher up does damage: near the upper-seam subcrop the local
                estimates of the two upper seams genuinely converge, and the
                constraint then pushes worked-seam coal up into the upper seam.
                """
                return all(c != deepest or not proved[c - 1]
                           or (want[c - 1] - z[j]) >= min_gap
                           for j, c in enumerate(combo))

            costs = sorted((sum(abs(z[j] - want[c]) for j, c in enumerate(combo)), combo)
                           for combo in combinations(range(len(surfaces)), len(idx))
                           if feasible(combo))
            if not costs:                                 # more layers than seams
                for j in idx:
                    c = int(np.argmin([abs(d["mid"].values[j] - w) for w in want]))
                    new[j], marg[j] = c, 0.0
                continue
            best = costs[0]
            # how much better the chosen arrangement is than the next one: small
            # means the hole could reasonably be read either way
            gap = (costs[1][0] - best[0]) if len(costs) > 1 else np.inf
            for j, c in zip(idx, best[1]):
                new[j] = c if abs(d["mid"].values[j] - want[c]) <= tol else -1
                marg[j] = gap
        if np.array_equal(new, label):
            label, margin = new, marg
            break
        label, margin = new, marg

    d = d.assign(seam=np.where(label >= 0, label, np.nan), margin=np.round(margin, 2))
    d["seam_name"] = [SEAM_NAMES[k] if k >= 0 else None for k in label]
    return mark_outliers(apply_overrides(d), surfaces), surfaces


def mark_outliers(d, surfaces, radius=25.0, k=6, tol=3.0):
    """Rule R9: ignore a hole whose coal is not at the level its neighbours show.

    The seam's level at a hole is worked out from the nearest `k` *other holes* that
    prove the same seam within `radius` m in plan. A hole more than `tol` m from that
    level is marked `ignored`: it stays in the table and on the figures, but is left
    out of the fits, because a hole that does not show the seam where the holes around
    it do is evidence about that hole, not about the seam.

    Leaves are tested as one horizon (R8): a hole's layers in the same seam are first
    merged to the mid-point of the whole seam, so a hole is never its own neighbour -
    two leaves at the same easting would otherwise pull each other's expected level
    towards the mid-point between them and both look wrong.

    A hand-made override (R6) is never ignored - a decision taken deliberately wins
    over an automatic rejection.
    """
    from scipy.spatial import cKDTree
    m = d.seam.notna()
    g = (d[m].groupby(["hole_id", "seam"], as_index=False)
         .agg(E=("E", "first"), N=("N", "first"), top=("ground_level_m", "first"),
              f=("From (m)", "min"), t=("To base (m)", "max")))
    g["mid"] = g.top - (g.f + g.t) / 2                          # R8: one point per hole per seam

    g["gap"] = np.nan
    for s in range(len(surfaces)):
        sel = (g.seam == s).to_numpy()
        if sel.sum() < k + 1:
            continue
        P = np.c_[g.E.values[sel], g.N.values[sel]]
        Z = g["mid"].values[sel]
        plane = surfaces[s]
        res = Z - _level(plane, P[:, 0], P[:, 1])
        dist, nb = cKDTree(P).query(P, k=min(k + 1, len(Z)), distance_upper_bound=radius)
        dist, nb = np.atleast_2d(dist)[:, 1:], np.atleast_2d(nb)[:, 1:]     # drop self
        ok = np.isfinite(dist)
        w = np.where(ok, 1.0 / np.maximum(dist, 0.5), 0.0)
        r = np.where(ok, res[np.minimum(nb, len(res) - 1)], 0.0)
        tot = w.sum(1)
        local = _level(plane, P[:, 0], P[:, 1]) + np.where(
            tot > 0, (r * w).sum(1) / np.where(tot > 0, tot, 1), 0.0)
        g.loc[sel, "gap"] = np.where(tot > 0, Z - local, np.nan)            # no neighbours -> keep

    d = d.copy()
    d["local_gap"] = np.round(d.set_index(["hole_id", "seam"]).index.map(
        g.set_index(["hole_id", "seam"]).gap).to_numpy(dtype=float), 2)
    d["ignored"] = (d.local_gap.abs() > tol).to_numpy() & ~d.get("overridden", False).to_numpy()
    return d


def doubtful(d, margin=1.5, out=None):
    """Holes where the seam could reasonably be read either way (rule R6).

    `margin` is how much better the chosen arrangement had to be than the next one,
    in metres of total misfit. Anything below the threshold, and anything left
    unassigned, is listed for a decision rather than quietly accepted.
    """
    q = d[(d.margin < margin) | d.seam.isna()].copy()
    q = q.sort_values(["margin", "hole_id"])
    cols = ["hole_id", "E", "N", "ground_level_m", "From (m)", "To base (m)", "mid",
            "depth", "thickness", "seam_name", "margin"]
    if out is not None:
        q[cols].rename(columns={"E": "x_m", "N": "y_m"}).to_csv(out / "seam_assignment_doubtful.csv", index=False)
    return q[cols]


def save_assignment(d, out=OUT):
    """One row per coal layer, so the section fits can look the seam up."""
    cols = ["hole_id", "From (m)", "To base (m)", "E", "N", "ground_level_m", "mid",
            "thickness", "depth", "plot", "kind", "seam", "seam_name", "margin", "overridden",
            "local_gap", "ignored"]
    (d[[c for c in cols if c in d.columns]].rename(columns={"E": "x_m", "N": "y_m", "kind": "hole_type"})
     .to_csv(out / "coal_seam_assignment.csv", index=False))
    return out / "coal_seam_assignment.csv"


def plot_seam_extent(d, planes, out=OUT):
    """Where each seam was proved, one panel per seam, coloured by seam level."""
    rc()
    holes = load_holes().rename(columns={"x_m": "E", "y_m": "N"})

    lo, hi = d["mid"].quantile([0.02, 0.98])
    names = SEAM_NAMES[:len(planes)]
    fig, axes = plt.subplots(1, len(names), figsize=(8 * len(names), 8.4), facecolor=SURFACE,
                             sharex=True, sharey=True)
    for k, (ax, name) in enumerate(zip(np.atleast_1d(axes), names)):
        dd = d[d.seam == k]
        if dd.empty:
            continue
        ax.set_facecolor(SURFACE)
        ax.scatter(holes.E, holes.N, s=1.5, color=GRID, linewidths=0, zorder=1)
        sc = ax.scatter(dd.E, dd.N, s=6, c=dd["mid"], cmap="viridis", vmin=lo, vmax=hi,
                        linewidths=0, zorder=3)
        p = planes[k]
        ee, nn = np.meshgrid(np.linspace(holes.E.min(), holes.E.max(), 60),
                             np.linspace(holes.N.min(), holes.N.max(), 60))
        cs = ax.contour(ee, nn, p[0] * (ee - E0) + p[1] * (nn - N0) + p[2],
                        levels=np.arange(np.floor(lo / 5) * 5 - 20, hi + 20, 5), colors=INK,
                        linewidths=0.7, zorder=4)
        ax.clabel(cs, fmt="%d m", fontsize=8, colors=INK)
        ax.set_aspect("equal")
        style_axes(ax, lw=0.5)
        ax.set_xlabel("x (m, local grid)", color=INK2)
        ax.set_title(f"{name} - {len(dd):,} layers in {dd.hole_id.nunique():,} holes\n"
                     f"x {dd.E.min():,.0f}-{dd.E.max():,.0f} m, depth "
                     f"{dd.depth.quantile(.05):.0f}-{dd.depth.quantile(.95):.0f} m "
                     f"(median {dd.depth.median():.0f} m)",
                     loc="left", fontsize=12, color=INK, pad=10)
    np.atleast_1d(axes)[0].set_ylabel("y (m, local grid)", color=INK2)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Seam mid-point level (m above site datum)", color=INK2)
    cb.outline.set_edgecolor(GRID)
    fig.suptitle(f"Extent of the {len(names)} coal seams (contours: fitted plane, m above site datum)", x=0.06, ha="left",
                 fontsize=15, color=INK)
    fig.savefig(out / "seam_extent.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    d = coal_points()
    coef, res, peaks, hist, centres, props = seam_levels(d)
    print(f"{len(d):,} coal layers in {d.hole_id.nunique():,} holes")
    print(f"regional plane: {100 * coef[0]:+.2f} m per 100 m east, {100 * coef[1]:+.2f} m per 100 m north")
    print(f"detrended levels: {len(peaks)} peaks at {np.round(peaks, 1)} m "
          f"(prominences {np.round(props['prominences'], 3)})")
    for p in peaks:
        n = int((np.abs(res - p) < 2).sum())
        print(f"   level {p:+6.1f} m: {n:5d} layers within 2 m")

    d, planes = fit_seams(d)
    print(f"\nk-planes result: {len(planes)} seams, "
          f"{int(d.seam.notna().sum()):,} layers assigned, {int(d.seam.isna().sum()):,} unassigned")
    print(seam_table(d, planes).to_string(index=False))

    d, surfaces = site_assignment(d, planes)
    print(f"\nsite-wide assignment (R1 + R5): {len(surfaces)} seams, "
          f"{int(d.seam.notna().sum()):,} layers assigned, {int(d.seam.isna().sum()):,} unassigned")
    print(seam_table(d, surfaces).assign(name=SEAM_NAMES[:len(surfaces)]).to_string(index=False))
    q = doubtful(d, out=OUT)
    print(f"\n{len(q)} layers ({100 * len(q) / len(d):.1f} %) in {q.hole_id.nunique()} holes "
          f"flagged for review -> {OUT / 'seam_assignment_doubtful.csv'}")
    print("wrote", save_assignment(d))
    plot_seam_extent(d, surfaces)
    print("wrote", OUT / "seam_extent.png")
