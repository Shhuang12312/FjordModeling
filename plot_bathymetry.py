"""
plot_bathymetry.py - plan view and along-fjord section of a bottom file for fjord.jl

Draws the bathymetry in model coordinates, as fjord.jl sees it, turned to match a map of a
fjord that runs west to the sea: mouth (y = Ly) on the left, glacier wall (y = 0) on the
right, x up (x points to the right looking down-fjord, so for Ilulissat x is roughly north).
Underneath is a section along the
thalweg: the path from the glacier wall to the mouth that keeps to the deepest water (so it
crosses the sill at its deepest pass), drawn dashed on the map. Works on any bottom file, including bottom.nc from
make_bathymetry.jl. If the file has a BedMachine `mask` (from make_bottom.py),
ice is drawn white and land grey.

    python plot_bathymetry.py bottom_ilulissat_large_bedmachine.nc --out Plots/ilulissat/bathymetry_ilulissat_large_bedmachine.png
    python plot_bathymetry.py bottom_ilulissat_large_bedmachine.nc --plume-x 13575 --sill 260 --gl 845 \
        --casts omg_casts/OMG_L2_CTD/OMG_Ocean_CTD_L2_20200414151158.nc

--casts places OMG casts by their lat/lon, so it needs a file from make_bottom.py (which
stores the glacier and mouth positions). Needs numpy, matplotlib and netCDF4.
"""

import argparse
import heapq
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTH_CMAP = LinearSegmentedColormap.from_list(      # same colours as plot_domain.py
    "depth", [(0, "#1d2f8f"), (.25, "#29a3e0"), (.5, "#f2e394"), (.75, "#f59a3c"), (1, "#fbd9b5")])


def cast_positions(paths, ds):
    """(name, x, y, max depth) of each cast in the box of a make_bottom.py file."""
    from make_bottom import FjordPath, latlon_to_xy
    sys.path.insert(0, os.path.join(HERE, "files"))
    import omg_profile_fit as opf
    path = FjordPath.from_file(ds)                      # straight or bent (--via) box
    out = []
    for p in paths:
        c = opf.read_cast(p)
        cx, cy = path.from_xy(*latlon_to_xy(c["lat"], c["lon"]))
        out.append((os.path.basename(p), cx, cy, np.nanmax(c["zraw"])))
    return out


_MOVES = [(dj, di, math.hypot(dj, di)) for dj in (-1, 0, 1) for di in (-1, 0, 1) if dj or di]


def _dijkstra(B, start, step_cost, better):
    """Best path from `start` to the last row of B over water cells. `step_cost(c, depth, ds)`
    gives the new path value; `better(a, b)` says whether a beats b. Returns (value, path)."""
    ny, nx = B.shape
    worst = math.inf if better(0, 1) else -math.inf
    val = np.full(B.shape, worst)
    prev = -np.ones(B.shape, dtype=np.int64)
    val[start] = step_cost(None, -B[start], 0.0)
    sign = 1 if better(0, 1) else -1                       # heapq pops the smallest
    heap = [(sign * val[start], *start)]
    while heap:
        c, j, i = heapq.heappop(heap)
        c *= sign
        if c != val[j, i]:
            continue
        if j == ny - 1:
            path, k = [], j * nx + i
            while k >= 0:
                path.append(divmod(k, nx))
                k = prev[path[-1]]
            return c, np.array(path[::-1])
        for dj, di, ds in _MOVES:
            jj, ii = j + dj, i + di
            if 0 <= jj < ny and 0 <= ii < nx and B[jj, ii] < 0:
                cc = step_cost(c, -B[jj, ii], ds)
                if better(cc, val[jj, ii]):
                    val[jj, ii], prev[jj, ii] = cc, j * nx + i
                    heapq.heappush(heap, (sign * cc, jj, ii))
    return None, None


def thalweg(b, x, y, x0=None, step=2, power=8, start=None):
    """Path through the water from the glacier wall (at x0, else its deepest cell) to the mouth
    row, on every `step`-th cell. First finds the sill: the deepest level at which the two stay
    connected. Then, using only water at least that deep, the path that keeps to deep water:
    a move costs its length times (deepest / depth)^power, so the path takes a long way round
    rather than cross a shallower spot (it follows the deep channel, not just any route deeper
    than the sill). `start` = (x, y) starts it at the water cell nearest that point instead (a
    glacier on a side wall). Returns the sill depth, and the depths and x, y of the path's cells."""
    B = b[::step, ::step]                                  # (y, x)
    xs, ys = x[::step], y[::step]
    if start is not None:
        d2 = np.where(B < 0, (xs[None, :] - start[0]) ** 2 + (ys[:, None] - start[1]) ** 2, np.inf)
        j0, i0 = np.unravel_index(int(np.argmin(d2)), B.shape)
    else:
        j0 = int(np.argmax((B < 0).any(axis=1)))               # first row with water
        i0 = (int(np.argmin(np.where(B[j0] < 0, np.abs(xs - x0), np.inf))) if x0 is not None
              else int(np.argmin(np.where(B[j0] < 0, B[j0], np.inf))))
    # 1. sill: maximise the shallowest depth met on the way out
    sill, _ = _dijkstra(B, (j0, i0), lambda c, d, ds: d if c is None else min(c, d), lambda a, b: a > b)
    if sill is None:
        return None
    # 2. deep-water path through cells no shallower than the sill
    deep = np.where(-B >= sill - 1e-6, B, 0.0)
    dmax = -B.min()
    _, path = _dijkstra(deep, (j0, i0), lambda c, d, ds: 0.0 if c is None else c + ds * (dmax / d) ** power,
                        lambda a, b: a < b)
    return sill, B[path[:, 0], path[:, 1]], xs[path[:, 1]], ys[path[:, 0]]


def plot_bathymetry(path, out, plume_x=None, sill=None, gl=None, casts=(), plume_y=None, plume_offset=0.0):
    with netCDF4.Dataset(path) as ds:
        x, y = ds["x"][:] / 1e3, ds["y"][:] / 1e3
        b = np.asarray(ds["bottom"][:], dtype=float)           # (y, x)
        mask = np.asarray(ds["mask"][:]) if "mask" in ds.variables else None
        where = cast_positions(casts, ds) if casts else []
        wall = str(getattr(ds, "glacier_wall", "y0"))
    wet = b < 0
    Lx, Ly = x[-1] + x[0], y[-1] + y[0]
    side = wall != "y0"                  # glacier on a side wall: the section runs from it, not along y
    if side and plume_y is None:
        raise SystemExit(f"this file's glacier is on the {wall} wall: pass --plume-y (plume_y_center)")
    off = plume_offset / 1e3                 # ice front this far out from the wall
    gx_km = (Lx - off if wall == "xL" else off) if side else None

    fig, (ax, sx) = plt.subplots(2, 1, figsize=(13, 9.5), sharex=not side,
                                 gridspec_kw=dict(height_ratios=[Lx / Ly * 2.2, 1]), constrained_layout=True)

    # plan view: y (along) across the page, x (across) up the page
    ax.set_facecolor("0.8")
    if mask is not None:
        ice = np.isin(mask, [2, 3]) & ~wet
        ax.pcolormesh(y, x, np.where(ice, 1.0, np.nan).T, cmap=ListedColormap(["#f4f7fa"]),
                      shading="nearest", rasterized=True)
    pc = ax.pcolormesh(y, x, np.where(wet, b, np.nan).T, cmap=DEPTH_CMAP, vmin=-1000, vmax=0,
                       shading="nearest", rasterized=True)
    ax.contour(y, x, b.T, [-500, -250], colors="k", linewidths=0.4, alpha=0.5)
    tw = thalweg(b, x, y, None if plume_x is None else plume_x / 1e3,
                 start=(gx_km, plume_y / 1e3) if side else None)
    if tw is not None:
        ax.plot(tw[3], tw[2], "k--", lw=1, label="thalweg (section below)")
    if side:
        ax.plot(plume_y / 1e3, gx_km, "r*", ms=16, clip_on=False, zorder=5,
                label=f"plume on the {wall} wall, y = {plume_y:.0f} m"
                       + (f", {plume_offset / 1e3:.1f} km out" if plume_offset else ""))
    elif plume_x is not None:
        ax.plot(0, plume_x / 1e3, "r*", ms=16, clip_on=False, zorder=5, label=f"plume, x = {plume_x:.0f} m")
    for name, cx, cy, _ in where:
        ax.plot(cy / 1e3, cx / 1e3, "ko", mfc="w", ms=7, zorder=5)
        ax.annotate(name.split("_")[-1][:8], (cy / 1e3, cx / 1e3), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlim=(Ly, 0), ylim=(0, Lx), aspect="equal", ylabel="x across [km]",
           title=f"{os.path.basename(path)}: Lx = {Lx:.1f} km, Ly = {Ly:.1f} km, deepest {-b.min():.0f} m"
                 + ("   (white = ice, grey = land)" if mask is not None else "   (grey = land)"))
    if not side:
        ax.text(0.997, 0.02, "glacier", transform=ax.transAxes, fontsize=9, color="r", ha="right")
    ax.text(0.003, 0.02, "mouth", transform=ax.transAxes, fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    fig.colorbar(pc, ax=ax, fraction=0.025, pad=0.01, label="bottom [m]")

    # along-fjord section along the thalweg (against y, so it lines up with the map)
    if tw is not None:
        sy, deepest = tw[3], tw[1]
        if side:                         # distance along the path from the glacier
            sy = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(tw[2]), np.diff(tw[3])))])
        crest = int(np.argmax(deepest))
        f = sy[crest] / sy[-1]
        sx.annotate(f"shallowest on the way out: {tw[0]:.0f} m", (sy[crest], deepest[crest]),
                    xytext=(0, 18), textcoords="offset points", fontsize=9,
                    ha="right" if f < 0.15 else "left" if f > 0.85 else "center",   # y runs right to left
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    else:
        sy, deepest = y, np.where(wet.any(axis=1), b.min(axis=1), np.nan)
    sx.fill_between(sy, deepest, -1.05 * np.nanmax(-deepest), color="0.55", lw=0)
    sx.plot(sy, deepest, "k", lw=1)
    sx.axhline(0, color="#29a3e0", lw=1)
    if sill is not None:
        sx.axhline(-sill, color="tab:orange", ls="--", lw=1.2, label=f"sill depth {sill:.0f} m")
    if gl is not None:
        sx.plot(0, -gl, "r<", ms=10, clip_on=False, zorder=5, label=f"grounding line {gl:.0f} m")
    for name, cx, cy, zmax in where:
        sx.plot([cy / 1e3] * 2, [0, -zmax], color="k", lw=2.5, alpha=0.6)
        sx.annotate(f"{name.split('_')[-1][:8]}\n{zmax:.0f} m", (cy / 1e3, -zmax), xytext=(4, -4),
                    textcoords="offset points", fontsize=8, va="top")
    sx.set(xlim=(sy[-1] if side else Ly, 0), ylim=(-1.05 * np.nanmax(-deepest), 20),
           xlabel="distance along the thalweg from the glacier [km]" if side else "y along fjord from glacier wall [km]",
           ylabel="depth [m]", title="section along the thalweg (casts as bars to their maximum depth)")
    sx.grid(alpha=0.3)
    if sill is not None or gl is not None:
        sx.legend(loc="lower left", fontsize=9)

    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bathymetry", help="bottom file, e.g. bottom_ilulissat_large_bedmachine.nc")
    p.add_argument("--out", help="PNG path (default Plots/<fjord>/bathymetry_<file>.png)")
    p.add_argument("--plume-x", type=float, help="plume_x_center [m], marked at the glacier wall")
    p.add_argument("--plume-y", type=float, help="plume_y_center [m], for a glacier on a side wall")
    p.add_argument("--plume-offset", type=float, default=0.0,
                   help="plume_wall_offset [m]: the ice front's distance out from that wall")
    p.add_argument("--sill", type=float, help="sill depth [m], drawn on the section")
    p.add_argument("--gl", type=float, help="grounding-line depth [m], marked at the glacier wall")
    p.add_argument("--casts", nargs="+", default=(), help="OMG cast files to place on the map and section")
    a = p.parse_args()
    name = os.path.splitext(os.path.basename(a.bathymetry))[0].removeprefix("bottom_")
    out = a.out or os.path.join(HERE, "Plots", name.split("_")[0], f"bathymetry_{name}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plot_bathymetry(a.bathymetry, out, a.plume_x, a.sill, a.gl, a.casts, a.plume_y, a.plume_offset)
