"""
make_fjord_entry.py - measure a fjord's entry for files/fjords.json from a make_bottom.py file

files/omg_select_casts.py sorts OMG casts into 'inside' and 'outside' the fjord using a
centerline, the sill's position along it, and the sill and grounding-line depths. This script
measures all of them from a BedMachine bottom file instead of by hand:

  1. the ice front: where the glacier meets the water (the plume position make_bottom.py prints),
  2. the thalweg: the path from the front to the box's mouth row that keeps to the deepest water
     (plot_bathymetry.thalweg, the dashed line on the bathymetry plots),
  3. H_sill: the shallowest depth deep water from the front has to pass on its way to the mouth
     (the deepest level at which the front stays connected to the mouth row),
  4. the centerline: the thalweg cut down to a few points (lat/lon, mouth first, glacier last),
  5. sill_km: where the sill crest is along that centerline, measured as omg_select_casts.py
     measures casts (the crest closest to the glacier, if the path sits at sill depth for a while),
  6. H_gl: the deepest water at the ice front,
  7. max_cross_km: about half the fjord's width (90th percentile of the distance from the
     thalweg to the nearest land, from the ice front until the path reaches open water, more
     than 8 km from land).
shelf_km and shelf_cross_km are not measured: they say how far onto the shelf 'outside' casts
may come from, and keep the defaults unless given.

    python3 make_fjord_entry.py bottom_sermilik_bedmachine.nc              # print the entry
    python3 make_fjord_entry.py bottom_sermilik_bedmachine.nc --write      # and put it in fjords.json
    python3 make_fjord_entry.py bottom_ilulissat_large_bedmachine.nc --H-gl 845 --write

The entry is named after the file (bottom_<name>_....nc -> <name>); --name overrides. It also
saves Plots/<name>/fjord_entry_<name>.png: the bathymetry with the thalweg, centerline and sill
crest, to check them. Needs numpy, scipy, matplotlib and netCDF4.
"""
import argparse
import json
import os
import re
import sys
from datetime import date

import numpy as np
import netCDF4
from scipy.ndimage import distance_transform_edt

from make_bottom import FjordPath, ice_front, side_front, latlon_to_xy, xy_to_latlon
from plot_bathymetry import DEPTH_CMAP, thalweg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "files"))
import omg_select_casts as osc   # noqa: E402

FJORDS_JSON = os.path.join(HERE, "files", "fjords.json")


def simplify(px, py, tol):
    """Ramer-Douglas-Peucker: indices of the points that keep the path within `tol` of the original."""
    keep = {0, len(px) - 1}
    stack = [(0, len(px) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        dx, dy = px[b] - px[a], py[b] - py[a]
        L = np.hypot(dx, dy) or 1.0
        d = np.abs(dy * (px[a + 1:b] - px[a]) - dx * (py[a + 1:b] - py[a])) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            keep.add(a + 1 + k)
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return sorted(keep)


def measure(path, max_points=10, step=2, open_km=8.0):
    with netCDF4.Dataset(path) as ds:
        x, y = np.asarray(ds["x"][:], float), np.asarray(ds["y"][:], float)
        b = np.asarray(ds["bottom"][:], float)                        # (y, x)
        if "mask" not in ds.variables:
            raise SystemExit(f"{path} has no BedMachine mask: make it with make_bottom.py (BedMachine)")
        mask = np.asarray(ds["mask"][:])
        wall = str(getattr(ds, "glacier_wall", "y0"))
        fp = FjordPath.from_file(ds)
        glacier = tuple(ds.glacier_latlon)
        ice = tuple(int(v) for v in np.ravel(getattr(ds["mask"], "flag_values", [0, 1, 2, 3])) if v in (2, 3))

    # 1. ice front (same rules as make_bottom.py uses to place the plume)
    if wall == "y0":
        f = ice_front(b.T, mask.T, ice, x, y)
        if f is None:
            raise SystemExit("no ice front found at the glacier end")
        fx, fy, H_gl = f
    else:
        _, gy = fp.from_xy(*latlon_to_xy(*glacier))
        f = side_front(b.T, mask.T, ice, x, y, wall, gy)
        if f is None:
            raise SystemExit(f"no ice front found on the {wall} wall near the glacier")
        fy, _, H_gl, fx = f

    # 2-3. thalweg and sill depth
    tw = thalweg(b, x, y, start=(fx, fy), step=step)
    if tw is None:
        raise SystemExit("the ice front is not connected to the mouth row")
    H_sill, B, px, py = tw
    depth = -np.asarray(B, float)                                     # along the path, glacier -> mouth

    # 4. centerline: simplify the path (tolerance grows until it is short enough), mouth first
    tol = 250.0
    while True:
        idx = simplify(px, py, tol)
        if len(idx) <= max_points:
            break
        tol *= 1.5
    idx = idx[::-1]                                                   # mouth -> glacier
    latlon = [xy_to_latlon(*(float(v) for v in fp.to_xy(px[i], py[i]))) for i in idx]
    centerline = np.round(np.array(latlon), 4)

    # 5. sill crest: the cell at sill depth (within 10 m) closest to the glacier
    crest = int(np.flatnonzero(depth <= H_sill + 10.0)[0])
    c_lat, c_lon = xy_to_latlon(*(float(v) for v in fp.to_xy(px[crest], py[crest])))
    sill_km, _ = osc.along_cross(c_lat, c_lon, centerline)

    # 7. half-width: distance from the thalweg to land, from the front until the path leaves the
    #    fjord (land more than `open_km` away: open shelf or bay)
    dist = distance_transform_edt(b < 0, sampling=(y[1] - y[0], x[1] - x[0]))
    jj = np.searchsorted(y, py).clip(0, len(y) - 1)
    ii = np.searchsorted(x, px).clip(0, len(x) - 1)
    d = dist[jj, ii]
    out = np.flatnonzero(d > open_km * 1e3)
    d = d[:out[0]] if len(out) and out[0] > 0 else d
    max_cross = max(1.0, round(2 * float(np.percentile(d, 90)) / 1e3) / 2)   # nearest 0.5 km

    return dict(wall=wall, front=(fx, fy), H_gl=float(H_gl), H_sill=float(H_sill), sill_km=float(sill_km),
                centerline=centerline, max_cross_km=max_cross, path=(px, py, depth), crest=crest,
                crest_latlon=(c_lat, c_lon), idx=idx, b=b, x=x, y=y, mask=mask)


def plot(m, path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    x, y, b, mask = m["x"] / 1e3, m["y"] / 1e3, m["b"], m["mask"]
    px, py, depth = m["path"]
    wet = b < 0
    Lx, Ly = x[-1] + x[0], y[-1] + y[0]
    fig, (ax, sx) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw=dict(width_ratios=[1, 1.3]),
                                 constrained_layout=True)
    ax.set_facecolor("0.8")
    ax.pcolormesh(x, y, np.where(np.isin(mask, [2, 3]) & ~wet, 1.0, np.nan), cmap=ListedColormap(["#f4f7fa"]),
                  shading="nearest", rasterized=True)
    pc = ax.pcolormesh(x, y, np.where(wet, b, np.nan), cmap=DEPTH_CMAP, vmin=-1000, vmax=0,
                       shading="nearest", rasterized=True)
    ax.contour(x, y, b, [-m["H_sill"]], colors="tab:orange", linewidths=0.6)
    ax.plot(px / 1e3, py / 1e3, "k--", lw=0.8, label="thalweg")
    ax.plot(px[m["idx"]] / 1e3, py[m["idx"]] / 1e3, "o-", color="m", ms=5, lw=1.5, label="centerline (fjords.json)")
    ax.plot(px[m["crest"]] / 1e3, py[m["crest"]] / 1e3, "D", color="tab:orange", ms=9,
            label=f"sill crest, {m['H_sill']:.0f} m (orange contour), {m['sill_km']:.1f} km along")
    ax.plot(m["front"][0] / 1e3, m["front"][1] / 1e3, "r*", ms=15, label=f"ice front, {m['H_gl']:.0f} m")
    ax.set(xlim=(0, Lx), ylim=(0, Ly), aspect="equal", xlabel="x across [km]", ylabel="y along [km] (mouth at top)",
           title=f"{os.path.basename(path)}: fjords.json entry (white = ice, grey = land)")
    ax.legend(loc="lower left", fontsize=9)
    fig.colorbar(pc, ax=ax, fraction=0.03, pad=0.01, label="bottom [m]")
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))]) / 1e3
    sx.fill_between(s, -depth, -1.05 * depth.max(), color="0.55", lw=0)
    sx.plot(s, -depth, "k", lw=1)
    sx.axhline(-m["H_sill"], color="tab:orange", ls="--", lw=1, label=f"sill {m['H_sill']:.0f} m")
    sx.plot(s[m["crest"]], -depth[m["crest"]], "D", color="tab:orange", ms=8)
    sx.plot(0, -m["H_gl"], "r<", ms=10, clip_on=False, label=f"ice front {m['H_gl']:.0f} m")
    sx.set(xlim=(0, s[-1]), ylim=(-1.05 * depth.max(), 20), xlabel="distance along the thalweg from the ice front [km]",
           ylabel="depth [m]")
    sx.grid(alpha=0.3)
    sx.legend(loc="lower right", fontsize=9)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


def dump(db):
    """fjords.json text: indent 2, each [lat, lon] pair on one line."""
    s = json.dumps(db, indent=2, ensure_ascii=False)
    return re.sub(r"\[\s*(-?[\d.eE+-]+),\s*(-?[\d.eE+-]+)\s*\]", r"[\1, \2]", s) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bathymetry", help="bottom file from make_bottom.py (BedMachine, with its mask)")
    p.add_argument("--name", help="entry name in fjords.json (default: from the file name)")
    p.add_argument("--H-gl", type=float, help="use this grounding-line depth [m] instead of the measured one")
    p.add_argument("--shelf-km", type=float, default=40.0)
    p.add_argument("--shelf-cross-km", type=float, default=30.0)
    p.add_argument("--points", type=int, default=10, help="most centerline points (default 10)")
    p.add_argument("--write", action="store_true", help=f"add/replace the entry in {os.path.relpath(FJORDS_JSON)}")
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    name = a.name or os.path.basename(a.bathymetry).removeprefix("bottom_").split("_")[0].split(".")[0]
    m = measure(a.bathymetry, a.points)
    H_gl = a.H_gl if a.H_gl is not None else round(m["H_gl"])
    gl_note = (f"H_gl {H_gl:.0f} m given on the command line (BedMachine: {m['H_gl']:.0f} m)"
               if a.H_gl is not None else "H_gl = deepest water at the BedMachine ice front")
    entry = {
        "_note": (f"Measured by make_fjord_entry.py from {os.path.basename(a.bathymetry)} (BedMachine) on "
                  f"{date.today()}: centerline = thalweg from the box's mouth row to the ice front, cut to "
                  f"{len(m['centerline'])} points; H_sill = shallowest depth on the deepest route out; sill_km = "
                  f"its crest along the centerline; {gl_note}; max_cross_km ~ half the fjord width. "
                  f"shelf_km and shelf_cross_km are defaults."),
        "centerline": m["centerline"].tolist(),
        "sill_km": round(m["sill_km"], 1),
        "H_sill": round(m["H_sill"]),
        "H_gl": H_gl,
        "max_cross_km": m["max_cross_km"],
        "shelf_km": a.shelf_km,
        "shelf_cross_km": a.shelf_cross_km,
    }
    print(f'"{name}": ' + dump(entry).rstrip())
    fx, fy = m["front"]
    print(f"\nice front at x = {fx:.0f} m, y = {fy:.0f} m ({m['wall']} wall); sill crest at "
          f"{m['crest_latlon'][0]:.4f}, {m['crest_latlon'][1]:.4f}")

    if not a.no_plot:
        plot(m, a.bathymetry, os.path.join(HERE, "Plots", name, f"fjord_entry_{name}.png"))
    if a.write:
        with open(FJORDS_JSON) as fh:
            db = json.load(fh)
        new = name not in db
        db[name] = entry
        if new and "my_fjord" in db:                                  # keep the template last
            db["my_fjord"] = db.pop("my_fjord")
        with open(FJORDS_JSON, "w") as fh:
            fh.write(dump(db))
        print(f"{'added' if new else 'replaced'} '{name}' in {os.path.relpath(FJORDS_JSON)}")


if __name__ == "__main__":
    main()
