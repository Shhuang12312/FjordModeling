"""
plot_domain.py - see where a model box sits in Greenland, and try out its dimensions

Draws a Greenland map (BedMachine depth, ice and land greyed out) with the model box on it, a
zoom of the box over BedMachine's bathymetry (ice white), and optionally the surface (or
outflow-depth) vorticity zeta/f from a finished run, placed in its true position and
orientation. Use it to choose the box before running make_bottom.py; it prints the matching
command and FjordConfig values. Everything comes from BedMachine Greenland
(~/Downloads/BedMachineGreenland-v7.nc), the same data make_bottom.py samples.

From Python (e.g. a notebook or the REPL):

    from plot_domain import plot_domain
    plot_domain(width=10_000)                              # Jakobshavn (table row 1), default mouth
    plot_domain(width=14_000, length=45_000)               # wider, shorter box
    plot_domain(bathymetry="bottom_ilulissat.nc", run="output_ilulissat_smoke")   # overlay a run

From the shell:

    python3 plot_domain.py --width 14000 --length 45000
    python3 plot_domain.py --bathymetry bottom_ilulissat.nc --run output_ilulissat_smoke --level outflow

Reading the model output (NetCDF-4) goes through Julia, since this Python has no NetCDF-4
reader; it needs `julia +1.12` and this project's environment.
"""

import argparse
import glob
import math
import os
import subprocess
import tempfile

import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import netcdf_file

from make_bottom import DEFAULT_BEDMACHINE, DEFAULT_CSV, BedMachine, FjordPath, latlon_to_xy, read_fjord

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
DEFAULT_MOUTH = (69.155, -51.12)       # Ilulissat Icefjord mouth, from the Slater et al. (2022) SI map
CORIOLIS_F = 1.22e-4                   # fjord.jl's coriolis_f

DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "depth", [(0, "#1d2f8f"), (.25, "#29a3e0"), (.5, "#f2e394"), (.75, "#f59a3c"), (1, "#fbd9b5")])

_C45 = math.sqrt(0.5)


def to_map(x, y):
    """Polar stereographic true at 75 N, lon_0 = 0 (EPSG:3996, make_bottom.py's working projection)
    [m] -> the same projection turned to lon_0 = -45 [km], so Greenland stands upright as in
    EPSG:3413 maps."""
    return (x * _C45 - y * _C45) / 1e3, (y * _C45 + x * _C45) / 1e3


def from_map(X, Y):
    """Inverse of to_map: map km -> EPSG:3996 metres."""
    X, Y = np.asarray(X) * 1e3, np.asarray(Y) * 1e3
    return (X + Y) * _C45, (Y - X) * _C45


class Box:
    """The model domain: glacier at y = 0, mouth at y = Ly, x across from 0 to Lx (as in fjord.jl).
    Lengths are metres on the ground; `k` (projected metres per ground metre) defaults to the
    projection's scale factor at the box, as in make_bottom.py. Pass k = 1 for bottom files made
    before make_bottom.py measured on the ground (they have no `projection_scale`). With `via`
    points the box follows a bent fjord, as make_bottom.py --via builds it."""

    def __init__(self, glacier, mouth, width, length=None, k=None, via=(), smooth=None, x_center=None):
        self.glacier, self.mouth, self.via = tuple(glacier), tuple(mouth), [tuple(p) for p in via]
        self.path = FjordPath([self.glacier, *self.via, self.mouth], width, length, k=k, smooth=smooth,
                              x_center=x_center)
        self.k, self.Lx, self.Ly, self.x0 = self.path.k, self.path.Lx, self.path.Ly, self.path.x0

    def model_to_map(self, xm, ym):
        """Model (x, y) [m] -> map km."""
        return to_map(*self.path.to_xy(xm, ym))

    def outline(self):
        t = np.linspace(0, 1, 200)
        xs = np.concatenate([0 * t, self.Lx * t, self.Lx + 0 * t, self.Lx * (1 - t)])
        ys = np.concatenate([self.Ly * t, self.Ly + 0 * t, self.Ly * (1 - t), 0 * t])
        return self.model_to_map(xs, ys)


def _julia_export(src, exprs, dst):
    """Copy arrays out of a NetCDF-4 file into a NetCDF-3 file scipy can read, via Julia.
    `exprs` maps output name -> Julia expression in terms of g("var") (missing -> NaN)."""
    lines = ["using NCDatasets",
             "g(k) = Float64.(replace(Array(d[k]), missing => NaN))",
             f'd = NCDataset("{src}")',
             f'out = NCDataset("{dst}", "c"; format = :netcdf3_64bit_offset)']
    for name, expr in exprs.items():
        lines += [f"a = {expr}",
                  f'dims = Tuple("{name}_$i" for i in 1:ndims(a))',
                  "for (n, s) in zip(dims, size(a)); defDim(out, n, s); end",
                  f'defVar(out, "{name}", a, dims)']
    lines += ["close(out); close(d)"]
    subprocess.run(["julia", "+1.12", f"--project={HERE}", "-e", "\n".join(lines)],
                   check=True, capture_output=True, text=True)
    with netcdf_file(dst, "r", mmap=False) as ds:
        return {k: ds.variables[k][:].copy() for k in ds.variables}


def _overview(bm, step=27):
    """BedMachine bed and mask every `step` cells (~4 km) over all of Greenland, on map km."""
    bed, mask = (np.asarray(bm.ds[v][::step, ::step]) for v in ("bed", "mask"))
    s = _C45 / bm.c / 1e3                    # EPSG:3413 metres -> map km (to_map's scale)
    X, Y = np.meshgrid(bm.x[::step] * s, bm.y[::step] * s)
    return X, Y, bed, mask


def _run_vorticity(run, level, box):
    """zeta/f at the last snapshot of <run>/*_<level>.nc, on map km."""
    files = sorted(glob.glob(os.path.join(run, f"*_{level}.nc")))
    if not files:
        raise SystemExit(f"no *_{level}.nc in {run}")
    with tempfile.TemporaryDirectory() as tmp:
        V = _julia_export(files[0], {
            "u": 'g("u")[:, :, 1, end]', "v": 'g("v")[:, :, 1, end]', "bot": 'g("bottom_height")',
            "xc": 'g("x_caa")', "yc": 'g("y_aca")', "t": 'g("time")'}, os.path.join(tmp, "s.nc"))
    u, v, bot = V["u"], V["v"], V["bot"]               # python order: (yc, xf), (yf, xc), (yc, xc)
    uc = .5 * (u[:, 1:] + u[:, :-1])
    vc = .5 * (v[1:, :] + v[:-1, :])
    zeta = np.gradient(vc, V["xc"], axis=1) - np.gradient(uc, V["yc"], axis=0)
    zf = np.where(bot < 0, zeta / CORIOLIS_F, np.nan)
    FX, FY = box.model_to_map(*np.meshgrid(V["xc"], V["yc"]))
    return FX, FY, zf, V["t"][-1], os.path.basename(files[0])


def plot_domain(width=None, length=None, mouth=None, glacier=None, row=1, bathymetry=None,
                run=None, level="surface", margin=8_000, out="domain.png", show=False,
                csv=DEFAULT_CSV, bedmachine=DEFAULT_BEDMACHINE, via=(), smooth=None,
                glacier_x=None):
    """
    Draw the model box on a Greenland map, a zoom over the BedMachine bathymetry, and optionally a
    run's zeta/f. Returns a dict with Lx, Ly, Lz and the commands to build/run this box.

    width      across-fjord box width [m]
    length     along-fjord box length [m]; default = glacier-to-mouth distance
    mouth      (lat, lon) the box points toward; default = Ilulissat Icefjord mouth
    glacier    (lat, lon) of the glacier end; default = row `row` of the Slater et al. table
    bathymetry a file from make_bottom.py: its glacier, mouth and width become the defaults
    run        an output folder from fjord.jl to overlay (needs `bathymetry` = the file it used)
    level      "surface" or "outflow": which saved map slice to show
    margin     how far around the box the zoom extends [m]
    via        (lat, lon) points along a bent fjord's centre line, glacier to mouth (make_bottom.py --via)
    smooth     with via: corner rounding [m] (default: width)
    bedmachine BedMachine Greenland NetCDF (bed and ice/land mask for both maps and Lz)
    out        PNG path to save (None to skip); show=True also opens a window
    """
    k, star, side_wall, real_glacier = None, None, None, None
    if bathymetry:
        with netcdf_file(bathymetry, "r", mmap=False) as b:
            glacier = glacier or tuple(b._attributes.get("start_latlon", b.glacier_latlon))   # the box's y = 0 end
            if "start_latlon" in b._attributes:      # glacier not at the y = 0 end (side wall): star on it
                real_glacier = tuple(b.glacier_latlon)
                star = to_map(*latlon_to_xy(*real_glacier))
                w = b._attributes.get("glacier_wall", b"y0")
                side_wall = w.decode() if isinstance(w, bytes) else str(w)
            mouth = mouth or tuple(b.mouth_latlon)
            width = width or float(b.Lx)
            length = length or float(b.Ly)
            k = float(b._attributes.get("projection_scale", 1.0))   # older files: projected metres
            vl = np.ravel(b._attributes.get("via_latlon", [])).tolist()
            via = via or [tuple(vl[i:i + 2]) for i in range(0, len(vl), 2)]
            smooth = smooth or b._attributes.get("smooth")
            glacier_x = glacier_x or b._attributes.get("x_center")
    elif run:
        raise SystemExit("to overlay a run, also pass bathymetry= (the file that run used)")
    fjord = read_fjord(csv, row)
    name = fjord["Glacier Name"] if not bathymetry else os.path.basename(bathymetry)
    glacier = glacier or (float(fjord["Latitude"]), float(fjord["Longitude"]))
    mouth = mouth or DEFAULT_MOUTH
    width = width or 10_000.0
    box = Box(glacier, mouth, width, length, k, via, smooth, glacier_x)

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    bm = BedMachine(bedmachine)
    bx, by = box.outline()

    # zoom: BedMachine resampled on a regular map grid around the box, ~100 m
    pad = margin / 1e3
    x0, x1, y0, y1 = bx.min() - pad, bx.max() + pad, by.min() - pad, by.max() + pad
    nzx, nzy = int((x1 - x0) / .1), int((y1 - y0) / .1)
    ZX, ZY = np.meshgrid(np.linspace(x0, x1, nzx), np.linspace(y0, y1, nzy))
    zxy = from_map(ZX.ravel(), ZY.ravel())
    zoom = bm.bed(*zxy).reshape(ZX.shape)
    zmask = bm.sample(*zxy).reshape(ZX.shape)
    zice = np.isin(zmask, bm.ice)
    inside = (zoom < 0) & (zmask == 0)                     # open ocean (not land, not under ice)
    # deepest point inside the box -> Lz
    xs, ys = np.meshgrid(np.linspace(0, box.Lx, 100), np.linspace(0, box.Ly, 400))
    bxy = from_map(*box.model_to_map(xs.ravel(), ys.ravel()))
    boxz = np.where(bm.sample(*bxy) == 0, bm.bed(*bxy), 0.0)
    Lz = math.ceil(-min(boxz.min(), 0) / 10) * 10

    has_run = run is not None
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_axes([0.53, 0.06, 0.40, 0.9])
    zax = fig.add_axes([0.07, 0.53 if has_run else 0.25, 0.38, 0.42 if has_run else 0.55])

    # overview
    GX, GY, GZ, GM = _overview(bm)
    ax.set_facecolor("#1d2f8f")
    pc = ax.pcolormesh(GX, GY, np.where(GM == 0, GZ, np.nan), cmap=DEPTH_CMAP, vmin=-800, vmax=0,
                       shading="nearest", rasterized=True)
    ax.pcolormesh(GX, GY, np.where(np.isin(GM, (1, 4)), 1.0, np.nan), cmap="Greys", vmin=0, vmax=8,
                  shading="nearest", rasterized=True)          # ice-free land: light grey
    ax.pcolormesh(GX, GY, np.where(np.isin(GM, bm.ice), 1.0, np.nan), cmap="Greys", vmin=0, vmax=2,
                  shading="nearest", rasterized=True)          # ice: mid grey
    cx, cy = bx.mean(), by.mean()
    ax.add_patch(plt.Rectangle((cx - 25, cy - 25), 50, 50, fill=False, ec="r", lw=1.5))
    ax.set_aspect("equal"); ax.set_xlim(-750, 950); ax.set_ylim(-3450, -650)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    fig.colorbar(pc, ax=ax, fraction=.03, pad=.01).set_label("z (m)")

    # zoom with the box
    zpc = zax.pcolormesh(ZX, ZY, np.where(inside, zoom, np.nan), cmap=DEPTH_CMAP, vmin=-1000, vmax=0,
                         shading="nearest", rasterized=True)
    zax.pcolormesh(ZX, ZY, np.where(zice, 1.0, np.nan), cmap="Greys", vmin=0, vmax=20,
                   shading="nearest", rasterized=True)          # BedMachine ice: near white
    zax.set_facecolor("0.8")
    zax.plot(bx, by, "r-", lw=1.8)
    star = star or box.model_to_map(box.x0, 0)
    zax.plot(*star, "r*", ms=14)
    # label each end on the inner side of the box, so the text stays inside the panel
    for yy, label in ((0, "box start (y = 0)" if side_wall else "glacier (y = 0)"), (box.Ly, "mouth (y = Ly)")):
        px, py = box.model_to_map(box.Lx / 2, yy)
        inward = 1 if box.model_to_map(box.Lx / 2, box.Ly / 2)[0] > px else -1
        zax.annotate(label, (px, py), xytext=(8 * inward, 10), textcoords="offset points",
                     ha="left" if inward > 0 else "right", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.8))
    zax.set_aspect("equal"); zax.set_xlabel("x (km)"); zax.set_ylabel("y (km)")
    zax.set_title(f"{name}: box Lx = {box.Lx / 1e3:.1f} km x Ly = {box.Ly / 1e3:.1f} km, deepest {Lz} m")
    fig.colorbar(zpc, ax=zax, fraction=.03, pad=.01).set_label("z (m)")
    for yy in (cy + 25, cy - 25):
        fig.add_artist(ConnectionPatch(xyA=(1, .5), coordsA=zax.transAxes, xyB=(cx - 25, yy),
                                       coordsB=ax.transData, color="r", ls="--", lw=.8))

    # optional run overlay
    if has_run:
        FX, FY, zf, t, fname = _run_vorticity(run, level, box)
        vax = fig.add_axes([0.07, 0.06, 0.38, 0.4])
        vpc = vax.pcolormesh(FX, FY, zf, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
        vax.set_facecolor("0.8"); vax.plot(bx, by, "k-", lw=.6)
        vax.plot(*star, "k*", ms=12)
        vax.set_xlim(zax.get_xlim()); vax.set_ylim(zax.get_ylim()); vax.set_aspect("equal")
        vax.set_xlabel("x (km)"); vax.set_ylabel("y (km)")
        vax.set_title(f"{os.path.basename(os.path.normpath(run))}/{fname}: $\\zeta/f$ at t = {t / 3600:.1f} h")
        fig.colorbar(vpc, ax=vax, fraction=.03, pad=.01).set_label(r"$\zeta/f$")

    if out:
        fig.savefig(out, dpi=90)
        print(f"saved {out}")
    if show:
        plt.show()
    plt.close(fig)

    g, m = box.glacier, box.mouth
    info = dict(Lx=box.Lx, Ly=round(box.Ly, 1), Lz=Lz, glacier=g, mouth=m,
                bathymetry_cmd=(f"python3 make_bottom.py --row {row} " + (
                                    f"--glacier {real_glacier[0]} {real_glacier[1]} --start {g[0]} {g[1]} "
                                    f"--glacier-wall {side_wall} " if side_wall else f"--glacier {g[0]} {g[1]} ")
                                + f"--mouth {m[0]} {m[1]} "
                                f"--width {box.Lx:.0f}" + (f" --length {box.Ly:.0f}" if length else "")
                                + (" --via " + " ".join(f"{a} {o}" for a, o in box.via) if box.via else "")
                                + (f" --smooth {float(smooth):.0f}" if box.via and smooth else "")
                                + (f" --glacier-x {box.x0:.0f}" if box.x0 != box.Lx / 2 else "")
                                + (f" --bedmachine {bedmachine}" if bedmachine != DEFAULT_BEDMACHINE else "")
                                + " --out bottom_new.nc"),
                fjord_args=f"--Lx={box.Lx:.0f} --Ly={box.Ly:.0f} --Lz={Lz} --bathymetry_file=bottom_new.nc")
    print(f"box: Lx = {info['Lx']:.0f} m, Ly = {info['Ly']:.0f} m, Lz = {Lz} m (deepest BedMachine ocean point in the box; "
          f"make_bottom.py prints the bottom file's own)")
    print("build it:  " + info["bathymetry_cmd"])
    print("run with:  " + info["fjord_args"])
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=float, help="across-fjord width [m] (default 10000)")
    p.add_argument("--length", type=float, help="along-fjord length [m] (default glacier-to-mouth distance)")
    p.add_argument("--mouth", nargs=2, type=float, metavar=("LAT", "LON"))
    p.add_argument("--glacier", nargs=2, type=float, metavar=("LAT", "LON"))
    p.add_argument("--via", nargs="+", type=float, default=[], metavar="LAT LON",
                   help="points along a bent fjord's centre line, glacier to mouth (as make_bottom.py --via)")
    p.add_argument("--smooth", type=float, help="with --via: corner rounding [m] (default --width)")
    p.add_argument("--glacier-x", type=float, help="across position of the glacier->mouth line [m] (default width/2)")
    p.add_argument("--row", type=int, default=1, help="row of the Slater et al. table (default 1, Jakobshavn)")
    p.add_argument("--bathymetry", help="file from make_bottom.py to take the box from")
    p.add_argument("--run", help="fjord.jl output folder to overlay (needs --bathymetry)")
    p.add_argument("--level", default="surface", choices=["surface", "outflow"])
    p.add_argument("--margin", type=float, default=8000, help="zoom margin around the box [m]")
    p.add_argument("--bedmachine", nargs="?", const=DEFAULT_BEDMACHINE, default=DEFAULT_BEDMACHINE, metavar="PATH",
                   help=f"BedMachine Greenland NetCDF (default {DEFAULT_BEDMACHINE})")
    p.add_argument("--out", default="domain.png")
    p.add_argument("--show", action="store_true", help="also open an interactive window")
    a = p.parse_args()
    plot_domain(width=a.width, length=a.length, mouth=a.mouth, glacier=a.glacier, row=a.row,
                bathymetry=a.bathymetry, run=a.run, level=a.level, margin=a.margin, out=a.out, show=a.show,
                bedmachine=a.bedmachine,
                via=[tuple(a.via[i:i + 2]) for i in range(0, len(a.via), 2)], smooth=a.smooth,
                glacier_x=a.glacier_x)
