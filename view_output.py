"""
view_output.py - look at fjord.jl output: quick plots, an interactive browser, and movies

fjord.jl writes, per scenario (e.g. "control"), into its output folder:
  <scenario>_section.nc   along-fjord vertical slice through the mooring   (y-z, over time)
  <scenario>_surface.nc   map at the surface                              (x-y, over time)
  <scenario>_outflow.nc   map at the plume outflow depth                  (x-y, over time)
  <scenario>_mooring.nc   one water column near the glacier               (z, over time)
  <scenario>_average.nc   time-averaged 3-D fields                        (x-y-z)

Every file stores fields as (time, z, y, x) on a staggered grid (u, v, w sit on cell faces);
this module moves everything to cell centres, blanks out the sea floor, and adds two derived
fields: "speed" = sqrt(u^2 + v^2) and "zeta/f" = (dv/dx - du/dy) / f (maps only).

From Python:
    import view_output as vo
    vo.info("output_ilulissat_smoke")
    vo.plot("output_ilulissat_smoke", "section", "v")              # last snapshot
    vo.plot("output_ilulissat_smoke", "outflow", "zeta/f", t=0)    # first snapshot
    vo.plot("output_ilulissat_smoke", "average", "S", depth=-50)   # 3-D average: map at 50 m
    vo.plot("output_ilulissat_smoke", "average", "v", x=5000)      # 3-D average: section at x = 5 km
    vo.browse("output_ilulissat_smoke")                            # interactive window
    vo.animate("output_ilulissat_smoke", "section", "S", out="section_S.gif")

From the shell:
    python3 view_output.py output_ilulissat_smoke                  # info + interactive browser
    python3 view_output.py output_ilulissat_smoke --kind section --var v --save v.png
    python3 view_output.py output_ilulissat_smoke --kind outflow --var zeta/f --animate outflow.gif
"""

import argparse
import glob
import os

import numpy as np
import netCDF4
import matplotlib

CORIOLIS_F = 1.22e-4                         # fjord.jl's default coriolis_f [1/s]
KINDS = ["section", "surface", "outflow", "mooring", "average"]
VARS = ["S", "T", "u", "v", "w", "speed", "zeta/f"]
LABELS = {"S": "salinity (g/kg)", "T": "temperature (°C)", "u": "u, across-fjord (m/s)",
          "v": "v, along-fjord (m/s), + toward mouth", "w": "w, vertical (m/s)",
          "speed": "horizontal speed (m/s)", "zeta/f": r"$\zeta/f$"}
DIVERGING = {"u", "v", "w", "zeta/f"}
CMAPS = {"S": "viridis", "T": "inferno", "speed": "magma"}


# -------------------------------------------------------------------------------------
# reading
# -------------------------------------------------------------------------------------
def _file(run, kind):
    files = sorted(glob.glob(os.path.join(run, f"*_{kind}.nc")))
    if not files:
        raise FileNotFoundError(f"no *_{kind}.nc in {run}")
    return files[0]


def _to_centres(a, dims, ds):
    """Average face-located values onto cell centres along any staggered axis (a is (z, y, x))."""
    for axis, (face, centre) in enumerate([("z_aaf", "z_aac"), ("y_afa", "y_aca"), ("x_faa", "x_caa")]):
        if dims[axis] == face and a.shape[axis] == len(ds[centre]) + 1:
            lo = [slice(None)] * 3; hi = [slice(None)] * 3
            lo[axis] = slice(None, -1); hi[axis] = slice(1, None)
            a = 0.5 * (a[tuple(lo)] + a[tuple(hi)])
    return a


def _raw(ds, var, t):
    """One snapshot of a stored field at cell centres, shape (z, y, x)."""
    v = ds[var]
    a = np.ma.filled(v[t].astype(float), np.nan)
    return _to_centres(a, v.dimensions[1:], ds)


def _field(ds, var, t):
    """A stored or derived field at time index t, shape (z, y, x), sea floor -> NaN."""
    if var == "speed":
        a = np.hypot(_raw(ds, "u", t), _raw(ds, "v", t))
    elif var == "zeta/f":
        u, v = _raw(ds, "u", t), _raw(ds, "v", t)
        x, y = ds["x_caa"][:], ds["y_aca"][:]
        if len(x) < 2 or len(y) < 2:
            raise ValueError("zeta/f needs a map (surface, outflow or average), not a section or mooring")
        a = (np.gradient(v, x, axis=2) - np.gradient(u, y, axis=1)) / CORIOLIS_F
    else:
        a = _raw(ds, var, t)
    z = ds["z_aac"][:]
    bottom = np.ma.filled(ds["bottom_height"][:].astype(float), np.nan)   # (y, x)
    solid = z[:, None, None] < bottom[None, :, :]
    return np.where(solid, np.nan, a)


def info(run):
    """Print what is in a run folder: files, times, grid size, variables."""
    for path in sorted(glob.glob(os.path.join(run, "*.nc"))):
        with netCDF4.Dataset(path) as ds:
            t = ds["time"][:]
            shape = f'{len(ds["x_caa"])} x {len(ds["y_aca"])} x {len(ds["z_aac"])}'
            fields = [k for k in ("S", "T", "u", "v", "w") if k in ds.variables]
            print(f"{os.path.basename(path):24s} {len(t):4d} times, {t[0] / 3600:6.2f}-{t[-1] / 3600:6.2f} h, "
                  f"x*y*z = {shape:14s} {', '.join(fields)}")
    other = [f for f in os.listdir(run) if not f.endswith(".nc")]
    if other:
        print("also:", ", ".join(sorted(other)))


# -------------------------------------------------------------------------------------
# slicing a field into something 2-D to draw
# -------------------------------------------------------------------------------------
def _nearest(values, target):
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def _slice(ds, kind, var, t, depth=None, x=None):
    """Return (X, Y, C, xlabel, ylabel, bottom_line, title_bit) for one panel."""
    xc, yc, zc = ds["x_caa"][:], ds["y_aca"][:], ds["z_aac"][:]
    bottom = np.ma.filled(ds["bottom_height"][:].astype(float), np.nan)
    if kind == "mooring":                                       # time-depth (Hovmoller)
        times = ds["time"][:]
        C = np.array([_field(ds, var, k)[:, 0, 0] for k in range(len(times))]).T
        return times / 3600, zc, C, "time (h)", "z (m)", None, f"mooring at x = {xc[0]:.0f} m, y = {yc[0]:.0f} m"
    a = _field(ds, var, t)
    if kind == "section" or (kind == "average" and x is not None):
        i = 0 if len(xc) == 1 else _nearest(xc, x)
        return yc / 1e3, zc, a[:, :, i], "y along fjord (km)", "z (m)", bottom[:, i], f"x = {xc[i]:.0f} m"
    k = 0 if len(zc) == 1 else _nearest(zc, -15.0 if depth is None else depth)
    # maps are drawn with the fjord lying horizontally, x across upward (see _orient)
    return yc / 1e3, xc / 1e3, a[k].T, "y along fjord (km)", "x across (km)", None, f"z = {zc[k]:.0f} m"


def _orient(ax, xlabel):
    """Run the along-fjord axis right to left: mouth on the left, glacier (y = 0) on the right,
    like a map of a fjord that opens to the west (x, to the right looking down-fjord, is then
    north and points up). Same layout as plot_bathymetry.py."""
    if xlabel.startswith("y along") and not ax.xaxis_inverted():
        ax.invert_xaxis()


def _limits(var, data):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0, 1
    if var in DIVERGING:
        # the full range: features like the plume jet fill only a few cells, so percentiles would clip them
        m = np.abs(finite).max() or 1e-6
        return -m, m
    lo, hi = np.percentile(finite, [1, 99])
    return (lo, hi) if hi > lo else (lo - 1e-3, hi + 1e-3)


def _draw(ax, X, Y, C, var, vmin, vmax, bottom_line=None):
    pc = ax.pcolormesh(X, Y, C, cmap="RdBu_r" if var in DIVERGING else CMAPS.get(var, "viridis"),
                       vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_facecolor("0.6")                                     # sea floor / land
    if bottom_line is not None:
        ax.fill_between(X, bottom_line, np.nanmin(Y) - 1e4, color="0.6", zorder=2)
        ax.set_ylim(np.nanmin(Y) - 0.5 * abs(Y[1] - Y[0]), 0)
    return pc


# -------------------------------------------------------------------------------------
# public functions
# -------------------------------------------------------------------------------------
def plot(run, kind="section", var="v", t=-1, depth=None, x=None, zmin=None, vmin=None, vmax=None,
         save=None, show=True, ax=None):
    """
    One panel of one output file.

    kind   "section", "surface", "outflow", "mooring" or "average"
    var    "S", "T", "u", "v", "w", "speed" or "zeta/f"
    t      time index (-1 = last snapshot; ignored for "mooring", which shows all times)
    depth  for "average": draw a map at this depth [m, negative] (default -15)
    x      for "average": draw an along-fjord section at this x [m] instead
    zmin   zoom sections/moorings to the top |zmin| metres (e.g. -150)
    """
    if not show and ax is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with netCDF4.Dataset(_file(run, kind)) as ds:
        X, Y, C, xl, yl, bline, where = _slice(ds, kind, var, t, depth, x)
        time = ds["time"][t] if kind != "mooring" else None
    lo, hi = _limits(var, C)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(12, 4.5))
    pc = _draw(ax, X, Y, C, var, lo if vmin is None else vmin, hi if vmax is None else vmax, bline)
    if zmin is not None and yl == "z (m)":
        ax.set_ylim(zmin, 0)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    _orient(ax, xl)
    when = "" if time is None else f", t = {time / 3600:.2f} h"
    ax.set_title(f"{os.path.basename(os.path.normpath(run))}: {kind} {var} ({where}{when})")
    ax.figure.colorbar(pc, ax=ax, pad=.01).set_label(LABELS.get(var, var))
    if own:
        ax.figure.tight_layout()
        if save:
            ax.figure.savefig(save, dpi=110)
            print(f"saved {save}")
        if show:
            plt.show()
        plt.close(ax.figure)
    return ax


def browse(run, depth=None, zmin=None):
    """Interactive window: pick the file and variable with buttons, step through time with a slider."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider

    kinds = [k for k in KINDS if glob.glob(os.path.join(run, f"*_{k}.nc"))]
    state = {"kind": "section" if "section" in kinds else kinds[0], "var": "v", "t": 0}
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_axes([0.20, 0.22, 0.62, 0.68])
    cax = fig.add_axes([0.84, 0.22, 0.015, 0.68])
    kind_btn = RadioButtons(fig.add_axes([0.01, 0.55, 0.13, 0.35]), kinds, active=kinds.index(state["kind"]))
    var_btn = RadioButtons(fig.add_axes([0.01, 0.08, 0.13, 0.42]), VARS, active=VARS.index(state["var"]))
    slider = Slider(fig.add_axes([0.20, 0.07, 0.62, 0.04]), "time", 0, 1, valinit=0, valstep=1)

    def ntimes(kind):
        with netCDF4.Dataset(_file(run, kind)) as ds:
            return len(ds["time"])

    def redraw(*_):
        kind, var = state["kind"], state["var"]
        nt = ntimes(kind)
        t = min(int(slider.val), nt - 1)
        ax.clear(); cax.clear()
        try:
            with netCDF4.Dataset(_file(run, kind)) as ds:
                X, Y, C, xl, yl, bline, where = _slice(ds, kind, var, t, depth)
                time = ds["time"][t]
            lo, hi = _limits(var, C)
            pc = _draw(ax, X, Y, C, var, lo, hi, bline)
            if zmin is not None and yl == "z (m)":
                ax.set_ylim(zmin, 0)
            fig.colorbar(pc, cax=cax).set_label(LABELS.get(var, var))
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            _orient(ax, xl)
            when = "all times" if kind == "mooring" else f"t = {time / 3600:.2f} h  ({t + 1}/{nt})"
            ax.set_title(f"{kind}: {var}   {where}   {when}")
        except ValueError as e:
            ax.text(.5, .5, str(e), ha="center", va="center", transform=ax.transAxes, wrap=True)
        fig.canvas.draw_idle()

    def set_kind(label):
        state["kind"] = label
        nt = ntimes(label)
        slider.valmax = max(nt - 1, 1)
        slider.ax.set_xlim(0, slider.valmax)
        slider.set_val(min(int(slider.val), nt - 1))
        redraw()

    def set_var(label):
        state["var"] = label
        redraw()

    kind_btn.on_clicked(set_kind)
    var_btn.on_clicked(set_var)
    slider.on_changed(redraw)
    set_kind(state["kind"])
    fig._widgets = (kind_btn, var_btn, slider)       # keep references alive
    plt.show()


def animate(run, kind="section", var="v", out=None, fps=4, depth=None, x=None, zmin=None):
    """
    Movie of one slice through all its snapshots. Saves .mp4 if ffmpeg is installed, otherwise .gif.
    Colour limits are fixed over the whole movie so frames are comparable.
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    if kind == "mooring":
        raise ValueError("a mooring already shows all times; use plot() instead")
    with netCDF4.Dataset(_file(run, kind)) as ds:
        nt = len(ds["time"])
        frames = [_slice(ds, kind, var, k, depth, x) for k in range(nt)]
        times = ds["time"][:]
    lo, hi = _limits(var, np.concatenate([f[2].ravel() for f in frames]))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    X, Y, C, xl, yl, bline, where = frames[0]
    pc = _draw(ax, X, Y, C, var, lo, hi, bline)
    if zmin is not None and yl == "z (m)":
        ax.set_ylim(zmin, 0)
    fig.colorbar(pc, ax=ax, pad=.01).set_label(LABELS.get(var, var))
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    _orient(ax, xl)
    title = ax.set_title("")
    fig.tight_layout()

    def update(k):
        pc.set_array(frames[k][2].ravel())
        title.set_text(f"{kind}: {var}   {where}   t = {times[k] / 3600:.2f} h")
        return pc, title

    use_mp4 = animation.writers.is_available("ffmpeg")
    out = out or f"{kind}_{var.replace('/', '')}.{'mp4' if use_mp4 else 'gif'}"
    if out.endswith(".mp4") and not use_mp4:
        out = out[:-4] + ".gif"
        print("ffmpeg not found (brew install ffmpeg for mp4); writing a GIF instead")
    writer = animation.FFMpegWriter(fps=fps) if out.endswith(".mp4") else animation.PillowWriter(fps=fps)
    animation.FuncAnimation(fig, update, frames=nt, blit=False).save(out, writer=writer, dpi=100)
    plt.close(fig)
    print(f"saved {out} ({nt} frames)")
    if nt < 5:
        print(f"note: only {nt} snapshots; for a smoother movie rerun fjord.jl with e.g. --section_interval=600")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="fjord.jl output folder")
    p.add_argument("--kind", choices=KINDS, help="which file (no --kind: open the interactive browser)")
    p.add_argument("--var", default="v", choices=VARS)
    p.add_argument("--t", type=int, default=-1, help="time index (-1 = last)")
    p.add_argument("--depth", type=float, help="average file: map at this depth [m, negative]")
    p.add_argument("--x", type=float, help="average file: along-fjord section at this x [m]")
    p.add_argument("--zmin", type=float, help="zoom depth axis to the top |zmin| m, e.g. -150")
    p.add_argument("--save", help="save the plot to this file instead of opening a window")
    p.add_argument("--animate", metavar="OUT", help="write a movie (.gif, or .mp4 with ffmpeg)")
    p.add_argument("--fps", type=int, default=4)
    a = p.parse_args()

    info(a.run)
    if a.animate:
        animate(a.run, a.kind or "section", a.var, out=a.animate, fps=a.fps, depth=a.depth, x=a.x, zmin=a.zmin)
    elif a.kind:
        plot(a.run, a.kind, a.var, t=a.t, depth=a.depth, x=a.x, zmin=a.zmin, save=a.save, show=not a.save)
    else:
        browse(a.run, depth=a.depth, zmin=a.zmin)
