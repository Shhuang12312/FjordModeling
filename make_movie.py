"""
make_movie.py - turn a fjord.jl output folder into a movie

By default each frame shows, at one snapshot time:
  1. along-fjord section of the along-fjord velocity v  (<scenario>_section.nc)
  2. along-fjord section of salinity S                   (<scenario>_section.nc)
  3. map of v at the plume outflow depth                 (<scenario>_outflow.nc)
or, with --sections, one along-fjord section panel per variable (e.g. T S v). The section
runs through x = mooring_x, so set --mooring_x to the plume when running fjord.jl.
Colour scales are fixed over the whole movie so frames are comparable. Writes .mp4 if ffmpeg
is installed (brew install ffmpeg), otherwise an animated .gif. The fjord is drawn mouth left,
glacier right (see view_output._orient).

How many frames you get is set when you RUN fjord.jl: one per --section_interval seconds of
model time (default 4320 s = 1.2 h). For a movie use e.g. --section_interval=300.

    python3 tools/make_movie.py output/ilulissat                      # -> output/ilulissat/movie.gif
    python3 tools/make_movie.py output/ilulissat --fields v T zeta/f  # choose the three panels
    python3 tools/make_movie.py output/ilulissat --zmin -100 --fps 6 --out fjord.mp4
    python3 tools/make_movie.py output/ilulissat --sections T S v           # three along-fjord sections

From Python:
    from make_movie import make_movie
    make_movie("output/ilulissat", fields=("v", "S", "v"))
"""

import argparse
import os

import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

import view_output as vo

# which file each panel comes from: the first two fields are drawn on the section, the third
# on the outflow-depth map (zeta/f only exists on maps, so it is only allowed in the third)
PANEL_KINDS = ("section", "section", "outflow")


def make_movie(run, fields=("v", "S", "v"), out=None, fps=4, zmin=None, dpi=100, kinds=PANEL_KINDS):
    """Write a movie of `run`, one panel per (kind, field); returns the path written."""
    data = []                                   # per panel: (frames, xlabel, ylabel, where, bottom)
    times = None
    for kind, var in zip(kinds, fields):
        if var == "zeta/f" and kind == "section":
            raise ValueError("zeta/f needs a map; put it as the third field")
        with netCDF4.Dataset(vo._file(run, kind)) as ds:
            t = ds["time"][:]
            times = t if times is None else times[:min(len(times), len(t))]
            frames = [vo._slice(ds, kind, var, k) for k in range(len(t))]
        data.append(frames)
    nt = len(times)
    if nt < 2:
        raise SystemExit(f"{run} has only {nt} snapshot(s); rerun fjord.jl with a smaller --section_interval")

    fig, axes = plt.subplots(len(data), 1, figsize=(12, 3.3 * len(data) + 0.4), constrained_layout=True,
                             squeeze=False)
    meshes = []
    for ax, kind, var, frames in zip(axes[:, 0], kinds, fields, data):
        lo, hi = vo._limits(var, np.concatenate([f[2].ravel() for f in frames[:nt]]))
        X, Y, C, xl, yl, bline, where = frames[0]
        pc = vo._draw(ax, X, Y, C, var, lo, hi, bline)
        if zmin is not None and yl == "z (m)":
            ax.set_ylim(zmin, 0)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        vo._orient(ax, xl)
        ax.set_title(f"{kind}: {vo.LABELS.get(var, var)}   ({where})", fontsize=10)
        fig.colorbar(pc, ax=ax, pad=.01)
        meshes.append(pc)
    suptitle = fig.suptitle("")

    def update(k):
        for pc, frames in zip(meshes, data):
            pc.set_array(frames[k][2].ravel())
        suptitle.set_text(f"{os.path.basename(os.path.normpath(run))}   t = {times[k] / 3600:5.2f} h"
                          f"   (frame {k + 1}/{nt})")
        return (*meshes, suptitle)

    use_mp4 = animation.writers.is_available("ffmpeg")
    out = out or os.path.join(run, "movie." + ("mp4" if use_mp4 else "gif"))
    if out.endswith(".mp4") and not use_mp4:
        out = out[:-4] + ".gif"
        print("ffmpeg not found (brew install ffmpeg for mp4); writing a GIF instead")
    writer = animation.FFMpegWriter(fps=fps) if out.endswith(".mp4") else animation.PillowWriter(fps=fps)
    animation.FuncAnimation(fig, update, frames=nt).save(out, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"saved {out}: {nt} frames, {times[0] / 3600:.2f}-{times[-1] / 3600:.2f} h of model time")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="fjord.jl output folder")
    p.add_argument("--fields", nargs=3, default=["v", "S", "v"], metavar=("SECTION1", "SECTION2", "MAP"),
                   help=f"variables for the three panels, from {vo.VARS}")
    p.add_argument("--sections", nargs="+", metavar="VAR",
                   help="instead of --fields: one along-fjord section panel per variable, e.g. T S v")
    p.add_argument("--out", help="output file (.gif or .mp4); default <run>/movie.gif")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--zmin", type=float, help="zoom the sections to the top |zmin| m, e.g. -100")
    a = p.parse_args()
    if a.sections:
        make_movie(a.run, fields=tuple(a.sections), out=a.out, fps=a.fps, zmin=a.zmin,
                   kinds=("section",) * len(a.sections))
    else:
        make_movie(a.run, fields=tuple(a.fields), out=a.out, fps=a.fps, zmin=a.zmin)
