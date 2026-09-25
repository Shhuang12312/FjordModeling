"""
ibcao_to_bottom.py - cut a fjord-aligned strip out of IBCAO v5 and write bottom.nc for fjord.jl

fjord.jl ignores the coordinates in the bathymetry file: it stretches the 2-D array over
0..Lx (across) by 0..Ly (along, glacier at y = 0, mouth at y = Ly). So the file has to be
rotated and cropped to the fjord before the model sees it. This script does that:

  1. project the glacier-front and mouth lat/lon into IBCAO polar stereographic (EPSG:3996),
  2. lay a rectangle of width `--width` centred on the glacier->mouth line,
  3. bilinearly sample the 100 m IBCAO tiles on a regular grid inside that rectangle,
  4. set land to 0 (no water) and write x, y, bottom to a NetCDF file.

Example:
  python3 ibcao_to_bottom.py --glacier 68.90 -50.30 --mouth 68.95 -50.60 --width 5000

It prints the Lx, Ly, Lz to put in FjordConfig. Needs numpy, scipy and tifffile.
"""

import argparse
import glob
import math
import os

import numpy as np
import tifffile
from scipy.io import netcdf_file
from scipy.ndimage import map_coordinates

# IBCAO v5 100 m mosaic: 4 x 4 tiles of 14550 x 14550 pixels covering +/-2910 km.
RES = 100.0
TILE = 14550
EDGE = 2_910_000.0

# WGS 84 / IBCAO Polar Stereographic (EPSG:3996): north pole, true scale at 75 N, lon_0 = 0.
A = 6_378_137.0
F = 1 / 298.257223563
E = math.sqrt(F * (2 - F))
LAT_TS = math.radians(75.0)


def _t(phi):
    return math.tan(math.pi / 4 - phi / 2) / ((1 - E * math.sin(phi)) / (1 + E * math.sin(phi))) ** (E / 2)


def latlon_to_xy(lat, lon):
    """Forward polar stereographic (Snyder 1987, eqs. 15-9, 21-33, 21-34)."""
    phi, lam = math.radians(lat), math.radians(lon)
    m_c = math.cos(LAT_TS) / math.sqrt(1 - (E * math.sin(LAT_TS)) ** 2)
    rho = A * m_c * _t(phi) / _t(LAT_TS)
    return rho * math.sin(lam), -rho * math.cos(lam)


class Mosaic:
    """Read-only view of the 16 tiles as one big (row, col) array, row 0 at y = +EDGE."""

    def __init__(self, tile_dir):
        self.tiles = {}
        for path in glob.glob(os.path.join(tile_dir, "*_part_*_*.tiff")):
            i, j = map(int, os.path.basename(path)[:-5].split("_")[-2:])
            self.tiles[(i, j)] = path
        if len(self.tiles) != 16:
            raise SystemExit(f"expected 16 IBCAO tiles in {tile_dir}, found {len(self.tiles)}")

    def window(self, r0, r1, c0, c1):
        """Global pixel rows r0:r1, cols c0:c1, stitched across tile boundaries."""
        out = np.full((r1 - r0, c1 - c0), np.nan, dtype=np.float32)
        for (i, j), path in self.tiles.items():
            tr0, tc0 = (3 - j) * TILE, i * TILE      # tile (i, j): column block i, row block 3-j
            rr0, rr1 = max(r0, tr0), min(r1, tr0 + TILE)
            cc0, cc1 = max(c0, tc0), min(c1, tc0 + TILE)
            if rr0 < rr1 and cc0 < cc1:
                m = tifffile.memmap(path, mode="r")
                out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = m[rr0 - tr0:rr1 - tr0, cc0 - tc0:cc1 - tc0]
        return out

    def sample(self, X, Y):
        """Bilinear elevation at projected points X, Y [m]."""
        rows = (EDGE - Y) / RES - 0.5      # fractional pixel-centre indices
        cols = (X + EDGE) / RES - 0.5
        r0, c0 = int(np.floor(rows.min())) - 1, int(np.floor(cols.min())) - 1
        r1, c1 = int(np.ceil(rows.max())) + 2, int(np.ceil(cols.max())) + 2
        win = self.window(r0, r1, c0, c1)
        return map_coordinates(win, [rows - r0, cols - c0], order=1, mode="nearest")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glacier", nargs=2, type=float, required=True, metavar=("LAT", "LON"),
                   help="point at the glacier front, on the fjord centre line (lon negative = W)")
    p.add_argument("--mouth", nargs=2, type=float, required=True, metavar=("LAT", "LON"),
                   help="point at the fjord mouth, on the centre line")
    p.add_argument("--width", type=float, required=True, help="across-fjord width of the strip [m]")
    p.add_argument("--dx", type=float, default=50.0, help="output grid spacing [m] (default 50)")
    p.add_argument("--tiles", default=os.path.expanduser("~/Downloads/ibcao_v5_2_2026_depth_100m_16_tiled"))
    p.add_argument("--out", default="bottom_ibcao.nc")
    args = p.parse_args()

    gx, gy = latlon_to_xy(*args.glacier)
    mx, my = latlon_to_xy(*args.mouth)
    Ly = math.hypot(mx - gx, my - gy)
    Lx = args.width
    ux, uy = (mx - gx) / Ly, (my - gy) / Ly    # along-fjord unit vector (glacier -> mouth)
    nx_, ny_ = uy, -ux                         # across-fjord: to the right looking down-fjord

    nx, ny = max(2, round(Lx / args.dx)), max(2, round(Ly / args.dx))
    x = (np.arange(nx) + 0.5) * Lx / nx
    y = (np.arange(ny) + 0.5) * Ly / ny
    Xm, Ym = np.meshgrid(x - Lx / 2, y, indexing="ij")      # shape (nx, ny)
    X = gx + Xm * nx_ + Ym * ux
    Y = gy + Xm * ny_ + Ym * uy

    elev = Mosaic(args.tiles).sample(X, Y).astype(np.float64)
    bottom = np.minimum(elev, 0.0)             # land -> 0: no water in that column

    wet = bottom < 0
    print(f"strip: Lx = {Lx:.0f} m across, Ly = {Ly:.0f} m along, grid {nx} x {ny} at ~{args.dx:.0f} m")
    print(f"deepest {bottom.min():.1f} m, {100 * wet.mean():.0f}% of columns are water")
    print(f"glacier-end row: {100 * wet[:, 0].mean():.0f}% water, mouth row: {100 * wet[:, -1].mean():.0f}% water")

    # Written as bottom(y, x) so Julia's NCDatasets reads it as (x, y), like make_bathymetry.jl.
    with netcdf_file(args.out, "w") as ds:
        ds.createDimension("x", nx)
        ds.createDimension("y", ny)
        ds.createVariable("x", "f8", ("x",))[:] = x
        ds.createVariable("y", "f8", ("y",))[:] = y
        v = ds.createVariable("bottom", "f8", ("y", "x"))
        v[:] = bottom.T
        v.long_name = "bottom elevation (negative below sea level), from IBCAO v5.2"
        v.units = "m"
        ds.source = "IBCAO v5.2 100 m, fjord-aligned strip"
        ds.glacier_latlon = np.array(args.glacier)
        ds.mouth_latlon = np.array(args.mouth)
        ds.Lx, ds.Ly = Lx, Ly

    lz = math.ceil(-bottom.min() / 10) * 10
    print(f"\nwrote {args.out}. In fjord.jl's FjordConfig set:")
    print(f'  Lx = {Lx:.1f}, Ly = {Ly:.1f}, Lz = {lz:.1f},')
    print(f'  bathymetry_file = "{args.out}", bathymetry_var = "bottom"')


if __name__ == "__main__":
    main()
