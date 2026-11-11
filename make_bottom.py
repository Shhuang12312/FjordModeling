"""
make_bottom.py - cut a fjord-aligned strip of real bathymetry and write a bottom file for fjord.jl

fjord.jl ignores the coordinates in the bathymetry file: it stretches the 2-D array over
0..Lx (across) by 0..Ly (along, glacier at y = 0, mouth at y = Ly). So the file has to be
rotated and cropped to the fjord before the model sees it. This script does that:

  1. take the glacier position from a row of the Slater et al. (2022, GRL) fjord table
     (2021gl097081-sup-0003 ... si-s03.csv), or from --glacier,
  2. project the glacier and mouth lat/lon into polar stereographic (EPSG:3996: north pole, true
     at 75 N; only the projection maths, no data from it),
  3. lay a rectangle of width `--width` centred on the glacier->mouth line,
  4. bilinearly sample BedMachine Greenland's `bed` (150 m; includes the fjord multibeam surveys
     right up to the ice fronts) on a regular grid inside that rectangle,
  5. set land to 0 (no water) and, using BedMachine's ice mask, columns under grounded or
     floating ice too (`bed` is the bed under the ice, so without this the fjord would carry
     on as open water past the ice front); the mask is saved as `mask`,
  6. set water shallower than --min-depth (default 20 m, as in Zhao et al.) to land, and any
     water not connected through cell faces to the mouth end (lakes, cut-off bays),
  7. write x, y, bottom (and mask) to a NetCDF file.
All lengths (--width, --length, --dx, and the Lx, Ly, x, y written) are metres on the ground.
The projection is true only at 75 N, so at Greenland fjord latitudes projected metres are
~1-2% longer; the file records that factor as `projection_scale`. (Files made before this
have no `projection_scale`: their lengths are projected metres, ~1.6% too long at 69 N.)

A fjord that bends (e.g. Helheim's branch joining Sermilik Fjord at a right angle) cannot fit a
straight box with the glacier at one end: give --via points along its centre line and the box
follows the fjord and is straightened (y = distance along the centre line). Or keep a straight box
along the main fjord and put the glacier on a side wall: --start (the box's y = 0 end, on the centre
line) and --glacier-wall xL (or x0); make it --width so that wall runs through the glacier front
(the script finds the front and says how much to trim), and run fjord.jl with --plume_wall=xL
--plume_y_center=<printed y>. Or keep a wider box, with ice between the wall and the front, and
pass the printed --plume_wall_offset as well. --glacier-x moves the centre line across the box,
to fit side branches on one side. Example (Sermilik Fjord / Helheim, the third row of the table;
the 120 x 85 km box of Zhao et al.'s Sermilik run, north-south, Helheim 21 km in from the west wall):
  python3 make_bottom.py --row 3 --start 66.5261 -37.6109 --mouth 65.4588 -37.9429 --width 85000 \
      --length 120000 --glacier-wall xL --out bottom_sermilik_bedmachine.nc

Example (Jakobshavn Isbrae, the first row of the table):
  python3 make_bottom.py --row 1 --mouth 69.155 -51.12 --width 10000 --out bottom_ilulissat.nc

It prints the Lx, Ly, Lz to put in FjordConfig, and the fjord's values from the table
(grounding-line depth, plume neutral-buoyancy depth, summer discharge) for setting up the
plume. With the ice mask it also finds the ice front and prints where to put the plume and,
if the front is not at y = 0, the --glacier that moves the box's glacier end onto it.
Needs numpy, scipy and netCDF4, and BedMachineGreenland-v7.nc in ~/Downloads (or --bedmachine).
"""

import argparse
import csv
import math
import os

import numpy as np
from scipy.io import netcdf_file
from scipy.ndimage import label, map_coordinates

DOWNLOADS = os.path.expanduser("~/Downloads")
DEFAULT_CSV = os.path.join(DOWNLOADS, "2021gl097081-sup-0003-supporting information si-s03.csv")
DEFAULT_BEDMACHINE = os.path.join(DOWNLOADS, "BedMachineGreenland-v7.nc")

# WGS 84 / IBCAO Polar Stereographic (EPSG:3996): north pole, true scale at 75 N, lon_0 = 0.
# The box is laid out in this projection; the sea floor itself comes from BedMachine (EPSG:3413).
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


def _k(lat_ts):
    """rho / t(phi) of a polar stereographic projection true at lat_ts [rad]."""
    return A * math.cos(lat_ts) / math.sqrt(1 - (E * math.sin(lat_ts)) ** 2) / _t(lat_ts)


def scale_factor(lat):
    """Point scale factor of the EPSG:3996 projection at latitude `lat` [deg]: projected metres per
    metre on the ground (Snyder 1987, eq. 21-32). About 1.016 at 69 N, since the projection is
    true only at 75 N. Over a fjord-sized box it varies by ~0.1%, so one value is used per box."""
    phi = math.radians(lat)
    m = math.cos(phi) / math.sqrt(1 - (E * math.sin(phi)) ** 2)
    return _k(LAT_TS) * _t(phi) / (A * m)


def xy_to_latlon(x, y):
    """Inverse of latlon_to_xy (Snyder 1987, eqs. 21-39, 7-9, 20-16), iterated to convergence."""
    t = math.hypot(x, y) / _k(LAT_TS)
    phi = math.pi / 2 - 2 * math.atan(t)
    for _ in range(10):
        es = E * math.sin(phi)
        phi = math.pi / 2 - 2 * math.atan(t * ((1 - es) / (1 + es)) ** (E / 2))
    return math.degrees(phi), math.degrees(math.atan2(x, -y))


class FjordPath:
    """The centre line of the model box, from the glacier through any `via` points to the mouth,
    and the map between model coordinates (x across, y along, metres on the ground) and EPSG:3996
    projected metres. With no via points it is the straight glacier->mouth line. With via points
    the box follows the bent fjord and is straightened: y is distance along the (smoothed) centre
    line and x distance across it, to the right looking down-fjord. Smoothing over `smooth` metres
    (default: the box width) rounds the corners so the box's cross-sections do not overlap."""

    def __init__(self, points, width, length=None, k=None, smooth=None, x_center=None):
        self.points = [(float(a), float(b)) for a, b in points]
        self.Lx = float(width)
        self.x0 = self.Lx / 2 if x_center is None else float(x_center)   # model x of the centre line
        self.k = scale_factor(0.5 * (self.points[0][0] + self.points[-1][0])) if k is None else float(k)
        P = np.array([latlon_to_xy(*p) for p in self.points])
        seg = np.diff(P, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        self.bent = len(P) > 2
        if not self.bent:
            self.smooth = 0.0
            self.s, self.c = np.array([0.0, L[0]]), P
            self.t = np.repeat(seg / L[0], 2, axis=0)
        else:
            self.smooth = self.Lx if smooth is None else float(smooth)      # ground metres
            h = 25.0 * self.k                                                 # densify every 25 m
            s_nodes = np.concatenate([[0.0], np.cumsum(L)])
            sd = np.linspace(0, s_nodes[-1], int(s_nodes[-1] / h) + 2)
            h = sd[1] - sd[0]
            D = np.column_stack([np.interp(sd, s_nodes, P[:, 0]), np.interp(sd, s_nodes, P[:, 1])])
            m = max(1, int(round(0.5 * self.smooth * self.k / h)))
            t0, t1 = seg[0] / L[0], seg[-1] / L[-1]
            kern = np.ones(2 * m + 1) / (2 * m + 1)
            for _ in range(2):                  # two box filters = a triangle: smooth curvature
                # extend straight past each end so the ends stay put and keep their direction
                E_ = np.vstack([D[0] - t0 * h * np.arange(m, 0, -1)[:, None], D,
                                D[-1] + t1 * h * np.arange(1, m + 1)[:, None]])
                D = np.column_stack([np.convolve(E_[:, 0], kern, "valid"), np.convolve(E_[:, 1], kern, "valid")])
            d = np.diff(D, axis=0)
            self.s = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
            self.c = D
            g = np.gradient(D, axis=0)
            self.t = g / np.hypot(g[:, 0], g[:, 1])[:, None]
        self.Ly = float(length) if length else self.s[-1] / self.k

    def _centre(self, sp):
        """Centre point and unit tangent at projected arc length sp (straight past the ends)."""
        sp = np.asarray(sp, dtype=float)
        if not self.bent:          # exact straight line, as make_bottom.py always did
            return (self.c[0, 0] + sp * self.t[0, 0], self.c[0, 1] + sp * self.t[0, 1],
                    np.full_like(sp, self.t[0, 0]), np.full_like(sp, self.t[0, 1]))
        cx, cy = np.interp(sp, self.s, self.c[:, 0]), np.interp(sp, self.s, self.c[:, 1])
        tx, ty = np.interp(sp, self.s, self.t[:, 0]), np.interp(sp, self.s, self.t[:, 1])
        n = np.hypot(tx, ty)
        tx, ty = tx / n, ty / n
        for end, i in ((sp < 0, 0), (sp > self.s[-1], -1)):
            cx = np.where(end, self.c[i, 0] + (sp - self.s[i]) * self.t[i, 0], cx)
            cy = np.where(end, self.c[i, 1] + (sp - self.s[i]) * self.t[i, 1], cy)
        return cx, cy, tx, ty

    def to_xy(self, xm, ym):
        """Model (x, y) [ground m] -> EPSG:3996 projected (X, Y) [m]."""
        cx, cy, tx, ty = self._centre(self.k * np.asarray(ym, dtype=float))
        a = self.k * (np.asarray(xm, dtype=float) - self.x0)
        return cx + a * ty, cy - a * tx          # across = (ty, -tx): right of the along direction

    def from_xy(self, X, Y):
        """EPSG:3996 projected (X, Y) -> model (x, y) [ground m], via the nearest centre-line point."""
        X, Y = float(X), float(Y)
        if not self.bent:
            dx, dy = X - self.c[0, 0], Y - self.c[0, 1]
            tx, ty = self.t[0]
            sp, a = dx * tx + dy * ty, dx * ty - dy * tx
        else:
            j = int(np.argmin((self.c[:, 0] - X) ** 2 + (self.c[:, 1] - Y) ** 2))
            dx, dy = X - self.c[j, 0], Y - self.c[j, 1]
            tx, ty = self.t[j]
            sp, a = self.s[j] + dx * tx + dy * ty, dx * ty - dy * tx
        return a / self.k + self.x0, sp / self.k

    def fold(self):
        """Largest |curvature| x half-width: the box's cross-sections overlap if this reaches 1,
        and the inner side of a bend is squeezed (outer side stretched) by this fraction."""
        if not self.bent:
            return 0.0
        theta = np.unwrap(np.arctan2(self.t[:, 1], self.t[:, 0]))
        return float(np.abs(np.gradient(theta, self.s)).max() * self.k * max(self.x0, self.Lx - self.x0))

    @classmethod
    def from_file(cls, ds):
        """The path of a bottom file written by make_bottom.py (netCDF4 or scipy dataset)."""
        get = (lambda n, d=None: getattr(ds, n, d)) if not hasattr(ds, "_attributes") else ds._attributes.get
        via = np.ravel(get("via_latlon", [])).tolist()
        start = get("start_latlon", None)
        pts = [tuple(get("glacier_latlon") if start is None else start)] \
            + [tuple(via[i:i + 2]) for i in range(0, len(via), 2)] \
            + [tuple(get("mouth_latlon"))]
        smooth, xc = get("smooth", None), get("x_center", None)
        return cls(pts, float(get("Lx")), float(get("Ly")), k=float(get("projection_scale", 1.0)),
                   smooth=None if smooth is None else float(smooth), x_center=None if xc is None else float(xc))


class BedMachine:
    """BedMachine Greenland mask (EPSG:3413, 150 m): 0 ocean, 1 ice-free land, 2 grounded ice,
    3 floating ice, 4 non-Greenland land, and `bed`. Sampled at EPSG:3996 projected points."""

    def __init__(self, path):
        import netCDF4                             # BedMachine is NetCDF-4, which scipy cannot read
        self.ds = netCDF4.Dataset(path)
        mp = self.ds.variables.get("mapping")
        if mp is not None and (float(getattr(mp, "standard_parallel", 70)) != 70
                               or float(getattr(mp, "straight_vertical_longitude_from_pole", -45)) != -45):
            raise SystemExit(f"{path} is not on the EPSG:3413 grid this script expects")
        self.x = np.asarray(self.ds["x"][:], dtype=float)
        self.y = np.asarray(self.ds["y"][:], dtype=float)
        self.mask = self.ds["mask"]
        self.mask.set_auto_maskandscale(False)
        values, meanings = getattr(self.mask, "flag_values", None), getattr(self.mask, "flag_meanings", "")
        if values is not None and meanings:
            self.ice = tuple(int(v) for v, m in zip(np.ravel(values), meanings.split())
                             if "ice" in m and "free" not in m)
        else:
            self.ice = (2, 3)
        # EPSG:3996 (true at 75 N, lon_0 = 0) -> EPSG:3413 (true at 70 N, lon_0 = -45): turn 45 deg, rescale.
        self.c = math.sqrt(0.5) * _k(math.radians(70.0)) / _k(LAT_TS)

    def _window(self, var, X, Y):
        """Fractional (row, col) of EPSG:3996 projected points X, Y [m] on the BedMachine grid,
        relative to the window of `var` that covers them (only that window is read)."""
        bx, by = self.c * (X - Y), self.c * (X + Y)
        cols = (bx - self.x[0]) / (self.x[1] - self.x[0])
        rows = (by - self.y[0]) / (self.y[1] - self.y[0])
        r0, c0 = int(np.floor(rows.min())) - 1, int(np.floor(cols.min())) - 1
        r1, c1 = int(np.ceil(rows.max())) + 2, int(np.ceil(cols.max())) + 2
        if r0 < 0 or c0 < 0 or r1 > self.y.size or c1 > self.x.size:
            raise SystemExit("the box reaches outside the BedMachine grid")
        v = self.ds[var]
        v.set_auto_maskandscale(False)
        return rows - r0, cols - c0, np.asarray(v[r0:r1, c0:c1])

    def sample(self, X, Y):
        """Mask code at EPSG:3996 projected points X, Y [m] (nearest cell: it is a category)."""
        rows, cols, win = self._window("mask", X, Y)
        return win[np.rint(rows).astype(int), np.rint(cols).astype(int)]

    def bed(self, X, Y):
        """Bed elevation [m, negative below sea level] at EPSG:3996 projected points X, Y, bilinear.
        In the ocean this is the bathymetry; under ice it is the bed beneath the ice."""
        rows, cols, win = self._window("bed", X, Y)
        return map_coordinates(win.astype(np.float64), [rows, cols], order=1, mode="nearest")


def ice_front(bottom, mask, ice, x, y):
    """Where the glacier wall and plume go: (x, y, depth). Scanning each column up-fjord from
    y = 0, the front is the first water cell with ice behind it (or water at y = 0 itself).
    The wall goes at the most up-fjord front cell, so no fjord water is cut off, and the
    plume at the deepest water on that row, so fjord.jl finds water under it."""
    wet = bottom < 0
    first = np.argmax(wet, axis=1)                     # first wet cell per column
    cols = np.arange(bottom.shape[0])
    behind = np.isin(mask[cols, np.maximum(first - 1, 0)], ice)
    front = wet.any(axis=1) & ((first == 0) | behind)
    if not front.any():
        return None
    j = first[front].min()
    i = int(np.argmin(np.where(front & (first == j), bottom[:, j], 0.0)))
    return x[i], j * (y[1] - y[0]), -bottom[i, j]      # y of the cell's glacier-side edge


def clean_water(bottom, min_depth):
    """In place: water shallower than min_depth becomes land, then water not connected through
    cell faces to the mouth row (bottom[:, -1]) does too. Returns the two cell counts."""
    shallow = (bottom < 0) & (bottom > -min_depth)
    bottom[shallow] = 0.0
    lab, _ = label(bottom < 0)                 # 4-connected, like the model's cell faces
    keep = np.unique(lab[:, -1])
    cut = (lab > 0) & ~np.isin(lab, keep[keep > 0])
    bottom[cut] = 0.0
    return int(shallow.sum()), int(cut.sum())


def side_front(bottom, mask, ice, x, y, wall, y_glacier, window=5000.0):
    """Like ice_front, for a glacier on a side wall (wall "xL": x = Lx, "x0": x = 0), searching
    only within `window` of y_glacier (so other fjords or the shelf touching that wall are not
    taken for it). Returns (y, distance of the front from the wall, depth, x of the front)."""
    B, M = (bottom[::-1, :].T, mask[::-1, :].T) if wall == "xL" else (bottom.T, mask.T)   # (along y, out)
    dist = (np.arange(B.shape[1]) + 0.5) * (x[1] - x[0])
    near = np.abs(y - y_glacier) <= window
    f = ice_front(np.where(near[:, None], B, 0.0), M, ice, y, dist)
    if f is None:
        return None
    fy, fd, fz = f
    fd = round(fd / (x[1] - x[0])) * (x[1] - x[0])        # ice_front returns the wall-side edge
    return fy, fd, fz, (x[-1] + x[0] - fd) if wall == "xL" else fd


def read_fjord(csv_path, row):
    """Row `row` (1 = first data row) of the Slater et al. (2022) fjord table, as a dict."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not 1 <= row <= len(rows):
        raise SystemExit(f"--row must be 1..{len(rows)}")
    return rows[row - 1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--row", type=int, default=1, help="row of the fjord table to use (default 1, Jakobshavn)")
    p.add_argument("--csv", default=DEFAULT_CSV, help="Slater et al. (2022) fjord table")
    p.add_argument("--glacier", nargs=2, type=float, metavar=("LAT", "LON"),
                   help="override the table's glacier position (lon negative = W)")
    p.add_argument("--mouth", nargs=2, type=float, required=True, metavar=("LAT", "LON"),
                   help="point at the fjord mouth, on the centre line (the table does not give it)")
    p.add_argument("--width", type=float, required=True, help="across-fjord width of the strip [m]")
    p.add_argument("--length", type=float,
                   help="along-fjord length of the strip [m], pointing at --mouth (default: reach the mouth)")
    p.add_argument("--via", nargs="+", type=float, default=[], metavar="LAT LON",
                   help="points on the fjord's centre line between the glacier and the mouth (lat lon pairs, "
                        "in order from the glacier): the box follows the bent fjord and is straightened")
    p.add_argument("--smooth", type=float,
                   help="with --via: round the centre line's corners over this many metres (default --width)")
    p.add_argument("--start", nargs=2, type=float, metavar=("LAT", "LON"),
                   help="where the box's y = 0 end crosses its centre line (default: the glacier). Use it "
                        "with --glacier-wall for a glacier on a side wall")
    p.add_argument("--glacier-wall", choices=["y0", "x0", "xL"], default="y0",
                   help="which wall the glacier is on: y0 (the glacier end, default), or a side wall x0 / xL "
                        "for a glacier at the end of a branch that meets the fjord at a right angle (Helheim)")
    p.add_argument("--glacier-x", type=float,
                   help="where across the box the glacier->mouth line runs [m from x = 0] (default --width / 2); "
                        "move it off-centre to fit side branches on one side without widening the box")
    p.add_argument("--dx", type=float, default=50.0, help="output grid spacing [m] (default 50)")
    p.add_argument("--bedmachine", default=DEFAULT_BEDMACHINE, metavar="PATH",
                   help=f"BedMachine Greenland NetCDF: sea floor and ice mask (default {DEFAULT_BEDMACHINE})")
    p.add_argument("--min-depth", type=float, default=20.0,
                   help="water shallower than this [m] becomes land (default 20; 0 keeps all)")
    p.add_argument("--out", default="bottom_new.nc")
    args = p.parse_args()

    fjord = read_fjord(args.csv, args.row)
    glacier = args.glacier or [float(fjord["Latitude"]), float(fjord["Longitude"])]
    print(f'fjord: {fjord["Glacier Name"]}, glacier at {glacier[0]:.5f} N, {glacier[1]:.5f} E')

    if len(args.via) % 2:
        raise SystemExit("--via needs lat lon pairs")
    via = [tuple(args.via[i:i + 2]) for i in range(0, len(args.via), 2)]
    # fjord.jl takes Lx, Ly and the grid spacing as metres on the ground, but projected metres
    # are k times longer here (k ~ 1.016 at 69 N). So the box is measured on the ground and
    # scaled by k only to find the points to sample.
    start = args.start or glacier
    path = FjordPath([start, *via, args.mouth], args.width, args.length, smooth=args.smooth,
                     x_center=args.glacier_x)
    k, Lx, Ly = path.k, path.Lx, path.Ly
    if via:
        fold = path.fold()
        print(f"bent box through {len(via)} via point(s), corners rounded over {path.smooth:.0f} m: "
              f"at the sharpest bend the inner side is squeezed and the outer stretched by {100 * fold:.0f}%")
        if fold >= 0.9:
            raise SystemExit("the box folds over itself at a bend: increase --smooth, narrow --width, "
                             "or move the --via points")

    nx, ny = max(2, round(Lx / args.dx)), max(2, round(Ly / args.dx))
    x = (np.arange(nx) + 0.5) * Lx / nx
    y = (np.arange(ny) + 0.5) * Ly / ny
    Xm, Ym = np.meshgrid(x, y, indexing="ij")              # shape (nx, ny), ground metres
    X, Y = path.to_xy(Xm, Ym)

    bm = BedMachine(args.bedmachine)
    elev = bm.bed(X, Y)
    source = f"BedMachine bed ({os.path.basename(args.bedmachine)}), 150 m"
    if not np.isfinite(elev).all() or elev.min() < -10_000:
        raise SystemExit("missing or fill values in the BedMachine sea floor inside the box")
    bottom = np.minimum(elev, 0.0)             # land -> 0: no water in that column

    mask = bm.sample(X, Y)
    ice = np.isin(mask, bm.ice)
    print(f"BedMachine: {100 * (ice & (bottom < 0)).mean():.0f}% of columns are below sea level "
          f"but under ice - set to 0")
    bottom[ice] = 0.0                          # ice -> 0: no water in that column either

    shallow, cut = clean_water(bottom, args.min_depth)
    print(f"set to land: {shallow} columns shallower than {args.min_depth:g} m, "
          f"{cut} columns of water cut off from the mouth")

    wet = bottom < 0
    print(f"strip: Lx = {Lx:.0f} m across, Ly = {Ly:.0f} m along (on the ground; projection scale "
          f"{k:.4f}), grid {nx} x {ny} at ~{args.dx:.0f} m")
    print(f"deepest {bottom.min():.1f} m, {100 * wet.mean():.0f}% of columns are water")
    print(f"glacier-end row: {100 * wet[:, 0].mean():.0f}% water, mouth row: {100 * wet[:, -1].mean():.0f}% water")
    table_L = float(fjord["Fjord length (km)"]) * 1e3
    if abs(Ly - table_L) > 0.1 * table_L:
        print(f"WARNING: strip length {Ly / 1e3:.1f} km differs from the table's fjord length "
              f"{table_L / 1e3:.0f} km by more than 10% - check --mouth")

    # Written as bottom(y, x) so Julia's NCDatasets reads it as (x, y), like make_bathymetry.jl.
    with netcdf_file(args.out, "w") as ds:
        ds.createDimension("x", nx)
        ds.createDimension("y", ny)
        ds.createVariable("x", "f8", ("x",))[:] = x
        ds.createVariable("y", "f8", ("y",))[:] = y
        v = ds.createVariable("bottom", "f8", ("y", "x"))
        v[:] = bottom.T
        v.long_name = f"bottom elevation (negative below sea level), from {source}"
        v.units = "m"
        ds.source = f"{source}, fjord-aligned strip"
        if mask is not None:
            v.long_name += ", 0 under BedMachine ice"
            m = ds.createVariable("mask", "b", ("y", "x"))
            m[:] = mask.T.astype(np.int8)
            m.long_name = "BedMachine Greenland mask, nearest 150 m cell"
            m.flag_values = np.asarray(getattr(bm.mask, "flag_values", [0, 1, 2, 3]), dtype=np.int8)
            m.flag_meanings = getattr(bm.mask, "flag_meanings", "ocean ice_free_land grounded_ice floating_ice")
            ds.source += f", ice masked with {os.path.basename(args.bedmachine)}"
        ds.fjord = fjord["Glacier Name"].encode("ascii", "replace")
        ds.glacier_latlon = np.array(glacier)
        if args.start:
            ds.start_latlon = np.array(start)
        ds.glacier_wall = args.glacier_wall
        ds.mouth_latlon = np.array(args.mouth)
        ds.Lx, ds.Ly = Lx, Ly                  # metres on the ground
        ds.projection_scale = k                # projected (EPSG:3996) metres per ground metre in this box
        ds.x_center = path.x0                  # model x of the glacier->mouth line
        if via:
            ds.via_latlon = np.array(via, dtype=float).ravel()
            ds.smooth = path.smooth
        ds.min_depth = args.min_depth

    lz = math.ceil(-bottom.min() / 10) * 10
    print(f"\nwrote {args.out}. In fjord.jl's FjordConfig set:")
    print(f'  Lx = {Lx:.1f}, Ly = {Ly:.1f}, Lz = {lz:.1f},')
    print(f'  bathymetry_file = "{args.out}", bathymetry_var = "bottom"')

    if mask is not None and args.glacier_wall != "y0":
        gxm, gym = path.from_xy(*latlon_to_xy(*glacier))
        f = side_front(bottom, mask, bm.ice, x, y, args.glacier_wall, gym)
        if f is None:
            print(f"\nno ice front on the {args.glacier_wall} wall within 5 km of the glacier "
                  f"(at y = {gym:.0f} m, x = {gxm:.0f} m) - check --start/--width/--glacier-x")
        else:
            fy, fd, fz, fx = f
            print(f"\nice front on the {args.glacier_wall} wall: deepest water {fz:.0f} m at y = {fy:.0f} m "
                  f"(x = {fx:.0f} m)")
            print(f"  plume_wall = :{args.glacier_wall}, plume_y_center = {fy:.1f}")
            if fd > 0:
                x0 = path.x0 - (fd if args.glacier_wall == "x0" else 0)
                print(f"  the front is {fd:.0f} m out from the wall: keep the box and set "
                      f"plume_wall_offset = {fd:.1f}, or move the wall onto it by running again with "
                      f"--width {Lx - fd:.0f}" + (f" --glacier-x {x0:.0f}" if args.glacier_wall == "x0" else ""))
    front = ice_front(bottom, mask, bm.ice, x, y) if mask is not None and args.glacier_wall == "y0" else None
    if front:
        fx, fy, fz = front
        print(f"\nice front: reaches furthest up-fjord at y = {fy:.0f} m; deepest water there is "
              f"{fz:.0f} m at x = {fx:.0f} m")
        print(f"  plume_x_center = {fx:.1f}")
        if fy > 0:
            lat, lon = xy_to_latlon(*(float(v) for v in path.to_xy(path.x0, fy)))
            print(f"  the model's glacier wall is y = 0, {fy:.0f} m behind this front. To move it onto the front,")
            print(f"  run again with --glacier {lat:.5f} {lon:.5f}"
                  + (f" --length {Ly - fy:.0f}" if args.length else ""))

    print("\nfrom the Slater et al. (2022) table, for setting up the plume:")
    for key in ["Grounding line depth (m)", "Fjord effective depth (m)", "Plume neutral buoyancy depth (m)",
                "Plume maximum height (m)", "Mean 2010-2020 JJA subglacial discharge (m^3/s)",
                "Plume upwelling flux (m^3/s)", "Fjord length (km)"]:
        print(f"  {key}: {fjord[key]}")


if __name__ == "__main__":
    main()
