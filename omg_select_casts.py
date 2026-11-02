#!/usr/bin/env python
"""
omg_select_casts.py

Pick the 'outside' (shelf / mouth) and 'inside' (near-glacier) casts for a
fjord from a folder of OMG CTD / AXCTD files, then fit and plot them with
omg_profile_fit.py.

The fjord is described in fjords.json by a centerline (mouth -> glacier
front), the sill position along it, and the sill and grounding-line depths.
Every cast is projected onto that centerline:

    along_km < 0              seaward of the mouth (shelf)
    0 <= along_km < sill_km   between mouth and sill        -> 'outside'
    along_km >= sill_km       landward of the sill          -> 'inside'
    too far from centerline   ignored ('elsewhere')

Selection rules (all adjustable on the command line):
  inside : deep enough (>= --inside-frac * H_gl), then closest to the glacier
  outside: deep enough (>= H_sill), then closest in time to the inside cast
  both   : within --max-days of each other (same campaign)

Usage
-----
    pip install numpy scipy matplotlib netCDF4 gsw
    # optional, for --download:  pip install earthaccess

    # 1. (optional) download casts for the fjord from NASA Earthdata
    python omg_select_casts.py ilulissat --download --start 2020-01-01 --end 2020-12-31

    # 2. classify, select, fit and plot
    python omg_select_casts.py ilulissat --data omg_casts/ilulissat

    # restrict to one year, or force particular files
    python omg_select_casts.py ilulissat --year 2020
    python omg_select_casts.py ilulissat --outside a.nc --inside b.nc

Outputs (in --out, default results/<fjord>/):
    casts_<fjord>.csv          every cast with position, date, depth, class
    map_<fjord>.png            cast positions, centerline and sill
    omg_profile_fits.png       Fig. 2f-g style profiles of the chosen casts
    ambient_profiles.jl        Julia functions for the chosen casts
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omg_profile_fit as opf   # noqa: E402

R_EARTH = 6371.0


# --------------------------------------------------------------------------
# 1. Fjord configuration
# --------------------------------------------------------------------------

def load_fjord(name, config_path):
    with open(config_path) as fh:
        db = json.load(fh)
    key = name.strip().lower()
    if key not in db or key.startswith("_"):
        options = [k for k in db if not k.startswith("_")]
        sys.exit(f"Fjord '{name}' not in {config_path}. Available: {options}. "
                 f"Add an entry (copy 'my_fjord').")
    cfg = db[key]
    cl = np.asarray(cfg["centerline"], float)
    if cl.shape[0] < 2 or np.allclose(cl, 0):
        sys.exit(f"Fill in the centerline for '{key}' in {config_path}.")
    for k in ("sill_km", "H_sill", "H_gl"):
        if not cfg.get(k):
            print(f"[warning] '{k}' is zero/missing for '{key}'; selection rules that use it are weak.")
    if "_note" in cfg:
        print(f"[note] {key}: {cfg['_note']}")
    cfg["name"] = key
    cfg["centerline"] = cl
    return cfg


# --------------------------------------------------------------------------
# 2. Geometry: project casts onto the centerline
# --------------------------------------------------------------------------

def _to_xy(lat, lon, lat0, lon0):
    x = np.radians(np.asarray(lon) - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * R_EARTH
    return x, y


def along_cross(lat, lon, centerline):
    """Distance along the centerline from the mouth (km; negative = seaward)
    and perpendicular distance from it (km)."""
    lat0, lon0 = centerline[:, 0].mean(), centerline[:, 1].mean()
    cx, cy = _to_xy(centerline[:, 0], centerline[:, 1], lat0, lon0)
    px, py = _to_xy(lat, lon, lat0, lon0)
    seg_len = np.hypot(np.diff(cx), np.diff(cy))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    best = (np.inf, np.nan)
    nseg = len(seg_len)
    for i in range(nseg):
        dx, dy = cx[i + 1] - cx[i], cy[i + 1] - cy[i]
        t = ((px - cx[i]) * dx + (py - cy[i]) * dy) / seg_len[i] ** 2
        lo = -np.inf if i == 0 else 0.0            # extend first segment seaward
        hi = np.inf if i == nseg - 1 else 1.0      # extend last segment landward
        t = min(max(t, lo), hi)
        d = np.hypot(px - (cx[i] + t * dx), py - (cy[i] + t * dy))
        if d < best[0]:
            best = (d, cum[i] + t * seg_len[i])
    return best[1], best[0]


def classify(along, cross, cfg):
    if along >= cfg["sill_km"]:
        return "inside" if cross <= cfg["max_cross_km"] else "elsewhere"
    if along >= 0:
        return "outside" if cross <= cfg["max_cross_km"] else "elsewhere"
    if -along <= cfg["shelf_km"] and cross <= cfg["shelf_cross_km"]:
        return "outside"
    return "elsewhere"


def bbox(cfg, margin_km=None):
    """(west, south, east, north) around the centerline plus the shelf margin."""
    m = margin_km if margin_km is not None else cfg["shelf_km"] + 10
    cl = cfg["centerline"]
    dlat = m / 111.0
    dlon = m / (111.0 * np.cos(np.radians(cl[:, 0].mean())))
    return (cl[:, 1].min() - dlon, cl[:, 0].min() - dlat,
            cl[:, 1].max() + dlon, cl[:, 0].max() + dlat)


# --------------------------------------------------------------------------
# 3. Cast time (from the file, else from the file name)
# --------------------------------------------------------------------------

def _parse_cf_time(value, units):
    m = re.match(r"\s*(\w+)\s+since\s+(.+)", units or "")
    if not m:
        return None
    unit, ref = m.group(1).lower(), m.group(2).strip().replace("T", " ").rstrip("Z")
    ref = re.sub(r"\s*(UTC|utc|\+00:?00)$", "", ref)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            t0 = datetime.strptime(ref, fmt)
            break
        except ValueError:
            t0 = None
    if t0 is None:
        return None
    scale = {"seconds": 1, "second": 1, "s": 1, "minutes": 60, "minute": 60,
             "hours": 3600, "hour": 3600, "days": 86400, "day": 86400}.get(unit)
    return None if scale is None else t0 + timedelta(seconds=float(value) * scale)


def cast_time(path):
    try:
        try:
            import netCDF4
            with netCDF4.Dataset(path) as ds:
                for name in ("time", "TIME", "profile_time", "date_time", "juld"):
                    if name in ds.variables:
                        v = ds.variables[name]
                        val = float(np.ma.filled(np.ravel(v[:])[0], np.nan))
                        t = _parse_cf_time(val, getattr(v, "units", ""))
                        if t:
                            return t
                for a in ("time_coverage_start", "date", "start_time"):
                    if a in ds.ncattrs():
                        return datetime.fromisoformat(str(ds.getncattr(a))[:19].replace("Z", ""))
        except ImportError:
            from scipy.io import netcdf_file
            with netcdf_file(path, "r", mmap=False) as ds:
                for name in ("time", "TIME", "profile_time"):
                    if name in ds.variables:
                        v = ds.variables[name]
                        units = v._attributes.get("units", b"")
                        units = units.decode() if isinstance(units, bytes) else units
                        t = _parse_cf_time(float(np.ravel(v[:])[0]), units)
                        if t:
                            return t
    except Exception:
        pass
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", os.path.basename(path))
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


# --------------------------------------------------------------------------
# 4. Optional download from NASA Earthdata
# --------------------------------------------------------------------------

def download(cfg, dest, start, end, keywords):
    try:
        import earthaccess
    except ImportError:
        sys.exit("pip install earthaccess  (and create a free NASA Earthdata login)")
    earthaccess.login()
    box = bbox(cfg)
    os.makedirs(dest, exist_ok=True)
    collections = []
    for kw in keywords:
        collections += earthaccess.search_datasets(keyword=kw, bounding_box=box,
                                                   temporal=(start, end))
    seen, total = set(), 0
    for col in collections:
        cid = col.concept_id()
        if cid in seen:
            continue
        seen.add(cid)
        short = col.summary().get("short-name", cid)
        if "OMG" not in short.upper():
            continue
        granules = earthaccess.search_data(concept_id=cid, bounding_box=box,
                                           temporal=(start, end))
        print(f"  {short}: {len(granules)} file(s)")
        if granules:
            earthaccess.download(granules, local_path=dest)
            total += len(granules)
    if total == 0:
        print("No OMG files found. Check the dates and the fjord centerline, or "
              "find the collection on https://search.earthdata.nasa.gov and pass "
              "its name with --keywords.")
    return total


# --------------------------------------------------------------------------
# 5. Inventory and selection
# --------------------------------------------------------------------------

def inventory(files, cfg):
    rows = []
    for f in files:
        try:
            c = opf.read_cast(f)
        except Exception as e:
            print(f"  skipping {os.path.basename(f)}: {e}")
            continue
        if not (np.isfinite(c["lat"]) and np.isfinite(c["lon"])):
            print(f"  skipping {os.path.basename(f)}: no position in file")
            continue
        a, x = along_cross(c["lat"], c["lon"], cfg["centerline"])
        zmax = float(np.nanmax(c["zraw"])) if c["zraw"].size else 0.0
        t = cast_time(f)
        rows.append(dict(file=f, lat=c["lat"], lon=c["lon"], time=t,
                         max_depth=zmax, along_km=a, cross_km=x,
                         cls=classify(a, x, cfg)))
    return rows


def select(rows, cfg, max_days, inside_frac, year=None):
    if year:
        rows = [r for r in rows if r["time"] and r["time"].year == year]
    ins = [r for r in rows if r["cls"] == "inside"]
    outs = [r for r in rows if r["cls"] == "outside"]
    if not ins or not outs:
        return None, None, f"need at least one inside and one outside cast (have {len(ins)} / {len(outs)})"

    need_in = inside_frac * cfg.get("H_gl", 0)
    need_out = cfg.get("H_sill", 0)
    deep_in = [r for r in ins if r["max_depth"] >= need_in] or ins
    deep_out = [r for r in outs if r["max_depth"] >= need_out] or outs
    msg = []
    if deep_in is ins and need_in and not any(r["max_depth"] >= need_in for r in ins):
        msg.append(f"no inside cast reaches {need_in:.0f} m; using shallower ones")
    if deep_out is outs and need_out and not any(r["max_depth"] >= need_out for r in outs):
        msg.append(f"no outside cast reaches sill depth {need_out:.0f} m")

    def dt_days(a, b):
        if a["time"] is None or b["time"] is None:
            return np.inf
        return abs((a["time"] - b["time"]).total_seconds()) / 86400

    # Try inside casts from closest-to-glacier outward; take the first with a
    # same-campaign outside partner.
    for rin in sorted(deep_in, key=lambda r: -r["along_km"]):
        partners = sorted(deep_out, key=lambda r: dt_days(rin, r))
        if dt_days(rin, partners[0]) <= max_days:
            return rin, partners[0], "; ".join(msg)
    rin = max(deep_in, key=lambda r: r["along_km"])
    rout = min(deep_out, key=lambda r: dt_days(rin, r))
    msg.append(f"no pair within {max_days:g} days; closest pair is "
               f"{dt_days(rin, rout):.0f} days apart")
    return rin, rout, "; ".join(msg)


# --------------------------------------------------------------------------
# 6. Output: table and map
# --------------------------------------------------------------------------

def write_table(rows, path, chosen):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "class", "selected", "date", "lat", "lon",
                    "max_depth_m", "along_km", "cross_km"])
        for r in sorted(rows, key=lambda r: (r["cls"], r["along_km"])):
            sel = next((k for k, v in chosen.items() if v is r), "")
            w.writerow([os.path.basename(r["file"]), r["cls"], sel,
                        r["time"].strftime("%Y-%m-%d %H:%M") if r["time"] else "",
                        f"{r['lat']:.4f}", f"{r['lon']:.4f}", f"{r['max_depth']:.0f}",
                        f"{r['along_km']:.1f}", f"{r['cross_km']:.1f}"])


def plot_map(rows, cfg, chosen, path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cl = cfg["centerline"]
    ax.plot(cl[:, 1], cl[:, 0], "k-", lw=1.5, label="centerline")
    ax.plot(cl[0, 1], cl[0, 0], "ks", ms=6, label="mouth")
    ax.plot(cl[-1, 1], cl[-1, 0], "b^", ms=9, label="glacier front")

    # sill marker: point along the centerline at sill_km
    lat0, lon0 = cl[:, 0].mean(), cl[:, 1].mean()
    cx, cy = _to_xy(cl[:, 0], cl[:, 1], lat0, lon0)
    cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
    sx, sy = np.interp(cfg["sill_km"], cum, cx), np.interp(cfg["sill_km"], cum, cy)
    slat = lat0 + np.degrees(sy / R_EARTH)
    slon = lon0 + np.degrees(sx / (R_EARTH * np.cos(np.radians(lat0))))
    ax.plot(slon, slat, "r|", ms=18, mew=3, label="sill")

    colors = {"outside": "#d95f02", "inside": "#1f3fff", "elsewhere": "0.7"}
    for cls, col in colors.items():
        sub = [r for r in rows if r["cls"] == cls]
        if sub:
            ax.scatter([r["lon"] for r in sub], [r["lat"] for r in sub], s=22,
                       c=col, label=f"{cls} ({len(sub)})", zorder=3)
    for k, r in chosen.items():
        if r is not None:
            ax.scatter(r["lon"], r["lat"], s=160, facecolors="none",
                       edgecolors="k", lw=2, zorder=4)
            ax.annotate(f"selected {k}", (r["lon"], r["lat"]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8)
    w, s, e, n = bbox(cfg)
    ax.set_xlim(w, e); ax.set_ylim(s, n)
    ax.set_aspect(1 / np.cos(np.radians(lat0)))
    ax.set(xlabel="Longitude", ylabel="Latitude", title=f"Casts: {cfg['name']}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fjord", help="name of an entry in fjords.json")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "fjords.json"))
    ap.add_argument("--data", help="folder of cast files (default omg_casts/<fjord>)")
    ap.add_argument("--out", help="output folder (default results/<fjord>)")
    ap.add_argument("--download", action="store_true", help="download casts with earthaccess first")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--keywords", nargs="+", default=["OMG AXCTD", "OMG CTD"],
                    help="Earthdata collection search keywords")
    ap.add_argument("--year", type=int, help="only use casts from this year")
    ap.add_argument("--max-days", type=float, default=14.0,
                    help="max time between inside and outside casts [days]")
    ap.add_argument("--inside-frac", type=float, default=0.8,
                    help="inside cast must reach this fraction of H_gl")
    ap.add_argument("--outside", help="force this file as the outside cast")
    ap.add_argument("--inside", help="force this file as the inside cast")
    ap.add_argument("--no-fit", action="store_true", help="only classify and map")
    ap.add_argument("--fit-args", default="",
                    help='extra options for omg_profile_fit, e.g. "--T-break -60 --S-deg 5 3"')
    a = ap.parse_args(argv)

    cfg = load_fjord(a.fjord, a.config)
    data = a.data or os.path.join("omg_casts", cfg["name"])
    out = a.out or os.path.join("results", cfg["name"])
    os.makedirs(out, exist_ok=True)

    if a.download:
        print(f"Downloading casts for {cfg['name']} into {data} ...")
        download(cfg, data, a.start, a.end, a.keywords)

    files = sorted(glob.glob(os.path.join(data, "**", "*.nc"), recursive=True) +
                   glob.glob(os.path.join(data, "**", "*.csv"), recursive=True))
    if not files:
        sys.exit(f"No .nc or .csv files in {data}. Download them first (--download) "
                 f"or point --data at your folder.")

    print(f"Reading {len(files)} file(s) ...")
    rows = inventory(files, cfg)
    by = {k: sum(r["cls"] == k for r in rows) for k in ("outside", "inside", "elsewhere")}
    print(f"Classified: {by['outside']} outside, {by['inside']} inside, {by['elsewhere']} elsewhere")

    if a.outside or a.inside:
        find = lambda f: next((r for r in rows if os.path.samefile(r["file"], f)), None)
        auto_in, auto_out, msg = select(rows, cfg, a.max_days, a.inside_frac, a.year)
        rin = find(a.inside) if a.inside else auto_in
        rout = find(a.outside) if a.outside else auto_out
    else:
        rin, rout, msg = select(rows, cfg, a.max_days, a.inside_frac, a.year)
    if msg:
        print(f"[selection] {msg}")
    chosen = {"outside": rout, "inside": rin}

    write_table(rows, os.path.join(out, f"casts_{cfg['name']}.csv"), chosen)
    plot_map(rows, cfg, chosen, os.path.join(out, f"map_{cfg['name']}.png"))
    print(f"Saved {out}/casts_{cfg['name']}.csv and map_{cfg['name']}.png")

    if rin is None or rout is None:
        sys.exit("Could not select a pair. Check the map: adjust the centerline, sill_km, "
                 "max_cross_km or shelf_km in fjords.json, or pass --outside/--inside.")
    for k, r in chosen.items():
        d = r["time"].strftime("%Y-%m-%d") if r["time"] else "unknown date"
        print(f"  {k:7s}: {os.path.basename(r['file'])}  ({d}, {r['max_depth']:.0f} m deep, "
              f"{r['along_km']:.1f} km along fjord)")

    if a.no_fit:
        return
    fit_argv = [rout["file"], rin["file"], "--labels", "outside", "inside",
                "--png", os.path.join(out, "omg_profile_fits.png"),
                "--julia", os.path.join(out, "ambient_profiles.jl"),
                "--toml", os.path.join(out, "ambient_profiles.toml")]
    fit_argv += a.fit_args.split()
    print("\nFitting: python omg_profile_fit.py " + " ".join(fit_argv))
    opf.main(fit_argv)


if __name__ == "__main__":
    main()
