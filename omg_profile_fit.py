#!/usr/bin/env python
"""
omg_profile_fit.py

Load OMG (or any) CTD / AXCTD casts, fit piecewise polynomials to the
temperature and salinity profiles, print the fits as Julia functions, and
plot data (solid) vs. fit (dashed) in the style of Zhao et al. (2023)
Fig. 2f-g: salinity on the bottom axis, theta on the top axis, z in km.

Usage
-----
    pip install numpy scipy matplotlib netCDF4 gsw     # gsw strongly recommended

    # one cast
    python omg_profile_fit.py inside_cast.nc

    # outside + inside casts, side by side, with custom labels
    python omg_profile_fit.py shelf.nc fjord.nc --labels outside inside

    # change the break depths / polynomial degrees
    python omg_profile_fit.py fjord.nc --T-break -40 --S-break -80 \
        --T-deg 7 1 --S-deg 7 3

Each profile is fitted in two pieces, as in the Oceananigans T∞/S∞ functions:
a high-degree polynomial above the break depth (where the near-surface
structure is complicated) and a low-degree polynomial below it. By default
the upper piece is pinned to the lower piece at the break so the profile is
continuous.

Files can be NetCDF (.nc) or CSV (.csv with columns for depth or pressure,
temperature and salinity). Variable names are guessed; override them with
--zvar/--tvar/--svar if the guess fails (the script prints what it found).
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    import gsw
except ImportError:
    gsw = None


# --------------------------------------------------------------------------
# 1. Reading casts
# --------------------------------------------------------------------------

Z_NAMES = ["depth", "DEPTH", "depSM", "z", "pressure", "PRES", "pres", "prDM", "p"]
T_NAMES = ["temperature", "TEMP", "temp", "t090C", "t", "potential_temperature", "theta", "ptemp"]
S_NAMES = ["salinity", "PSAL", "psal", "sal00", "salt", "practical_salinity", "S"]
LAT_NAMES = ["lat", "latitude", "LATITUDE", "LAT"]
LON_NAMES = ["lon", "longitude", "LONGITUDE", "LON"]

STD_NAMES = {
    "z": ["depth", "sea_water_pressure", "height"],
    "t": ["sea_water_temperature", "sea_water_potential_temperature",
          "sea_water_conservative_temperature"],
    "s": ["sea_water_practical_salinity", "sea_water_salinity"],
}


def _open_netcdf(path):
    """Return (variables: dict name->(array, attrs), global_attrs)."""
    try:
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(True)
            vars_ = {}
            for name, v in ds.variables.items():
                arr = np.ma.filled(v[:].astype(float), np.nan) if v.dtype.kind in "fiu" else None
                vars_[name] = (arr, {a: v.getncattr(a) for a in v.ncattrs()})
            gattrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
        return vars_, gattrs
    except ImportError:
        pass
    # Fallback: scipy only reads NetCDF3 (classic) files
    from scipy.io import netcdf_file
    with netcdf_file(path, "r", mmap=False) as ds:
        vars_ = {}
        for name, v in ds.variables.items():
            arr = np.array(v[:], dtype=float) if v.data.dtype.kind in "fiu" else None
            if arr is not None:
                fill = getattr(v, "_FillValue", None)
                if fill is not None:
                    arr[arr == fill] = np.nan
            vars_[name] = (arr, dict(v._attributes))
        gattrs = {k: (v.decode() if isinstance(v, bytes) else v)
                  for k, v in ds._attributes.items()}
    return vars_, gattrs


def _pick(vars_, names, std_names, override=None):
    if override:
        if override not in vars_:
            raise KeyError(f"variable '{override}' not in file; available: {list(vars_)}")
        return override
    lower = {k.lower(): k for k in vars_}
    for n in names:
        if n in vars_:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    for k, (_, attrs) in vars_.items():
        sn = str(attrs.get("standard_name", "")).lower()
        if sn in std_names:
            return k
    return None


def _scalar(vars_, gattrs, names):
    for n in names:
        if n in vars_ and vars_[n][0] is not None:
            return float(np.nanmean(vars_[n][0]))
        for k, v in gattrs.items():
            if k.lower() == n.lower():
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    return np.nan


def read_cast(path, zvar=None, tvar=None, svar=None, profile=0, lat=None):
    """Return dict with z (m, negative down), T, S, lat, lon and metadata."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        vars_ = {c: (df[c].to_numpy(dtype=float), {}) for c in df.columns
                 if np.issubdtype(df[c].dtype, np.number)}
        gattrs = {}
    else:
        vars_, gattrs = _open_netcdf(path)

    kz = _pick(vars_, Z_NAMES, STD_NAMES["z"], zvar)
    kt = _pick(vars_, T_NAMES, STD_NAMES["t"], tvar)
    ks = _pick(vars_, S_NAMES, STD_NAMES["s"], svar)
    if None in (kz, kt, ks):
        raise KeyError(f"{path}: could not identify variables (z={kz}, T={kt}, S={ks}).\n"
                       f"Available: {list(vars_)}\nUse --zvar/--tvar/--svar.")

    def get(k):
        a = np.asarray(vars_[k][0], dtype=float)
        if a.ndim == 2:                       # (profile, level) files
            a = a[profile] if a.shape[0] < a.shape[1] else a[:, profile]
        return a.squeeze()

    zr, t, s = get(kz), get(kt), get(ks)
    la = lat if lat is not None else _scalar(vars_, gattrs, LAT_NAMES)
    lo = _scalar(vars_, gattrs, LON_NAMES)

    is_pressure = kz.lower().startswith(("p", "pr")) or \
        "pressure" in str(vars_[kz][1].get("standard_name", "")).lower() or \
        str(vars_[kz][1].get("units", "")).lower() in ("dbar", "decibar", "db")
    t_is_potential = any(w in (kt + str(vars_[kt][1].get("standard_name", ""))).lower()
                         for w in ("potential", "theta", "ptemp", "conservative"))

    print(f"{os.path.basename(path)}: z='{kz}' ({'pressure' if is_pressure else 'depth'}), "
          f"T='{kt}', S='{ks}', lat={la:.3f}, lon={lo:.3f}, n={zr.size}")
    return dict(path=path, zraw=np.abs(zr), t=t, s=s, lat=la, lon=lo,
                is_pressure=is_pressure, t_is_potential=t_is_potential)


# --------------------------------------------------------------------------
# 2. Clean, convert to theta, bin-average
# --------------------------------------------------------------------------

def prepare(cast, dz=1.0, use_theta=True):
    zr, t, s = cast["zraw"], cast["t"], cast["s"]
    ok = (np.isfinite(zr) & np.isfinite(t) & np.isfinite(s)
          & (t > -3) & (t < 30) & (s > 0) & (s < 42))
    zr, t, s = zr[ok], t[ok], s[ok]
    lat = cast["lat"] if np.isfinite(cast["lat"]) else 70.0
    lon = cast["lon"] if np.isfinite(cast["lon"]) else -45.0

    if gsw is not None:
        if cast["is_pressure"]:
            p = zr
            z = gsw.z_from_p(p, lat)
        else:
            z = -zr
            p = gsw.p_from_z(z, lat)
        if use_theta and not cast["t_is_potential"]:
            SA = gsw.SA_from_SP(s, p, lon, lat)
            t = gsw.pt0_from_t(SA, t, p)
    else:
        print("  [warning] gsw not installed: using 1 dbar = 1 m and in-situ T "
              "(error < ~0.05 degC above 1000 m). pip install gsw to fix.")
        z = -zr

    # Bin-average onto dz bins (removes spikes and duplicate depths)
    edges = np.arange(np.floor(z.min() / dz) * dz, dz, dz)
    idx = np.digitize(z, edges)
    zb, tb, sb = [], [], []
    for i in np.unique(idx):
        sel = idx == i
        zb.append(z[sel].mean()); tb.append(t[sel].mean()); sb.append(s[sel].mean())
    order = np.argsort(zb)
    return np.array(zb)[order], np.array(tb)[order], np.array(sb)[order]


# --------------------------------------------------------------------------
# 3. Piecewise polynomial fit
# --------------------------------------------------------------------------

def fit_piecewise(z, y, zbreak, deg_up, deg_dn, continuous=True):
    """Fit y(z) with a polynomial above zbreak and another below it.

    Returns raw power-series coefficients (highest power first, like
    np.polyfit) so they can be written straight into Julia.
    """
    from numpy.polynomial import Polynomial as P
    up, dn = z > zbreak, z <= zbreak
    if dn.sum() <= deg_dn or up.sum() <= deg_up:
        raise ValueError(f"not enough points either side of z = {zbreak} m "
                         f"({up.sum()} above, {dn.sum()} below)")

    # Fit in a scaled domain (numerically stable), then convert to raw z
    p_dn = P.fit(z[dn], y[dn], deg_dn)
    zu, yu, w = z[up], y[up], np.ones(up.sum())
    if continuous:          # pin the upper piece to the lower one at the break
        zu = np.append(zu, zbreak); yu = np.append(yu, p_dn(zbreak))
        w = np.append(w, 1e3)
    p_up = P.fit(zu, yu, deg_up, w=w)

    c_up = p_up.convert().coef[::-1]
    c_dn = p_dn.convert().coef[::-1]
    rms_up = np.sqrt(np.mean((np.polyval(c_up, z[up]) - y[up]) ** 2))
    rms_dn = np.sqrt(np.mean((np.polyval(c_dn, z[dn]) - y[dn]) ** 2))
    jump = np.polyval(c_up, zbreak) - np.polyval(c_dn, zbreak)
    return dict(c_up=c_up, c_dn=c_dn, zbreak=zbreak, zmin=z.min(),
                rms_up=rms_up, rms_dn=rms_dn, jump=jump)


def eval_piecewise(f, z):
    zc = np.clip(z, f["zmin"], 0.0)          # hold constant outside the data
    return np.where(zc > f["zbreak"], np.polyval(f["c_up"], zc), np.polyval(f["c_dn"], zc))


# --------------------------------------------------------------------------
# 4. Julia output
# --------------------------------------------------------------------------

def _poly_julia(c):
    n = len(c) - 1
    terms = []
    for i, a in enumerate(c):
        k = n - i
        mono = "" if k == 0 else ("*z" if k == 1 else f"*z^{k}")
        terms.append((a, f"{abs(a):.10g}{mono}"))
    s = ("-" if terms[0][0] < 0 else "") + terms[0][1]
    for a, txt in terms[1:]:
        s += (" - " if a < 0 else " + ") + txt
    return s


def toml_profile(label, var, f):
    """One fit as a TOML table, read by fjord.jl's `ambient_file` (coefficients highest power first)."""
    nums = lambda c: "[" + ", ".join(repr(float(a)) for a in c) + "]"
    return (f"[{label}.{var}]\nzmin = {float(f['zmin'])!r}\nzbreak = {float(f['zbreak'])!r}\n"
            f"up = {nums(f['c_up'])}\ndn = {nums(f['c_dn'])}\n")


def julia_function(name, f):
    return (f"@inline function {name}(z)\n"
            f"    z = clamp(z, {f['zmin']:.1f}, 0.0)   # hold constant outside the cast\n"
            f"    if z > {f['zbreak']:g}\n"
            f"        return {_poly_julia(f['c_up'])}\n"
            f"    else\n"
            f"        return {_poly_julia(f['c_dn'])}\n"
            f"    end\n"
            f"end\n")


# --------------------------------------------------------------------------
# 5. Plot (Zhao et al. 2023 Fig. 2f-g style)
# --------------------------------------------------------------------------

def plot_profiles(results, png):
    cS, cT = "#d95f02", "#1f3fff"
    n = len(results)
    fig, axs = plt.subplots(1, n, figsize=(3.4 * n, 4.6), squeeze=False)
    letters = "fghijklmn" if n == 2 else "abcdefghij"
    for k, (ax, r) in enumerate(zip(axs[0], results)):
        zkm = r["z"] / 1000
        zf = np.linspace(r["z"].min(), 0, 800)

        ax.plot(r["S"], zkm, color=cS, lw=3.5, alpha=0.45, label="S data")
        ax.plot(eval_piecewise(r["fS"], zf), zf / 1000, "--", color="#8c2d04", lw=1.4, label="S fit")
        ax.set_xlabel("Salinity (psu)", color=cS)
        ax.tick_params(axis="x", colors=cS)
        ax.set_ylim(r["z"].min() / 1000, 0)
        ax.grid(True, alpha=0.4)

        axT = ax.twiny()
        axT.plot(r["T"], zkm, color=cT, lw=3.5, alpha=0.45, label="θ data")
        axT.plot(eval_piecewise(r["fT"], zf), zf / 1000, "--", color="#08306b", lw=1.4, label="θ fit")
        axT.set_xlabel("θ (°C)" if r["theta"] else "T (°C)", color=cT)
        axT.tick_params(axis="x", colors=cT)

        for zb, c in [(r["fT"]["zbreak"], cT), (r["fS"]["zbreak"], cS)]:
            ax.axhline(zb / 1000, color=c, ls=":", lw=0.8)

        if k == n - 1 and n > 1:
            ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
        if k == 0 or k == n - 1:
            ax.set_ylabel("z (km)")
        axT.set_title(f"({letters[k]}) {r['label']}", loc="left", fontsize=10)
        r["axT"] = axT

    h1, l1 = axs[0, 0].get_legend_handles_labels()
    h2, l2 = results[0]["axT"].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(png, dpi=200)
    print(f"Saved figure: {png}")


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="cast files (.nc or .csv), e.g. outside then inside")
    ap.add_argument("--labels", nargs="+", help="panel labels, one per file")
    ap.add_argument("--T-break", type=float, default=-40.0, help="break depth for T fit [m]")
    ap.add_argument("--S-break", type=float, default=-80.0, help="break depth for S fit [m]")
    ap.add_argument("--T-deg", type=int, nargs=2, default=[7, 1], metavar=("UP", "DOWN"))
    ap.add_argument("--S-deg", type=int, nargs=2, default=[7, 3], metavar=("UP", "DOWN"))
    ap.add_argument("--zmin", type=float, default=None,
                    help="ignore data below this depth [m], e.g. -650 for a grounding line")
    ap.add_argument("--ztop", type=float, default=None,
                    help="ignore data above this depth [m], e.g. -1 to drop a bad surface bin")
    ap.add_argument("--dz", type=float, default=1.0, help="bin size [m]")
    ap.add_argument("--in-situ", action="store_true", help="fit in-situ T instead of theta")
    ap.add_argument("--no-continuity", action="store_true",
                    help="do not pin the upper fit to the lower fit at the break")
    ap.add_argument("--zvar"); ap.add_argument("--tvar"); ap.add_argument("--svar")
    ap.add_argument("--profile", type=int, default=0, help="profile index in multi-cast files")
    ap.add_argument("--lat", type=float, default=None, help="latitude if not in the file")
    ap.add_argument("--png", default="omg_profile_fits.png")
    ap.add_argument("--julia", default="ambient_profiles.jl")
    ap.add_argument("--toml", default="ambient_profiles.toml",
                    help="the same fits as data, for fjord.jl's --ambient_file")
    a = ap.parse_args(argv)

    labels = a.labels or [os.path.splitext(os.path.basename(f))[0] for f in a.files]
    if len(labels) != len(a.files):
        sys.exit("--labels needs one label per file")

    results, julia = [], ["# Generated by omg_profile_fit.py\n"]
    toml = ["# Generated by omg_profile_fit.py: piecewise polynomial fits, z in m (negative down),\n"
            "# T in degC, S in psu. Coefficients highest power first (as np.polyval); `up` applies\n"
            "# for z > zbreak, `dn` below; z is clamped to [zmin, 0]. Use with fjord.jl --ambient_file.\n"]
    for path, label in zip(a.files, labels):
        cast = read_cast(path, a.zvar, a.tvar, a.svar, a.profile, a.lat)
        z, T, S = prepare(cast, a.dz, use_theta=not a.in_situ)
        if a.zmin is not None:
            keep = z >= a.zmin
            z, T, S = z[keep], T[keep], S[keep]
        if a.ztop is not None:
            keep = z <= a.ztop
            z, T, S = z[keep], T[keep], S[keep]

        fT = fit_piecewise(z, T, a.T_break, *a.T_deg, continuous=not a.no_continuity)
        fS = fit_piecewise(z, S, a.S_break, *a.S_deg, continuous=not a.no_continuity)
        for nm, f, u in [("T", fT, "degC"), ("S", fS, "psu")]:
            print(f"  {nm}: rms above break {f['rms_up']:.3f} {u}, below {f['rms_dn']:.3f} {u}, "
                  f"jump at break {f['jump']:+.4f} {u}")

        tag = "".join(ch if ch.isalnum() else "_" for ch in label)
        julia.append(f"#+++ {label}: fitted from {os.path.basename(path)} "
                     f"(data {z.min():.0f} to {z.max():.0f} m)\n")
        julia.append(julia_function(f"T_{tag}", fT))
        julia.append(julia_function(f"S_{tag}", fS))
        julia.append("#---\n")
        toml.append(f'[{tag}]\nsource = "{os.path.basename(path)}"\n')
        toml.append(toml_profile(tag, "T", fT))
        toml.append(toml_profile(tag, "S", fS))
        results.append(dict(label=label, z=z, T=T, S=S, fT=fT, fS=fS,
                            theta=(gsw is not None and not a.in_situ) or cast["t_is_potential"]))

    with open(a.julia, "w") as fh:
        fh.write("\n".join(julia))
    print(f"Saved Julia functions: {a.julia}")
    with open(a.toml, "w") as fh:
        fh.write("\n".join(toml))
    print(f"Saved fits for fjord.jl --ambient_file: {a.toml}")
    plot_profiles(results, a.png)


if __name__ == "__main__":
    main()
