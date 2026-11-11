#!/usr/bin/env python
"""
Download OMG CTD and AXCTD casts near the Ilulissat Icefjord mouth from PO.DAAC.

    pip install earthaccess xarray netCDF4

Needs a free NASA Earthdata login (earthaccess will prompt, or reads ~/.netrc).

    python get_omg_casts.py            # search + download
    python get_omg_casts.py --dry      # search only, print granule counts
"""
import argparse
from pathlib import Path

import earthaccess

# (west, south, east, north), same box as the QGreenland clip script
BBOX = (-52.5, 68.9, -50.5, 69.5)

# PO.DAAC collection short names. If a search returns 0, check the name at
# https://podaac.jpl.nasa.gov (search "OMG CTD").
COLLECTIONS = ["OMG_L2_CTD", "OMG_L2_AXCTD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="omg_casts")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    earthaccess.login()

    for name in COLLECTIONS:
        results = earthaccess.search_data(short_name=name, bounding_box=BBOX)
        print(f"{name}: {len(results)} granules in box")
        if args.dry or not results:
            continue
        dest = Path(args.out) / name
        dest.mkdir(parents=True, exist_ok=True)
        earthaccess.download(results, str(dest))

    if not args.dry:
        print(f"\nFiles in {args.out}/. Open with xarray.open_dataset(path).")


if __name__ == "__main__":
    main()
