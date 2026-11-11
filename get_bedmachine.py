#!/usr/bin/env python
"""
Download BedMachine Greenland (NSIDC dataset IDBMG4) for make_bottom.py.

    pip install earthaccess

Needs a free NASA Earthdata login (earthaccess will prompt, or reads ~/.netrc).

    python get_bedmachine.py --dry          # list available files, download nothing
    python get_bedmachine.py                # download the file matching "v7" to ~/Downloads
    python get_bedmachine.py --match v6     # pick a different version
    python get_bedmachine.py -o data/       # different output folder
"""
import argparse
import sys
from pathlib import Path

import earthaccess

SHORT_NAME = "IDBMG4"


def filenames(granule):
    return [link.split("/")[-1] for link in granule.data_links()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(Path.home() / "Downloads"))
    ap.add_argument("--match", default="v7",
                    help="substring the .nc filename must contain (default: v7)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    earthaccess.login()

    results = earthaccess.search_data(short_name=SHORT_NAME)
    if not results:
        sys.exit(f"No granules found for {SHORT_NAME}. Check the dataset at "
                 "https://nsidc.org/data/idbmg4")

    print(f"{SHORT_NAME}: {len(results)} granule(s) available")
    for g in results:
        for f in filenames(g):
            print("   ", f)

    picked = [g for g in results
              if any(args.match in f and f.endswith(".nc") for f in filenames(g))]

    if not picked:
        sys.exit(f"\nNo .nc file containing '{args.match}'. "
                 "Rerun with --match set to one of the names above.")

    if args.dry:
        print(f"\nWould download: {[f for g in picked for f in filenames(g)]}")
        return

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    paths = earthaccess.download(picked, str(out))

    print("\nDownloaded:")
    for p in paths:
        print("   ", p)
    print("\nIf the name differs from what make_bottom.py expects, run it with\n"
          "    python make_bottom.py --bedmachine <path above> ...")


if __name__ == "__main__":
    main()
