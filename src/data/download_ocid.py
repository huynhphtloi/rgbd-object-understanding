"""
Download + extract the OCID dataset.

OCID is hosted by TU Wien. The download is large (tens of GB). Because the
hosting URL can change and may require accepting terms, this script takes the
archive URL as an argument and streams it to disk, then extracts it.

Usage
-----
    python3 -m src.data.download_ocid --url <ARCHIVE_URL> --out data/raw/ocid

If you already downloaded the archive manually, point --archive at it:
    python3 -m src.data.download_ocid --archive OCID.zip --out data/raw/ocid

Dataset page (for the current link + license):
    https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
import zipfile


def download(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")

    def _hook(block, bsize, total):
        if total > 0:
            pct = min(100, block * bsize * 100 / total)
            sys.stdout.write(f"\r  {pct:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _hook)
    print("\n  done.")
    return dest


def extract(archive: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Extracting {archive} -> {out_dir}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")
    print("  done.")


def main():
    ap = argparse.ArgumentParser(description="Download/extract OCID")
    ap.add_argument("--url", help="archive URL to download")
    ap.add_argument("--archive", help="path to an already-downloaded archive")
    ap.add_argument("--out", default="data/raw/ocid", help="extraction directory")
    ap.add_argument("--keep-archive", action="store_true")
    args = ap.parse_args()

    if not args.url and not args.archive:
        ap.error("provide --url to download, or --archive for a local file")

    archive = args.archive
    if args.url:
        archive = os.path.join(args.out + "_download",
                               os.path.basename(args.url.split("?")[0]) or "ocid.zip")
        download(args.url, archive)

    extract(archive, args.out)

    if args.url and not args.keep_archive:
        os.remove(archive)
        print(f"Removed {archive}")

    print(f"\nOCID extracted under: {args.out}\n"
          f"Next: python3 -m src.data.inspect_ocid --root {args.out}")


if __name__ == "__main__":
    main()
