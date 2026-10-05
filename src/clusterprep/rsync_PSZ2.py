#!/usr/bin/env python3
"""
PSZ2 downloader: re-downloads sources with incomplete FITS file sets.

Checks for expected FITS files (RAW, T25kpc, T50kpc, T100kpc, and SUB variants)
and re-downloads if any are missing.

Usage:
    python scripts/0.1.0.rsync_PSZ2.py
    python scripts/0.1.0.rsync_PSZ2.py --force
    python scripts/0.1.0.rsync_PSZ2.py --check-only

MIT License

Copyright (c) 2025 Markus Johan Bredberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import requests
import tarfile
import os
import csv
import argparse
import shutil
from pathlib       import Path
from bs4           import BeautifulSoup
from urllib.parse  import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Inlined completeness check (avoids importing torch via dcreclass) ─────────
def check_complete_download(dest_dir, slug):
    """Return (is_complete, n_existing, missing_files)."""
    dest_dir = Path(dest_dir)
    if not dest_dir.exists():
        return False, 0, []
    existing = {f for f in os.listdir(dest_dir) if f.endswith('.fits')}
    if len(existing) < 2:
        return False, len(existing), []
    required = [
        f"{slug}.fits",
        f"{slug}T25kpc.fits", f"{slug}T50kpc.fits", f"{slug}T100kpc.fits",
        f"{slug}T25kpcSUB.fits", f"{slug}T50kpcSUB.fits", f"{slug}T100kpcSUB.fits",
    ]
    missing = [f for f in required if f not in existing]
    return (len(missing) == 0), len(existing), missing

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_ROOT  = Path("/Users/antoinegdrn/PH_Project_I/scratch/data/PSZ2")
FITS_ROOT = OUT_ROOT / "fits"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
FITS_ROOT.mkdir(parents=True, exist_ok=True)

# ── LOFAR surveys index ────────────────────────────────────────────────────────
BASE  = "https://lofar-surveys.org"
INDEX = BASE + "/planck_dr2.html"

def download_and_extract(page_url, slug, force=False):
    """
    Download and extract the tarball for a single PSZ2 source.

    Args:
        page_url: URL of the cluster detail page
        slug:     Source name, e.g. PSZ2G023.17+86.71
        force:    If True, download even if directory already exists

    Returns:
        Tuple ('downloaded'|'skipped'|'failed', n_fits_files)
    """
    dest = FITS_ROOT / slug

    # Skip if already complete, unless forced
    if not force:
        is_complete, existing_count, missing = check_complete_download(dest, slug)
        if is_complete:
            print(f"✓ {slug} (complete with {existing_count} FITS files)")
            return ('skipped', existing_count)
        elif existing_count > 0:
            print(f"⚠️  {slug} (incomplete: {existing_count} files, "
                  f"missing {len(missing)}) — re-downloading")

    # Fetch cluster detail page
    r = requests.get(page_url, timeout=30)
    if r.status_code != 200:
        print(f"✖️  {slug}: could not fetch {page_url}")
        return ('failed', 0)

    soup  = BeautifulSoup(r.text, "html.parser")
    tar_a = soup.find(
        "a", href=lambda h: h and (h.endswith(".tar") or h.endswith(".tar.gz"))
    )
    if not tar_a:
        print(f"⚠️  {slug}: no tarball link on {page_url}")
        return ('failed', 0)

    tar_url = urljoin(page_url, tar_a["href"])
    out_tar = OUT_ROOT / f"{slug}.tar.gz"

    print(f"⏬  {slug} downloading...")

    # Download tarball
    try:
        tr = requests.get(tar_url, timeout=600)
        tr.raise_for_status()
    except Exception as e:
        print(f"❌ {slug}: download failed: {e}")
        return ('failed', 0)

    out_tar.write_bytes(tr.content)

    # Remove stale directory before extracting fresh copy
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Extract tarball
    try:
        with tarfile.open(out_tar, "r:gz") as tf:
            tf.extractall(dest)
    except Exception as e:
        print(f"❌ {slug}: extraction failed: {e}")
        out_tar.unlink()
        return ('failed', 0)

    # Report extracted files and clean up
    fits_files = sorted(f for f in os.listdir(dest) if f.endswith('.fits'))
    print(f"   → Extracted {len(fits_files)} FITS files: {', '.join(fits_files)}")
    out_tar.unlink()

    return ('downloaded', len(fits_files))


def main():
    parser = argparse.ArgumentParser(
        description="Download PSZ2 cluster FITS files with completeness checking"
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-download all sources, even if already complete'
    )
    parser.add_argument(
        '--check-only', action='store_true',
        help='Report incomplete downloads without downloading anything'
    )
    parser.add_argument(
        '--csv-only', action='store_true',
        help='Fetch cluster metadata from LOFAR surveys and write cluster_source_data.csv without downloading FITS files'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PSZ2 CLUSTER FITS DOWNLOADER")
    print("=" * 60)
    if args.force:
        print("⚠️  FORCE MODE: re-downloading all sources")
    if args.check_only:
        print("ℹ️  CHECK-ONLY MODE: reporting incomplete sources only")
    print()

    # Fetch and parse the main cluster table
    print("Fetching cluster list from LOFAR surveys...")
    r = requests.get(INDEX, timeout=30)
    r.raise_for_status()
    soup  = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    rows  = table.find_all("tr")[1:]  # skip header row

    cluster_pages = []
    cluster_info  = {}

    for tr in rows:
        tds = tr.find_all("td")
        a   = tr.find("a", href=lambda h: h and "clusters/" in h)
        if not a:
            continue

        slug = os.path.basename(a["href"]).replace(".html", "")
        cluster_pages.append((urljoin(BASE, a["href"]), slug))

        # Extract cluster metadata from table columns
        def safe_float(td):
            try:
                return float(td.text.strip())
            except (ValueError, AttributeError):
                return None

        cluster_info[slug] = {
            'z':              safe_float(tds[4]),
            'M500':           safe_float(tds[5]),
            'M500_err':       safe_float(tds[6]),
            'r500':           safe_float(tds[7]),
            'r500_err':       safe_float(tds[8]),
            'classification': tds[11].text.strip() if len(tds) > 11 else "",
        }

    print(f"Found {len(cluster_pages)} cluster pages\n")

    # CSV-only mode: write metadata and exit
    if args.csv_only:
        metadata_csv = OUT_ROOT / "cluster_source_data.csv"
        with open(metadata_csv, "w", newline="") as fd:
            fieldnames = ["slug", "z", "M500", "M500_err", "r500", "r500_err", "classification"]
            w = csv.DictWriter(fd, fieldnames=fieldnames)
            w.writeheader()
            for slug, info in cluster_info.items():
                w.writerow({"slug": slug, **info})
        print(f"Wrote {len(cluster_info)} rows to {metadata_csv}")
        return

    # Check-only mode: report and exit
    if args.check_only:
        print("=" * 60)
        incomplete = []
        for _, slug in cluster_pages:
            dest = FITS_ROOT / slug
            is_complete, count, missing = check_complete_download(dest, slug)
            if not is_complete:
                incomplete.append((slug, count, missing))
                status = "NOT DOWNLOADED" if count == 0 else f"INCOMPLETE ({count} files)"
                print(f"✗ {slug}: {status}")
        print(f"\n{len(incomplete)} incomplete out of {len(cluster_pages)} total")
        return

    # Download all sources
    stats = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'total_fits': 0}

    MAX_WORKERS = 8  # number of simultaneous downloads

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all downloads at once
        futures = {
            executor.submit(download_and_extract, page_url, slug, args.force): slug
            for page_url, slug in cluster_pages
        }
        # Process results as they complete (not in submission order)
        for i, future in enumerate(as_completed(futures), 1):
            slug = futures[future]
            if i % 50 == 0:
                print(f"\n--- Progress: {i}/{len(cluster_pages)} ---\n")
            try:
                status, count = future.result()
                stats[status] += 1
                if status == 'downloaded':
                    stats['total_fits'] += count
            except Exception as e:
                print(f"❌ {slug}: unexpected error: {e}")
                stats['failed'] += 1

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print(f"  ✅ Downloaded:  {stats['downloaded']} ({stats['total_fits']} FITS files)")
        print(f"  ✓  Skipped:     {stats['skipped']}")
        print(f"  ❌ Failed:      {stats['failed']}")
        print(f"  📊 Total:       {len(cluster_pages)}")
        print("=" * 60)

        # Write metadata CSV (cluster_source_data.csv is the canonical name used by other scripts)
        metadata_csv = OUT_ROOT / "cluster_source_data.csv"
        with open(metadata_csv, "w", newline="") as fd:
            fieldnames = ["slug", "z", "M500", "M500_err", "r500", "r500_err", "classification"]
            w = csv.DictWriter(fd, fieldnames=fieldnames)
            w.writeheader()
            for slug, info in cluster_info.items():
                w.writerow({"slug": slug, **info})

        print(f"\n🗒️  Wrote {len(cluster_info)} rows to {metadata_csv}")


if __name__ == "__main__":
    main()
    print("\nAll done!")