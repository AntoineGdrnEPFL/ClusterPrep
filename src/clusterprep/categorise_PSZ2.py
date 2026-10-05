#!/usr/bin/env python3
"""
Organize PSZ2 FITS files into classification folders based on cluster_source_data.csv

This script:
1. Reads cluster_source_data.csv with classification information
2. Copies files from fits/<source>/*.fits to classified/<version>/<class>/*.fits
3. Maps classifications to class folders:
   - NDE (No Diffuse Emission) → class 51 → folder "NDE"
   - RH (Radio Halo) → class 52 → folder "RH" (also in DE)
   - RR (Radio Relic) → class 53 → folder "RR" (also in DE)
   - RH or RR (Diffuse Emission) → class 50 → folder "DE"
   - cRH (candidate RH) → class 54 → folder "cRH" (also in cDE)
   - cRR (candidate RR) → class 55 → folder "cRR" (also in cDE)
   - cDE (candidate DE) → class 56 → folder "cDE"
   - Other classes mapped accordingly

Usage:
    python classify_psz2_files.py [--symlink]  # use --symlink for symlinks instead of copying

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

import os
import sys
import csv
import shutil
from pathlib import Path

# Paths
BASE_DIR = "/Users/antoinegdrn/PH_Project_I/scratch/data/PSZ2"
FITS_DIR = os.path.join(BASE_DIR, "fits")
CLASSIFIED_DIR = os.path.join(BASE_DIR, "classified")
SOURCEDATA_CSV = os.path.join(BASE_DIR, "cluster_source_data.csv")

# Classification mapping
# Maps classification strings to (class_tag, folder_name)
CLASS_MAP = {
    'NDE': (51, 'NDE'),           # No Diffuse Emission
    'U': (57, 'U'),               # Uncertain
    'unclassified': (58, 'unclassified'),
}

# Special handling for compound classifications
def parse_classification(classification_str):
    """
    Parse classification string and return list of (class_tag, folder_name) tuples.
    Files can be placed in multiple folders.
    
    Rules:
    - If classification contains 'RH' (not cRH):
        → class 52 (RH) folder "RH"
        → class 50 (DE) folder "DE"
    - If classification contains 'RR' (not cRR):
        → class 53 (RR) folder "RR"
        → class 50 (DE) folder "DE"
    - If classification contains 'cRH':
        → class 54 (cRH) folder "cRH"
        → class 56 (cDE) folder "cDE"
    - If classification contains 'cRR':
        → class 55 (cRR) folder "cRR"
        → class 56 (cDE) folder "cDE"
    - If classification contains 'cDE' (standalone):
        → class 56 (cDE) folder "cDE"
    - If classification is 'NDE' → class 51 (NDE)
    - If classification is 'U' → class 57 (U)
    - If classification is 'N/A' or empty → class 58 (unclassified)
    
    Returns:
        List of (class_tag, folder_name) tuples
    """
    if not classification_str or classification_str.strip() == '':
        return [(58, 'unclassified')]
    
    # Normalize: remove whitespace, convert to uppercase for checking
    cls = classification_str.strip()
    cls_upper = cls.upper()
    
    folders = []
    
    # Check for RH (but not cRH)
    if 'RH' in cls_upper:
        # Check if it's cRH or standalone RH
        if 'CRH' in cls_upper:
            # It's candidate RH
            folders.append((54, 'cRH'))
            folders.append((56, 'cDE'))
        else:
            # It's confirmed RH
            folders.append((52, 'RH'))
            folders.append((50, 'DE'))
    
    # Check for RR (but not cRR)
    if 'RR' in cls_upper:
        # Check if it's cRR or standalone RR
        if 'CRR' in cls_upper:
            # It's candidate RR
            folders.append((55, 'cRR'))
            folders.append((56, 'cDE'))
        else:
            # It's confirmed RR
            folders.append((53, 'RR'))
            # Only add DE if not already added by RH
            if (50, 'DE') not in folders:
                folders.append((50, 'DE'))
    
    # Check for standalone cDE (not already handled by cRH/cRR)
    if 'CDE' in cls_upper and not folders:
        folders.append((56, 'cDE'))
    
    # Check for NDE
    if cls_upper == 'NDE':
        if not folders:  # Only if no other classification found
            folders.append((51, 'NDE'))
    
    # Check for uncertain
    if cls_upper == 'U':
        if not folders:  # Only if no other classification found
            folders.append((57, 'U'))
    
    # Default: unclassified
    if not folders:
        folders.append((58, 'unclassified'))
    
    return folders


def get_version_dirs():
    """
    Find all version directories in FITS_DIR.
    Returns list of version folder names (e.g., ['RAW', 'T50kpc', 'T100kpc', ...])
    """
    versions = []
    for source_dir in os.listdir(FITS_DIR):
        source_path = os.path.join(FITS_DIR, source_dir)
        if not os.path.isdir(source_path):
            continue
        # Check what FITS files exist
        fits_files = [f for f in os.listdir(source_path) if f.endswith('.fits')]
        if not fits_files:
            continue
        # Infer versions from filenames
        # Format: PSZ2G023.17+86.71.fits (RAW) or PSZ2G023.17+86.71T50kpc.fits
        for fname in fits_files:
            base = fname.replace('.fits', '')
            if base == source_dir:
                # This is RAW
                if 'RAW' not in versions:
                    versions.append('RAW')
            elif base.startswith(source_dir):
                # Extract version suffix
                suffix = base.replace(source_dir, '')
                if suffix and suffix not in versions:
                    versions.append(suffix)
    
    return sorted(set(versions))


def file_needs_update(src_file, dst_file, use_symlink=False):
    """
    Check if destination file needs to be created/updated.
    
    Args:
        src_file: Source file path
        dst_file: Destination file path
        use_symlink: If True, check symlink validity
    
    Returns:
        True if file needs to be created/updated, False otherwise
    """
    # Destination doesn't exist
    if not os.path.exists(dst_file):
        return True
    
    if use_symlink:
        # Check if it's a valid symlink pointing to the right place
        if not os.path.islink(dst_file):
            return True  # Not a symlink, recreate it
        
        # Check if symlink points to correct target
        link_target = os.readlink(dst_file)
        expected_target = os.path.relpath(src_file, os.path.dirname(dst_file))
        if link_target != expected_target:
            return True  # Points to wrong place, recreate
        
        # Check if target still exists
        if not os.path.exists(os.path.join(os.path.dirname(dst_file), link_target)):
            return True  # Broken symlink, recreate
        
        return False  # Valid symlink, skip
    else:
        # For copies, check if file size and modification time match
        try:
            src_stat = os.stat(src_file)
            dst_stat = os.stat(dst_file)
            
            # Check size first (fast)
            if src_stat.st_size != dst_stat.st_size:
                return True
            
            # Check modification time (source newer than destination)
            if src_stat.st_mtime > dst_stat.st_mtime:
                return True
            
            return False  # Files match, skip
        except (OSError, IOError):
            return True  # Can't stat, recreate


def classify_files(use_symlink=False):
    """
    Main classification function.
    
    Args:
        use_symlink: If True, create symlinks; if False (default), copy files
    """
    # Read sourcedata
    print(f"Reading sourcedata from {SOURCEDATA_CSV}...")
    classifications = {}
    with open(SOURCEDATA_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row['slug']
            classification = row.get('classification', '')
            folders = parse_classification(classification)
            classifications[slug] = {
                'classification': classification,
                'folders': folders  # List of (class_tag, folder_name) tuples
            }
    
    print(f"Loaded {len(classifications)} cluster classifications")
    
    # Count classifications (a source can be in multiple folders)
    from collections import Counter
    folder_counts = Counter()
    for info in classifications.values():
        for _, folder_name in info['folders']:
            folder_counts[folder_name] += 1
    
    print("\nClassification distribution:")
    for folder, count in sorted(folder_counts.items()):
        print(f"  {folder}: {count}")
    
    # Get all version directories
    print(f"\nScanning FITS directories in {FITS_DIR}...")
    source_dirs = [d for d in os.listdir(FITS_DIR) 
                   if os.path.isdir(os.path.join(FITS_DIR, d))]
    print(f"Found {len(source_dirs)} source directories")
    
    # Process each source
    processed_count = 0
    skipped_count = 0
    skip_exists_count = 0
    skip_nosource_count = 0
    
    for source in sorted(source_dirs):
        if source not in classifications:
            print(f"⚠️  {source}: not in sourcedata, skipping")
            skip_nosource_count += 1
            continue
        
        info = classifications[source]
        source_path = os.path.join(FITS_DIR, source)
        
        # Get all FITS files for this source
        fits_files = [f for f in os.listdir(source_path) if f.endswith('.fits')]
        
        if not fits_files:
            print(f"⚠️  {source}: no FITS files found")
            skip_nosource_count += 1
            continue
        
        # Determine versions from filenames
        for fname in fits_files:
            # Extract version from filename
            # Formats: PSZ2G023.17+86.71.fits or PSZ2G023.17+86.71T50kpc.fits
            base = fname.replace('.fits', '')
            
            if base == source:
                version = 'RAW'
            elif base.startswith(source):
                version = base.replace(source, '')
            else:
                print(f"⚠️  {source}/{fname}: unexpected filename format")
                continue
            
            # Source file path
            src_file = os.path.join(source_path, fname)
            
            # Copy/link to ALL applicable folders for this classification
            for class_tag, folder_name in info['folders']:
                # Create classified directory structure
                class_dir = os.path.join(CLASSIFIED_DIR, version, folder_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Destination path
                dst_file = os.path.join(class_dir, fname)
                
                # Check if file needs to be created/updated (rsync-like behavior)
                if not file_needs_update(src_file, dst_file, use_symlink):
                    skip_exists_count += 1
                    continue
                
                # Remove old file if it exists and needs updating
                if os.path.exists(dst_file):
                    try:
                        if os.path.islink(dst_file):
                            os.unlink(dst_file)
                        else:
                            os.remove(dst_file)
                    except Exception as e:
                        print(f"⚠️  Failed to remove old {dst_file}: {e}")
                        continue
                
                # Create symlink or copy
                try:
                    if use_symlink:
                        # Create relative symlink
                        rel_path = os.path.relpath(src_file, class_dir)
                        os.symlink(rel_path, dst_file)
                    else:
                        shutil.copy2(src_file, dst_file)
                    processed_count += 1
                except Exception as e:
                    print(f"✖️  Failed to {'link' if use_symlink else 'copy'} {fname} to {folder_name}: {e}")
        
        # Print progress every 50 sources
        if (source_dirs.index(source) + 1) % 50 == 0:
            print(f"  Processed {source_dirs.index(source) + 1}/{len(source_dirs)} sources...")
    
    print(f"\n{'='*60}")
    print(f"Classification complete:")
    print(f"  ✅ {'Linked' if use_symlink else 'Copied'}: {processed_count} files")
    print(f"  ⏭️  Skipped (already up-to-date): {skip_exists_count}")
    print(f"  ⏭️  Skipped (no source/sourcedata): {skip_nosource_count}")
    print(f"  📊 Total sources processed: {len(source_dirs)}")
    print(f"{'='*60}")
    print(f"\nClassified files are in: {CLASSIFIED_DIR}")
    print("Directory structure:")
    for version in sorted(os.listdir(CLASSIFIED_DIR)):
        version_path = os.path.join(CLASSIFIED_DIR, version)
        if os.path.isdir(version_path):
            print(f"  {version}/")
            for cls_folder in sorted(os.listdir(version_path)):
                cls_path = os.path.join(version_path, cls_folder)
                if os.path.isdir(cls_path):
                    n_files = len([f for f in os.listdir(cls_path) if f.endswith('.fits')])
                    print(f"    {cls_folder}/ ({n_files} files)")


if __name__ == "__main__":
    use_symlink = '--symlink' in sys.argv
    
    if not os.path.exists(SOURCEDATA_CSV):
        print(f"❌ Sourcedata file not found: {SOURCEDATA_CSV}")
        print("   Run download_psz2_rsync.py first to generate sourcedata.")
        sys.exit(1)
    
    if not os.path.isdir(FITS_DIR):
        print(f"❌ FITS directory not found: {FITS_DIR}")
        print("   Run download_psz2_rsync.py first to download FITS files.")
        sys.exit(1)
    
    print("PSZ2 File Classification")
    print("="*60)
    print(f"Mode: {'SYMLINK' if use_symlink else 'COPY'}")
    print(f"Source: {FITS_DIR}")
    print(f"Destination: {CLASSIFIED_DIR}")
    print("="*60 + "\n")
    
    classify_files(use_symlink=use_symlink)