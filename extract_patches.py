"""
extract_patches.py
--------------------
Turns large, full-resolution Sentinel-2 tiles (downloaded via
download_maharashtra_sentinel2.py) into many small 64x64 training patches,
saved in the same folder format data_pipeline.py already expects.

WHY RANDOM SAMPLING, NOT EVERY POSSIBLE PATCH: a single 10,980 x 10,980
tile could be sliced into ~29,000 non-overlapping 64x64 patches. Across
20-30 tiles, that's 600,000+ tiny files - excessive, slow to process, and
mostly redundant (neighboring patches look nearly identical). This script
instead RANDOMLY samples a configurable number of patches per tile,
spread across the whole tile, which gives you genuine geographic variety
without the bloat.

BASIC QUALITY FILTERING: real satellite tiles have clouds and blank
edge/nodata regions. Two simple, honest filters are applied:
  - Skip patches that are mostly very bright (likely dense cloud)
  - Skip patches that are mostly black/zero (nodata edge of the tile)
This is a basic brightness-based filter, not a proper cloud-detection
model - good enough for keeping obviously-bad patches out of training,
not a substitute for the SCL (Scene Classification) band Sentinel-2
provides for rigorous cloud masking.

Usage:
    python extract_patches.py --input_dir maharashtra_sentinel2 --out_dir data/raw/EuroSAT_RGB/MaharashtraSentinel2 --patches_per_tile 300
"""

import os
import glob
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2


def is_low_quality(patch, cloud_threshold=230, nodata_threshold=15):
    """Returns True if the patch looks like solid cloud or nodata/black edge."""
    mean_brightness = patch.mean()
    if mean_brightness > cloud_threshold:
        return True  # likely dense cloud
    if mean_brightness < nodata_threshold:
        return True  # likely nodata/black edge
    return False


def extract_from_tile(tif_path, out_dir, patches_per_tile, patch_size=64, seed=None):
    """Returns (saved, attempts, error) - error is None on success, or a
    message string if the tile couldn't be read (corrupted/incomplete
    download), so the caller can skip it without losing progress on
    every other tile."""
    rng = np.random.default_rng(seed)
    saved = 0
    attempts = 0
    max_attempts = patches_per_tile * 5  # avoid an infinite loop on a very bad tile

    try:
        with rasterio.open(tif_path) as src:
            width, height = src.width, src.height
            tile_name = os.path.splitext(os.path.basename(tif_path))[0]

            while saved < patches_per_tile and attempts < max_attempts:
                attempts += 1
                row_off = rng.integers(0, max(1, height - patch_size))
                col_off = rng.integers(0, max(1, width - patch_size))
                window = Window(col_off, row_off, patch_size, patch_size)

                try:
                    data = src.read([1, 2, 3], window=window)
                except Exception:
                    continue  # this specific region is unreadable, try elsewhere

                if data.shape[1] != patch_size or data.shape[2] != patch_size:
                    continue  # ran off the edge of the tile

                patch = np.transpose(data, (1, 2, 0))  # (H, W, 3)

                if is_low_quality(patch):
                    continue

                out_name = f"{tile_name}_p{saved:04d}.jpg"
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                saved += 1

    except Exception as e:
        # The file itself couldn't be opened, or failed severely enough that
        # continuing is pointless - report it so the caller can skip this
        # tile and move on rather than crashing the whole run.
        return saved, attempts, str(e)

    return saved, attempts, None


def main():
    parser = argparse.ArgumentParser(description="Extract training patches from large Sentinel-2 tiles")
    parser.add_argument("--input_dir", required=True, help="Folder containing downloaded .tif tiles")
    parser.add_argument("--out_dir", required=True, help="Where to save extracted patches (a new class folder)")
    parser.add_argument("--patches_per_tile", type=int, default=300)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tif_files = sorted(glob.glob(os.path.join(args.input_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {args.input_dir}")
        return

    print(f"Found {len(tif_files)} tiles. Extracting up to {args.patches_per_tile} patches each...")
    total_saved = 0
    failed_tiles = []
    for tif_path in tif_files:
        saved, attempts, error = extract_from_tile(
            tif_path, args.out_dir, args.patches_per_tile, args.patch_size, seed=args.seed
        )
        if error:
            print(f"  {os.path.basename(tif_path)}: FAILED to read ({error}) - "
                  f"likely a corrupted or incomplete download. Skipping this tile.")
            failed_tiles.append(os.path.basename(tif_path))
            continue
        total_saved += saved
        print(f"  {os.path.basename(tif_path)}: saved {saved} patches ({attempts} attempts, "
              f"{attempts - saved} rejected as cloud/nodata)")

    if failed_tiles:
        print(f"\n{len(failed_tiles)} tile(s) could not be read and were skipped:")
        for name in failed_tiles:
            print(f"  - {name}")
        print("Consider re-downloading these specific files if you want their data too.\n")

    print(f"\nDone. Total patches saved: {total_saved}")
    print(f"Saved to: {args.out_dir}")
    print("This folder is now ready to be included by data_pipeline.py - point")
    print("--data_dir at its PARENT folder (the one containing all class folders),")
    print("e.g.: python src/data_pipeline.py --data_dir data/raw/EuroSAT_RGB --out_dir data/processed")


if __name__ == "__main__":
    main()
