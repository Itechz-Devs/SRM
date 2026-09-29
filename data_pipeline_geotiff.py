"""
data_pipeline_geotiff.py
--------------------------
A GeoTIFF-native alternative to data_pipeline.py + extract_patches.py -
reads directly from large downloaded Sentinel-2 tiles (.tif) and produces
the same LR/HR .npy training files train.py already expects, WITHOUT ever
converting anything to JPG first.

This does everything in one pass, using rasterio's windowed reading so
the full multi-hundred-MB tiles are never fully loaded into memory:
  1. Randomly samples patches from each tile (rejecting obvious cloud/
     nodata regions, same filter as extract_patches.py)
  2. Normalizes to [0, 1] float32
  3. Generates the LR input by downsampling then upsampling back
     (same degradation approach as data_pipeline.py, so a model trained
     on this data behaves consistently with one trained on EuroSAT)
  4. Splits into train/val/test and saves .npy files - SAME format and
     filenames as data_pipeline.py's output, so train.py works with
     these files completely unchanged.

Usage:
    python data_pipeline_geotiff.py --input_dir maharashtra_sentinel2 --out_dir data/processed_geotiff --patches_per_tile 300
"""

import os
import glob
import argparse
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window


def is_low_quality(patch, cloud_threshold=230, nodata_threshold=15):
    """Same filter as extract_patches.py: rejects likely-cloud or likely-nodata patches."""
    mean_brightness = patch.mean()
    return mean_brightness > cloud_threshold or mean_brightness < nodata_threshold


def sample_patches_from_tile(tif_path, patches_per_tile, patch_size, seed=None):
    """
    Randomly samples clean patches from one GeoTIFF tile using windowed reads.
    Returns (patches, attempts, error) - error is None on success, or a
    message string if the tile couldn't be read at all (e.g. corrupted or
    incompletely downloaded), so the caller can skip it and keep going
    rather than losing progress on every other tile.
    """
    rng = np.random.default_rng(seed)
    patches = []
    attempts = 0
    max_attempts = patches_per_tile * 5

    try:
        with rasterio.open(tif_path) as src:
            width, height = src.width, src.height
            band_count = src.count

            while len(patches) < patches_per_tile and attempts < max_attempts:
                attempts += 1
                row_off = rng.integers(0, max(1, height - patch_size))
                col_off = rng.integers(0, max(1, width - patch_size))
                window = Window(col_off, row_off, patch_size, patch_size)

                try:
                    if band_count >= 3:
                        data = src.read([1, 2, 3], window=window)
                    else:
                        single = src.read(1, window=window)
                        data = np.stack([single, single, single])
                except Exception:
                    # This specific windowed region is unreadable (a common
                    # symptom of a partially-corrupted/incomplete download,
                    # even if the file opened fine overall) - skip just this
                    # one attempt and keep sampling elsewhere in the tile.
                    continue

                if data.shape[1] != patch_size or data.shape[2] != patch_size:
                    continue

                patch = np.transpose(data, (1, 2, 0))
                if is_low_quality(patch):
                    continue

                patches.append(patch)

    except Exception as e:
        # The file itself couldn't even be opened, or failed severely enough
        # that continuing is pointless - report it and let the caller move
        # on to the next tile instead of crashing the whole run.
        return patches, attempts, str(e)

    return patches, attempts, None


def make_lr_hr_pairs(hr_images_float, scale=4):
    """Identical approach to data_pipeline.py: downsample then upsample back."""
    n, h, w, c = hr_images_float.shape
    lr_images = np.zeros_like(hr_images_float)
    for i in range(n):
        hr = hr_images_float[i]
        small = cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        lr_images[i] = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return lr_images, hr_images_float


def train_val_test_split(lr, hr, val_frac=0.1, test_frac=0.1, seed=42):
    n = lr.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    test_n = int(n * test_frac)
    val_n = int(n * val_frac)
    test_idx = idx[:test_n]
    val_idx = idx[test_n:test_n + val_n]
    train_idx = idx[test_n + val_n:]
    return (
        (lr[train_idx], hr[train_idx]),
        (lr[val_idx], hr[val_idx]),
        (lr[test_idx], hr[test_idx]),
    )


def main():
    parser = argparse.ArgumentParser(description="Build LR/HR training data directly from GeoTIFF tiles")
    parser.add_argument("--input_dir", required=True, help="Folder containing downloaded .tif tiles")
    parser.add_argument("--out_dir", default="data/processed_geotiff")
    parser.add_argument("--patches_per_tile", type=int, default=300)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tif_files = sorted(glob.glob(os.path.join(args.input_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {args.input_dir}")
        return

    print(f"Found {len(tif_files)} tiles. Sampling up to {args.patches_per_tile} patches each...")
    all_patches = []
    failed_tiles = []
    for tif_path in tif_files:
        patches, attempts, error = sample_patches_from_tile(
            tif_path, args.patches_per_tile, args.patch_size, seed=args.seed
        )
        if error:
            print(f"  {os.path.basename(tif_path)}: FAILED to read ({error}) - "
                  f"likely a corrupted or incomplete download. Skipping this tile.")
            failed_tiles.append(os.path.basename(tif_path))
            continue
        all_patches.extend(patches)
        print(f"  {os.path.basename(tif_path)}: {len(patches)} patches ({attempts} attempts)")

    if failed_tiles:
        print(f"\n{len(failed_tiles)} tile(s) could not be read and were skipped:")
        for name in failed_tiles:
            print(f"  - {name}")
        print("Consider re-downloading these specific files if you want their data too -")
        print("everything else that succeeded is still used below.\n")

    if not all_patches:
        print("No usable patches extracted - check your tiles or lower quality thresholds.")
        return

    print(f"\nTotal patches: {len(all_patches)}")
    hr_uint8 = np.array(all_patches, dtype=np.uint8)
    hr_float = hr_uint8.astype(np.float32) / 255.0

    print(f"Generating LR/HR pairs (downsample factor {args.scale})...")
    lr_float, hr_float = make_lr_hr_pairs(hr_float, scale=args.scale)

    print("Splitting into train/val/test and saving...")
    (lr_train, hr_train), (lr_val, hr_val), (lr_test, hr_test) = train_val_test_split(lr_float, hr_float)

    np.save(os.path.join(args.out_dir, "lr_train.npy"), lr_train)
    np.save(os.path.join(args.out_dir, "hr_train.npy"), hr_train)
    np.save(os.path.join(args.out_dir, "lr_val.npy"), lr_val)
    np.save(os.path.join(args.out_dir, "hr_val.npy"), hr_val)
    np.save(os.path.join(args.out_dir, "lr_test.npy"), lr_test)
    np.save(os.path.join(args.out_dir, "hr_test.npy"), hr_test)

    print(f"\nDone. Saved to: {args.out_dir}")
    print(f"  Train: {lr_train.shape[0]} pairs")
    print(f"  Val:   {lr_val.shape[0]} pairs")
    print(f"  Test:  {lr_test.shape[0]} pairs")
    print("\nNext step - train.py works unchanged, just point it at this folder:")
    print(f"  python src/train.py --data_dir {args.out_dir} --model deep_resnet --epochs 30")


if __name__ == "__main__":
    main()
