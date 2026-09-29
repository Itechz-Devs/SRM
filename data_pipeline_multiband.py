"""
data_pipeline_multiband.py
------------------------------
Generalizes data_pipeline_upsample.py to work with ANY number of bands,
from ANY of the three new download scripts:

    download_sentinel2_infrared.py  -> --bands B04,B03,B02,B08  (4 channels)
    download_sentinel1_sar.py        -> --bands vv,vh            (2 channels)
    download_landsat_thermal.py       -> --bands lwir11           (1 channel)

HOW MULTI-FILE TILES WORK: each download script saves one file PER BAND
per tile (e.g. "43QCB_2026-06-01_B04.tif", "...{_B03.tif", etc.) rather
than one merged multi-band file. This script finds matching sets of band
files for the same tile+date and reads synchronized windows from each,
stacking them into one multi-channel patch.

DATA-TYPE-AWARE NORMALIZATION: optical bands (Sentinel-2, Landsat) and
SAR backscatter have very different native value ranges and statistics -
using the same 0-255 percentile stretch for both would be wrong. Use
--data_type to select the right handling:
    optical  (default) - percentile stretch, same as the RGB pipeline
    sar                - SAR backscatter is often in a different range
                         and can include negative/log-scaled values;
                         uses a wider percentile stretch tuned for that
    thermal            - single-band, percentile stretch, no cloud-based
                         quality filter (thermal "brightness" doesn't
                         mean the same thing as optical cloud brightness)

Usage:
    python data_pipeline_multiband.py --input_dir maharashtra_infrared --out_dir data/processed_x4_ir --bands B04,B03,B02,B08 --scale 4 --lr_size 16 --data_type optical
    python data_pipeline_multiband.py --input_dir maharashtra_sar --out_dir data/processed_x4_sar --bands vv,vh --scale 4 --lr_size 16 --data_type sar
    python data_pipeline_multiband.py --input_dir maharashtra_thermal --out_dir data/processed_x4_thermal --bands lwir11 --scale 4 --lr_size 16 --data_type thermal
"""

import os
import re
import glob
import argparse
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window


def find_tile_groups(input_dir, bands):
    """
    Groups files by tile+date, matching the naming convention each download
    script uses: "{tile_id}_{date}_{band}.tif". Returns a dict:
        {"43QCB_2026-06-01": {"B04": "path/to/file.tif", "B03": "...", ...}}
    Only tiles that have ALL requested bands present are included - a tile
    missing even one band can't be used (there'd be a hole in the stack).
    """
    groups = {}
    for band in bands:
        pattern = os.path.join(input_dir, f"*_{band}.tif")
        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            key = fname[: -(len(band) + 5)]  # strip "_{band}.tif"
            groups.setdefault(key, {})[band] = path

    complete_groups = {k: v for k, v in groups.items() if all(b in v for b in bands)}
    incomplete = set(groups.keys()) - set(complete_groups.keys())
    if incomplete:
        print(f"Note: {len(incomplete)} tile(s) skipped - missing one or more requested bands "
              f"(likely a download that didn't finish, or a band unavailable for that scene).")
    return complete_groups


def percentile_stretch(band, low=2, high=98):
    p_low, p_high = np.percentile(band, (low, high))
    if p_high - p_low < 1e-6:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band.astype(np.float32) - p_low) / (p_high - p_low), 0, 1)


def is_low_quality_optical(patch_uint8, cloud_threshold=230, nodata_threshold=15):
    mean_brightness = patch_uint8.mean()
    return mean_brightness > cloud_threshold or mean_brightness < nodata_threshold


def sample_multiband_patches(band_paths, patches_per_tile, hr_size, data_type, seed=None):
    """
    band_paths: dict of {band_name: file_path} for ONE tile, all bands.
    Returns a list of (hr_size, hr_size, num_bands) uint8 patches (already
    normalized per-band to 0-255 for consistent storage regardless of
    original data type).
    """
    rng = np.random.default_rng(seed)
    band_names = list(band_paths.keys())
    patches = []
    attempts = 0
    max_attempts = patches_per_tile * 5

    try:
        srcs = {b: rasterio.open(p) for b, p in band_paths.items()}
        # All bands for one tile should share the same footprint/size - use
        # the first band's dimensions as reference.
        ref = srcs[band_names[0]]
        width, height = ref.width, ref.height

        while len(patches) < patches_per_tile and attempts < max_attempts:
            attempts += 1
            row_off = rng.integers(0, max(1, height - hr_size))
            col_off = rng.integers(0, max(1, width - hr_size))
            window = Window(col_off, row_off, hr_size, hr_size)

            try:
                band_arrays = []
                ok = True
                for b in band_names:
                    arr = srcs[b].read(1, window=window)
                    if arr.shape != (hr_size, hr_size):
                        ok = False
                        break
                    band_arrays.append(arr)
                if not ok:
                    continue
            except Exception:
                continue

            # Normalize each band independently to 0-255 for consistent storage.
            stretch_low, stretch_high = (2, 98) if data_type != "sar" else (1, 99)
            normalized = [
                (percentile_stretch(a, stretch_low, stretch_high) * 255).astype(np.uint8)
                for a in band_arrays
            ]
            patch = np.stack(normalized, axis=-1)  # (H, W, num_bands)

            if data_type == "optical" and is_low_quality_optical(patch[..., :3] if patch.shape[-1] >= 3 else patch):
                continue
            # SAR and thermal: skip the cloud-brightness heuristic (doesn't
            # apply to that kind of data) - just check for all-zero/no-data.
            if data_type != "optical" and patch.mean() < 2:
                continue

            patches.append(patch)

    finally:
        for s in srcs.values():
            s.close()

    return patches, attempts


def make_true_lr_hr_pairs(hr_images_float, scale, lr_size):
    n, hr_h, hr_w, c = hr_images_float.shape
    lr_images = np.zeros((n, lr_size, lr_size, c), dtype=np.float32)
    for i in range(n):
        resized = cv2.resize(hr_images_float[i], (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        # cv2.resize drops the channel dimension entirely for single-channel
        # (e.g. thermal) images, returning (H, W) instead of (H, W, 1) - put
        # it back so every array here has a consistent (H, W, C) shape.
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        lr_images[i] = resized
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
    parser = argparse.ArgumentParser(description="Build multi-band (IR/SAR/thermal) LR/HR training pairs")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bands", required=True, help="Comma-separated band names, e.g. B04,B03,B02,B08 or vv,vh or lwir11")
    parser.add_argument("--data_type", choices=["optical", "sar", "thermal"], default="optical")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--lr_size", type=int, default=16)
    parser.add_argument("--patches_per_tile", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bands = [b.strip() for b in args.bands.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    hr_size = args.lr_size * args.scale

    print(f"Bands: {bands} ({len(bands)} channel(s)) | data_type={args.data_type}")
    print(f"Building x{args.scale} pairs: LR {args.lr_size}x{args.lr_size} -> HR {hr_size}x{hr_size}\n")

    tile_groups = find_tile_groups(args.input_dir, bands)
    if not tile_groups:
        print(f"No complete tile groups found in {args.input_dir} for bands {bands}.")
        print("Check that the download script finished and filenames match the expected pattern.")
        return

    print(f"Found {len(tile_groups)} complete tile(s).\n")

    all_patches = []
    for tile_key, band_paths in sorted(tile_groups.items()):
        patches, attempts = sample_multiband_patches(band_paths, args.patches_per_tile, hr_size, args.data_type, seed=args.seed)
        all_patches.extend(patches)
        print(f"  {tile_key}: {len(patches)} patches ({attempts} attempts)")

    if not all_patches:
        print("No usable patches extracted.")
        return

    print(f"\nTotal patches: {len(all_patches)}")
    hr_uint8 = np.array(all_patches, dtype=np.uint8)
    hr_float = hr_uint8.astype(np.float32) / 255.0

    print("Generating true LR/HR pairs...")
    lr_float, hr_float = make_true_lr_hr_pairs(hr_float, args.scale, args.lr_size)

    print("Splitting and saving...")
    (lr_train, hr_train), (lr_val, hr_val), (lr_test, hr_test) = train_val_test_split(lr_float, hr_float, seed=args.seed)

    np.save(os.path.join(args.out_dir, "lr_train.npy"), lr_train)
    np.save(os.path.join(args.out_dir, "hr_train.npy"), hr_train)
    np.save(os.path.join(args.out_dir, "lr_val.npy"), lr_val)
    np.save(os.path.join(args.out_dir, "hr_val.npy"), hr_val)
    np.save(os.path.join(args.out_dir, "lr_test.npy"), lr_test)
    np.save(os.path.join(args.out_dir, "hr_test.npy"), hr_test)

    with open(os.path.join(args.out_dir, "scale.txt"), "w") as f:
        f.write(str(args.scale))
    with open(os.path.join(args.out_dir, "bands.txt"), "w") as f:
        f.write(",".join(bands))
    with open(os.path.join(args.out_dir, "data_type.txt"), "w") as f:
        f.write(args.data_type)

    print(f"\nDone. Saved to: {args.out_dir}")
    print(f"  Train: {lr_train.shape[0]} pairs | {lr_train.shape[-1]} channel(s)")
    print(f"\nNext step:")
    print(f"  python src/train.py --data_dir {args.out_dir} --model espcn --scale {args.scale} "
          f"--epochs 30 --model_suffix {args.data_type if args.data_type != 'optical' else 'ir'}")


if __name__ == "__main__":
    main()
