"""
data_pipeline_upsample.py
----------------------------
Builds TRUE upsampling training pairs for the "espcn" model - unlike
data_pipeline.py / data_pipeline_geotiff.py (which create same-size
refinement pairs), this produces a genuinely SMALLER low-resolution input
and a LARGER high-resolution target, at whatever --scale you choose.

Works from either:
  - A folder of large GeoTIFF tiles (--input_dir, e.g. maharashtra_sentinel2)
  - A folder of already-cropped JPG/PNG images organized in class folders
    (--input_dir pointing at e.g. data/raw/EuroSAT_RGB)

HOW THE PAIRS ARE MADE: for each sample, a patch of size
(hr_size = lr_size * scale) is taken as the HR ground truth, then
downsampled (bicubic) to lr_size for the LR input. The model then learns
to reconstruct the HR image from the smaller LR one - genuine super-
resolution, not refinement of an already-upsampled image.

RESOLUTION MEANING: if your source imagery is native 10m/pixel Sentinel-2,
then a trained model at a given --scale effectively produces output at
(10 / scale) meters-per-pixel equivalent detail, e.g.:
    scale 2 -> ~5m      scale 3 -> ~3.3m
    scale 4 -> ~2.5m    scale 5 -> ~2m

Usage (from GeoTIFF tiles):
    python data_pipeline_upsample.py --input_dir maharashtra_sentinel2 --out_dir data/processed_x4 --scale 4 --lr_size 16 --patches_per_tile 300

Usage (from a JPG class-folder dataset like EuroSAT):
    python data_pipeline_upsample.py --input_dir data/raw/EuroSAT_RGB --out_dir data/processed_x4 --scale 4 --lr_size 16 --mode jpg
"""

import os
import glob
import argparse
import numpy as np
import cv2


def is_low_quality(patch, cloud_threshold=230, nodata_threshold=15):
    mean_brightness = patch.mean()
    return mean_brightness > cloud_threshold or mean_brightness < nodata_threshold


def sample_patches_from_geotiff(tif_path, patches_per_tile, hr_size, seed=None):
    import rasterio
    from rasterio.windows import Window

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
                row_off = rng.integers(0, max(1, height - hr_size))
                col_off = rng.integers(0, max(1, width - hr_size))
                window = Window(col_off, row_off, hr_size, hr_size)

                try:
                    if band_count >= 3:
                        data = src.read([1, 2, 3], window=window)
                    else:
                        single = src.read(1, window=window)
                        data = np.stack([single, single, single])
                except Exception:
                    continue

                if data.shape[1] != hr_size or data.shape[2] != hr_size:
                    continue

                patch = np.transpose(data, (1, 2, 0))
                if is_low_quality(patch):
                    continue
                patches.append(patch)

    except Exception as e:
        return patches, attempts, str(e)

    return patches, attempts, None


def sample_patches_from_jpg_folder(root_dir, hr_size, limit_per_class=None):
    """Loads images from a EuroSAT-style class-folder dataset, resizing each
    to hr_size (these are typically already small, e.g. 64x64)."""
    patches = []
    class_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    for class_name in class_folders:
        class_path = os.path.join(root_dir, class_name)
        file_paths = sorted(glob.glob(os.path.join(class_path, "*.jpg")))
        if limit_per_class:
            file_paths = file_paths[:limit_per_class]
        for fp in file_paths:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (hr_size, hr_size), interpolation=cv2.INTER_CUBIC)
            patches.append(img)
        print(f"  Loaded {len(file_paths)} images from {class_name}")
    return patches


def make_true_lr_hr_pairs(hr_images_float, scale, lr_size):
    """HR stays as-is; LR is a genuinely smaller downsampled version - no
    upsampling back. This is what makes the model actually do super-
    resolution instead of same-size refinement."""
    n = hr_images_float.shape[0]
    hr_size = lr_size * scale
    lr_images = np.zeros((n, lr_size, lr_size, 3), dtype=np.float32)
    for i in range(n):
        lr_images[i] = cv2.resize(hr_images_float[i], (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
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
    parser = argparse.ArgumentParser(description="Build TRUE upsampling LR/HR pairs for the espcn model")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=["geotiff", "jpg"], default="geotiff",
                         help="'geotiff' for a folder of large .tif tiles, 'jpg' for a EuroSAT-style class-folder dataset")
    parser.add_argument("--scale", type=int, default=4, help="Upsampling factor - HR size = lr_size * scale")
    parser.add_argument("--lr_size", type=int, default=16, help="Low-resolution input patch size in pixels")
    parser.add_argument("--patches_per_tile", type=int, default=300, help="(geotiff mode only)")
    parser.add_argument("--limit_per_class", type=int, default=None, help="(jpg mode only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hr_size = args.lr_size * args.scale
    resulting_resolution = 10.0 / args.scale
    print(f"Building x{args.scale} upsampling pairs: LR {args.lr_size}x{args.lr_size} -> HR {hr_size}x{hr_size}")
    print(f"For 10m-native Sentinel-2 input, this scale targets ~{resulting_resolution:.1f}m equivalent resolution.\n")

    all_patches = []

    if args.mode == "geotiff":
        tif_files = sorted(glob.glob(os.path.join(args.input_dir, "*.tif")))
        if not tif_files:
            print(f"No .tif files found in {args.input_dir}")
            return
        print(f"Found {len(tif_files)} tiles.")
        failed_tiles = []
        for tif_path in tif_files:
            patches, attempts, error = sample_patches_from_geotiff(tif_path, args.patches_per_tile, hr_size, seed=args.seed)
            if error:
                print(f"  {os.path.basename(tif_path)}: FAILED ({error}) - skipping")
                failed_tiles.append(os.path.basename(tif_path))
                continue
            all_patches.extend(patches)
            print(f"  {os.path.basename(tif_path)}: {len(patches)} patches ({attempts} attempts)")
        if failed_tiles:
            print(f"\n{len(failed_tiles)} tile(s) skipped due to read errors: {failed_tiles}")

    else:  # jpg mode
        all_patches = sample_patches_from_jpg_folder(args.input_dir, hr_size, args.limit_per_class)

    if not all_patches:
        print("No usable patches - check your input data.")
        return

    print(f"\nTotal HR patches: {len(all_patches)}")
    hr_uint8 = np.array(all_patches, dtype=np.uint8)
    hr_float = hr_uint8.astype(np.float32) / 255.0

    print("Generating true LR/HR pairs (genuine downsampling, no upsample-back)...")
    lr_float, hr_float = make_true_lr_hr_pairs(hr_float, args.scale, args.lr_size)

    print("Splitting into train/val/test and saving...")
    (lr_train, hr_train), (lr_val, hr_val), (lr_test, hr_test) = train_val_test_split(lr_float, hr_float, seed=args.seed)

    np.save(os.path.join(args.out_dir, "lr_train.npy"), lr_train)
    np.save(os.path.join(args.out_dir, "hr_train.npy"), hr_train)
    np.save(os.path.join(args.out_dir, "lr_val.npy"), lr_val)
    np.save(os.path.join(args.out_dir, "hr_val.npy"), hr_val)
    np.save(os.path.join(args.out_dir, "lr_test.npy"), lr_test)
    np.save(os.path.join(args.out_dir, "hr_test.npy"), hr_test)

    # Save the scale alongside the data so train.py/app.py can auto-detect it
    with open(os.path.join(args.out_dir, "scale.txt"), "w") as f:
        f.write(str(args.scale))

    print(f"\nDone. Saved to: {args.out_dir}")
    print(f"  Train: {lr_train.shape[0]} pairs | LR {lr_train.shape[1:3]} -> HR {hr_train.shape[1:3]}")
    print(f"  Val:   {lr_val.shape[0]} pairs")
    print(f"  Test:  {lr_test.shape[0]} pairs")
    print(f"\nNext step:")
    print(f"  python src/train.py --data_dir {args.out_dir} --model espcn --scale {args.scale} --epochs 30")


if __name__ == "__main__":
    main()
