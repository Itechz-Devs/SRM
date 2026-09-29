"""
data_pipeline.py
-----------------
Loads satellite image patches (EuroSAT_RGB, 64x64) using OpenCV, converts them
into numpy matrices, and builds Low-Resolution / High-Resolution (LR/HR) pairs
for training a super-resolution CNN.

WHY SYNTHETIC PAIRS: real paired LR/HR satellite datasets (true 10m + true <4m
of the SAME location) are hard to obtain. The standard approach used in SR
research is: treat your available imagery as "HR" ground truth, then
synthetically degrade it (downsample) to create the "LR" input. The model
then learns to reverse that degradation. This is what SRCNN/EDSR papers do
during training.

This pipeline is written to be BAND-COUNT AGNOSTIC: it reads however many
channels are in the image (3 for RGB here). If you later have Sentinel-2
GeoTIFFs with more bands (e.g. NIR), point this at a rasterio-based loader
instead of cv2.imread and the rest of the pipeline (LR/HR generation, model,
training) does not need to change.

Run directly to preprocess the whole dataset:
    python src/data_pipeline.py --data_dir data/raw/EuroSAT_RGB --out_dir data/processed
"""

import os
import glob
import argparse
import numpy as np
import cv2


def load_images_as_matrices(data_dir, img_size=64, limit_per_class=None):
    """
    Walks every class subfolder in data_dir, reads each image with OpenCV,
    and returns a single 4D numpy array of shape (N, img_size, img_size, 3).

    OpenCV loads images as BGR by default - we convert to RGB so channel
    order matches what everyone expects (matplotlib, TensorFlow, etc.)
    """
    class_folders = sorted(
        [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    )
    print(f"Found {len(class_folders)} classes: {class_folders}")

    images = []
    for class_name in class_folders:
        class_path = os.path.join(data_dir, class_name)
        file_paths = sorted(glob.glob(os.path.join(class_path, "*.jpg")))
        if limit_per_class:
            file_paths = file_paths[:limit_per_class]

        for fp in file_paths:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)  # matrix, shape (H, W, 3), BGR, uint8
            if img is None:
                continue  # skip unreadable files rather than crash the whole run
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != img_size or img.shape[1] != img_size:
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            images.append(img)

        print(f"  Loaded {len(file_paths)} images from {class_name}")

    images = np.array(images, dtype=np.uint8)
    print(f"Total images loaded: {images.shape}")
    return images


def normalize(images_uint8):
    """Scale pixel values from [0, 255] to [0, 1] float32 for the network."""
    return images_uint8.astype(np.float32) / 255.0


def make_lr_hr_pairs(hr_images_float, scale=4):
    """
    Given HR images (already normalized to [0,1]), create matching LR inputs:
      1. Downsample by `scale` (this simulates the loss of detail a real
         medium-resolution sensor would capture).
      2. Upsample back to the original size using bicubic interpolation.
         This gives the network a same-size, blurry input to sharpen - this
         is the classic SRCNN training setup (the network learns to REFINE
         a bicubic-upsampled image, not to change the image dimensions).

    Returns (lr_images, hr_images) - both same shape, ready for training.
    """
    n, h, w, c = hr_images_float.shape
    lr_images = np.zeros_like(hr_images_float)

    for i in range(n):
        hr = hr_images_float[i]
        small = cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        upscaled_back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        lr_images[i] = upscaled_back

    return lr_images, hr_images_float


def train_val_test_split(lr, hr, val_frac=0.1, test_frac=0.1, seed=42):
    """Simple shuffled split - keeps LR/HR pairs aligned."""
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
    parser = argparse.ArgumentParser(description="Preprocess satellite imagery into LR/HR training pairs")
    parser.add_argument("--data_dir", type=str, default="data/raw/EuroSAT_RGB")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--scale", type=int, default=4, help="Downsampling factor to simulate medium-res input")
    parser.add_argument("--limit_per_class", type=int, default=None,
                         help="Cap images per class for a faster first run, e.g. 300")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Step 1/4: Loading images with OpenCV into matrices...")
    hr_uint8 = load_images_as_matrices(args.data_dir, args.img_size, args.limit_per_class)

    print("Step 2/4: Normalizing to [0, 1]...")
    hr_float = normalize(hr_uint8)

    print(f"Step 3/4: Generating LR/HR pairs (downsample factor {args.scale})...")
    lr_float, hr_float = make_lr_hr_pairs(hr_float, scale=args.scale)

    print("Step 4/4: Splitting into train/val/test and saving...")
    (lr_train, hr_train), (lr_val, hr_val), (lr_test, hr_test) = train_val_test_split(lr_float, hr_float)

    np.save(os.path.join(args.out_dir, "lr_train.npy"), lr_train)
    np.save(os.path.join(args.out_dir, "hr_train.npy"), hr_train)
    np.save(os.path.join(args.out_dir, "lr_val.npy"), lr_val)
    np.save(os.path.join(args.out_dir, "hr_val.npy"), hr_val)
    np.save(os.path.join(args.out_dir, "lr_test.npy"), lr_test)
    np.save(os.path.join(args.out_dir, "hr_test.npy"), hr_test)

    print("Done. Saved to:", args.out_dir)
    print(f"  Train: {lr_train.shape[0]} pairs")
    print(f"  Val:   {lr_val.shape[0]} pairs")
    print(f"  Test:  {lr_test.shape[0]} pairs")


if __name__ == "__main__":
    main()
