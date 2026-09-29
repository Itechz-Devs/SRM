"""
crop_geotiff.py
-----------------
Full Sentinel-2 tiles from Planetary Computer (or Copernicus) are huge
(~11,000 x 11,000 pixels, several hundred MB) - too big to comfortably
upload through a browser or process in one web request.

This script uses rasterio's WINDOWED reading, which only reads the small
piece you ask for rather than loading the entire multi-hundred-MB file
into memory - fast, and works even on modest laptops.

Usage:
    python crop_geotiff.py --input maharashtra_sentinel2\\43QCB_2026-08-29.tif --size 512 --output crop1.tif

This crops a 512x512 pixel square from the CENTER of the input tile by
default. Use --row_off and --col_off to pick a different area instead.
"""

import argparse
import rasterio
from rasterio.windows import Window


def crop_geotiff(input_path, output_path, size=512, row_off=None, col_off=None):
    with rasterio.open(input_path) as src:
        print(f"Input tile size: {src.width} x {src.height} pixels")

        if row_off is None:
            row_off = (src.height - size) // 2
        if col_off is None:
            col_off = (src.width - size) // 2

        row_off = max(0, min(row_off, src.height - size))
        col_off = max(0, min(col_off, src.width - size))

        window = Window(col_off, row_off, size, size)
        transform = src.window_transform(window)

        # This only reads the small windowed region, not the whole file
        data = src.read(window=window)

        profile = src.profile.copy()
        profile.update({
            "height": size,
            "width": size,
            "transform": transform,
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

    print(f"Saved {size}x{size} crop to {output_path}")
    print("This file is ready to upload directly to your /enhance page.")


def main():
    parser = argparse.ArgumentParser(description="Crop a manageable area from a large Sentinel-2 GeoTIFF")
    parser.add_argument("--input", required=True, help="Path to the large downloaded GeoTIFF")
    parser.add_argument("--output", default="crop.tif", help="Where to save the cropped GeoTIFF")
    parser.add_argument("--size", type=int, default=512, help="Width/height of the crop in pixels")
    parser.add_argument("--row_off", type=int, default=None, help="Row offset (default: centered)")
    parser.add_argument("--col_off", type=int, default=None, help="Column offset (default: centered)")
    args = parser.parse_args()

    crop_geotiff(args.input, args.output, args.size, args.row_off, args.col_off)


if __name__ == "__main__":
    main()
