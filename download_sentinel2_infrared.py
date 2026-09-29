"""
download_sentinel2_infrared.py
---------------------------------
Downloads REAL Sentinel-2 Near-Infrared data alongside RGB, as separate
per-band GeoTIFFs (Red=B04, Green=B03, Blue=B02, NIR=B08) - this is
genuine infrared data from the same Sentinel-2 satellites, not a
simulation. Unlike the "visual" composite used for the RGB pipeline
(which is a pre-made true-color JPEG-like image), these are the raw
individual band files, which is what lets us include the NIR band at all.

Setup (one-time):
    pip install pystac-client planetary-computer requests

Run:
    python download_sentinel2_infrared.py

Output: 4 files per tile (one per band) into ./maharashtra_infrared/, e.g.:
    43QCB_2026-06-01_B04.tif   (Red)
    43QCB_2026-06-01_B03.tif   (Green)
    43QCB_2026-06-01_B02.tif   (Blue)
    43QCB_2026-06-01_B08.tif   (Near-Infrared)

These 4 files per tile are read together by data_pipeline_multiband.py to
build 4-channel (RGB+NIR) training patches.
"""

import os
import requests
import pystac_client
import planetary_computer

MAHARASHTRA_BBOX = [72.6, 15.6, 80.9, 22.1]
OUTPUT_DIR = "maharashtra_infrared"
MAX_CLOUD_COVER = 20
SEARCH_MONTHS_BACK = "2025-01-01/2026-09-01"
BANDS = ["B04", "B03", "B02", "B08"]  # Red, Green, Blue, Near-Infrared


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Connecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    print(f"Searching Sentinel-2 L2A over Maharashtra for RGB+NIR bands "
          f"(cloud cover < {MAX_CLOUD_COVER}%)...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=MAHARASHTRA_BBOX,
        datetime=SEARCH_MONTHS_BACK,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        max_items=300,
    )
    items = list(search.items())
    print(f"Found {len(items)} candidate scenes.\n")

    if not items:
        print("No results - try raising MAX_CLOUD_COVER.")
        return

    best_per_tile = {}
    for item in items:
        tile_id = item.properties.get("s2:mgrs_tile", item.id)
        cloud = item.properties.get("eo:cloud_cover", 100)
        if tile_id not in best_per_tile or cloud < best_per_tile[tile_id].properties.get("eo:cloud_cover", 100):
            best_per_tile[tile_id] = item

    print(f"Unique tiles: {len(best_per_tile)}")
    print(f"Downloading {len(BANDS)} bands per tile (~{len(best_per_tile) * len(BANDS)} files total)...\n")

    for i, (tile_id, item) in enumerate(sorted(best_per_tile.items())):
        date = item.properties.get("datetime", "?")[:10]
        print(f"[{i+1}/{len(best_per_tile)}] Tile {tile_id} | {date}")

        for band in BANDS:
            asset = item.assets.get(band)
            if asset is None:
                print(f"  Band {band} not found for this item, skipping.")
                continue

            out_path = os.path.join(OUTPUT_DIR, f"{tile_id}_{date}_{band}.tif")
            if os.path.exists(out_path):
                print(f"  {band}: already downloaded.")
                continue

            try:
                with requests.get(asset.href, stream=True) as r:
                    r.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192 * 16):
                            f.write(chunk)
                print(f"  {band}: downloaded.")
            except Exception as e:
                print(f"  {band}: FAILED ({e})")

    print("\nDone. Next step:")
    print("  python data_pipeline_multiband.py --input_dir maharashtra_infrared "
          "--out_dir data/processed_x4_ir --bands B04,B03,B02,B08 --scale 4 --lr_size 16")


if __name__ == "__main__":
    main()
