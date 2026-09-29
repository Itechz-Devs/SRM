"""
download_landsat_thermal.py
-------------------------------
Downloads REAL Landsat thermal band data (the "lwir11" band - long-wave
infrared, ~11 micron wavelength, used for surface temperature). This is
from a different satellite mission (Landsat, not Sentinel-2), since
Sentinel-2 does not carry a thermal sensor at all.

Setup (one-time):
    pip install pystac-client planetary-computer requests

Run:
    python download_landsat_thermal.py

Output: 1 file per tile into ./maharashtra_thermal/, e.g.:
    LC09_..._lwir11.tif

Note: Landsat's native resolution for thermal is 100m/pixel (resampled to
30m in the product) - much coarser than Sentinel-2's 10m. This is a real
physical limitation of thermal sensors, not a processing choice - thermal
imaging inherently needs a larger detector area, so it comes at lower
spatial resolution. Keep this in mind when interpreting a "target
resolution" for a thermal model - the starting point is already coarser.
"""

import os
import requests
import pystac_client
import planetary_computer

MAHARASHTRA_BBOX = [72.6, 15.6, 80.9, 22.1]
OUTPUT_DIR = "maharashtra_thermal"
MAX_CLOUD_COVER = 20
SEARCH_MONTHS_BACK = "2025-01-01/2026-09-01"
BAND = "lwir11"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Connecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    print(f"Searching Landsat Collection 2 Level 2 over Maharashtra for the "
          f"thermal band (cloud cover < {MAX_CLOUD_COVER}%)...")
    search = catalog.search(
        collections=["landsat-c2-l2"],
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

    # Landsat tiles by WRS path/row, not MGRS - dedupe on that combination
    best_per_tile = {}
    for item in items:
        path = item.properties.get("landsat:wrs_path", "?")
        row = item.properties.get("landsat:wrs_row", "?")
        tile_id = f"P{path}R{row}"
        cloud = item.properties.get("eo:cloud_cover", 100)
        if tile_id not in best_per_tile or cloud < best_per_tile[tile_id].properties.get("eo:cloud_cover", 100):
            best_per_tile[tile_id] = item

    print(f"Unique path/row tiles: {len(best_per_tile)}\n")

    for i, (tile_id, item) in enumerate(sorted(best_per_tile.items())):
        date = item.properties.get("datetime", "?")[:10]
        print(f"[{i+1}/{len(best_per_tile)}] Tile {tile_id} | {date}")

        asset = item.assets.get(BAND)
        if asset is None:
            print(f"  Band {BAND} not found for this item, skipping.")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{tile_id}_{date}_{BAND}.tif")
        if os.path.exists(out_path):
            print(f"  Already downloaded.")
            continue

        try:
            with requests.get(asset.href, stream=True) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192 * 16):
                        f.write(chunk)
            print(f"  Downloaded.")
        except Exception as e:
            print(f"  FAILED ({e})")

    print("\nDone. Next step:")
    print("  python data_pipeline_multiband.py --input_dir maharashtra_thermal "
          "--out_dir data/processed_x4_thermal --bands lwir11 --scale 4 --lr_size 16 --data_type thermal")


if __name__ == "__main__":
    main()
