"""
download_sentinel2_infrared_mumbai.py
------------------------------------------
Same as download_sentinel2_infrared.py, scoped to just the Mumbai
Metropolitan Region (Mumbai, Thane, Vasai-Virar) instead of all
Maharashtra - much smaller download, only 1-3 tiles expected.

Downloads REAL Sentinel-2 Near-Infrared data alongside RGB, as separate
per-band GeoTIFFs (Red=B04, Green=B03, Blue=B02, NIR=B08).

Setup (one-time):
    pip install pystac-client planetary-computer requests

Run:
    python download_sentinel2_infrared_mumbai.py

Output: 4 files per tile into ./mumbai_infrared/
"""

import os
import requests
import pystac_client
import planetary_computer

MUMBAI_METRO_BBOX = [72.55, 18.85, 73.15, 19.55]
OUTPUT_DIR = "mumbai_infrared"
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

    print(f"Searching Sentinel-2 L2A over Mumbai/Thane/Vasai for RGB+NIR bands "
          f"(cloud cover < {MAX_CLOUD_COVER}%)...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=MUMBAI_METRO_BBOX,
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
    print("  python data_pipeline_multiband.py --input_dir mumbai_infrared "
          "--out_dir data/processed_x4_ir --bands B04,B03,B02,B08 --scale 4 --lr_size 16")


if __name__ == "__main__":
    main()
