"""
download_mumbai_metro_sentinel2.py
--------------------------------------
Searches Microsoft's Planetary Computer for Sentinel-2 imagery covering
just the Mumbai Metropolitan Region (Mumbai, Thane, Vasai-Virar), then
downloads the CLEAREST available image for each unique tile touching
this area.

This is a MUCH smaller area than all of Maharashtra (~4,300 km² vs
~307,000 km²) - expect only 1-3 unique Sentinel-2 tiles, a few hundred MB
total, and a download that finishes in minutes rather than hours. Good
for a fast, focused demo centered on one metro region.

Setup (one-time):
    pip install pystac-client planetary-computer requests

Run:
    python download_mumbai_metro_sentinel2.py

Output: 1-3 GeoTIFF files saved into ./mumbai_metro_sentinel2/
"""

import os
import requests
import pystac_client
import planetary_computer

# Bounding box covering Mumbai, Thane, and Vasai-Virar (Mumbai Metropolitan
# Region) - west, south, east, north
MUMBAI_METRO_BBOX = [72.55, 18.85, 73.15, 19.55]

OUTPUT_DIR = "mumbai_metro_sentinel2"
MAX_CLOUD_COVER = 20
SEARCH_MONTHS_BACK = "2025-01-01/2026-09-01"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Connecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    print(f"Searching Sentinel-2 L2A over the Mumbai Metropolitan Region "
          f"(Mumbai/Thane/Vasai) (cloud cover < {MAX_CLOUD_COVER}%, {SEARCH_MONTHS_BACK})...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=MUMBAI_METRO_BBOX,
        datetime=SEARCH_MONTHS_BACK,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        max_items=500,  # search broadly - we'll de-duplicate below
    )

    items = list(search.items())
    print(f"Found {len(items)} candidate scenes across all dates/tiles.\n")

    if not items:
        print("No results at all - try raising MAX_CLOUD_COVER further.")
        return

    # Keep only the CLEAREST (lowest cloud cover) scene per unique tile,
    # so we get one good image per patch of ground, not duplicates.
    best_per_tile = {}
    for item in items:
        tile_id = item.properties.get("s2:mgrs_tile", item.id)
        cloud = item.properties.get("eo:cloud_cover", 100)
        if tile_id not in best_per_tile or cloud < best_per_tile[tile_id].properties.get("eo:cloud_cover", 100):
            best_per_tile[tile_id] = item

    print(f"Unique tiles covering Mumbai/Thane/Vasai: {len(best_per_tile)}")
    print("Downloading the clearest available image for each tile...\n")

    for i, (tile_id, item) in enumerate(sorted(best_per_tile.items())):
        cloud = item.properties.get("eo:cloud_cover", "?")
        date = item.properties.get("datetime", "?")
        print(f"[{i+1}/{len(best_per_tile)}] Tile {tile_id} | {date} | cloud cover: {cloud}%")

        visual_asset = item.assets.get("visual")
        if visual_asset is None:
            print("  No 'visual' asset found, skipping.")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{tile_id}_{date[:10]}.tif")
        if os.path.exists(out_path):
            print(f"  Already downloaded: {out_path}")
            continue

        url = visual_asset.href
        print(f"  Downloading to {out_path} (~100-300MB, please be patient)...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192 * 16):
                        f.write(chunk)
            print(f"  Done: {out_path}\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")

    print("All downloads attempted. Next step: extract training patches with")
    print("extract_patches.py - see MAHARASHTRA_DATA_STEPS.txt for the full workflow.")


if __name__ == "__main__":
    main()
