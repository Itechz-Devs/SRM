"""
download_sentinel1_sar_mumbai.py
------------------------------------
Same as download_sentinel1_sar.py, scoped to just the Mumbai
Metropolitan Region (Mumbai, Thane, Vasai-Virar) instead of all
Maharashtra.

Downloads REAL Sentinel-1 SAR (radar) data - VV and VH polarization bands.
No cloud filter needed - radar sees through clouds and works at night.

Setup (one-time):
    pip install pystac-client planetary-computer requests

Run:
    python download_sentinel1_sar_mumbai.py

Output: 2 files per scene into ./mumbai_sar/
"""

import os
import requests
import pystac_client
import planetary_computer

MUMBAI_METRO_BBOX = [72.55, 18.85, 73.15, 19.55]
OUTPUT_DIR = "mumbai_sar"
SEARCH_MONTHS_BACK = "2025-06-01/2026-09-01"
MAX_SCENES = 10
BANDS = ["vv", "vh"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Connecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    print(f"Searching Sentinel-1 GRD over Mumbai/Thane/Vasai "
          f"(no cloud filter needed - radar sees through clouds)...")
    search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=MUMBAI_METRO_BBOX,
        datetime=SEARCH_MONTHS_BACK,
        max_items=MAX_SCENES,
    )
    items = list(search.items())
    print(f"Found {len(items)} scenes.\n")

    if not items:
        print("No results - try widening SEARCH_MONTHS_BACK.")
        return

    print(f"Downloading {len(BANDS)} bands per scene "
          f"(~{len(items) * len(BANDS)} files total - SAR files are large, this will take a while)...\n")

    for i, item in enumerate(items):
        date = item.properties.get("datetime", "?")[:10]
        scene_id = item.id
        print(f"[{i+1}/{len(items)}] Scene {scene_id} | {date}")

        for band in BANDS:
            asset = item.assets.get(band)
            if asset is None:
                print(f"  Band {band} not found for this scene, skipping.")
                continue

            out_path = os.path.join(OUTPUT_DIR, f"{scene_id}_{band}.tif")
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
    print("  python data_pipeline_multiband.py --input_dir mumbai_sar "
          "--out_dir data/processed_x4_sar --bands vv,vh --scale 4 --lr_size 16 --data_type sar")


if __name__ == "__main__":
    main()
