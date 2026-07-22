"""
Download nuScenes Dataset
==========================
Helper script to download nuScenes mini or full dataset.
Requires registration at https://www.nuscenes.org/nuscenes#download

Usage:
    python scripts/download_nuscenes.py --version mini
    python scripts/download_nuscenes.py --version full
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download nuScenes dataset")
    parser.add_argument(
        "--version",
        choices=["mini", "full"],
        default="mini",
        help="Which version to download",
    )
    parser.add_argument(
        "--data_root",
        default="data/nuscenes",
        help="Where to store the dataset",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_root)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  nuScenes Dataset Download Helper")
    print("=" * 60)
    print()

    if args.version == "mini":
        print("  📦 Mini set (~4 GB) — great for testing your pipeline")
        print()
        print("  Steps:")
        print("  1. Go to: https://www.nuscenes.org/nuscenes#download")
        print("  2. Sign up / Log in")
        print("  3. Download 'v1.0-mini.tgz' (~4 GB)")
        print(f"  4. Save to: {data_dir.absolute()}")
        print()
        print("  After downloading, extract with:")
        print(f"    cd {data_dir.absolute()}")
        print(f"    tar -xzf v1.0-mini.tgz")
        print()
        print("  Then install the devkit:")
        print("    pip install nuscenes-devkit")
        print()
        print("  Verify with:")
        print("    python -c \"from nuscenes.nuscenes import NuScenes; "
              f"nusc = NuScenes('v1.0-mini', '{data_dir}'); "
              "print(f'Scenes: {len(nusc.scene)}')\"")

    elif args.version == "full":
        print("  📦 Full trainval set (~350 GB) — for actual training")
        print()
        print("  Steps:")
        print("  1. Go to: https://www.nuscenes.org/nuscenes#download")
        print("  2. Sign up / Log in")
        print("  3. Download ALL of these:")
        print("     - v1.0-trainval_meta.tgz")
        print("     - v1.0-trainval01_blobs.tgz through v1.0-trainval10_blobs.tgz")
        print("     - nuScenes-map-expansion-v1.3.zip")
        print(f"  4. Save ALL to: {data_dir.absolute()}")
        print()
        print("  After downloading, extract ALL into the SAME directory:")
        print(f"    cd {data_dir.absolute()}")
        print("    tar -xzf v1.0-trainval_meta.tgz")
        print("    tar -xzf v1.0-trainval01_blobs.tgz")
        print("    tar -xzf v1.0-trainval02_blobs.tgz")
        print("    # ... continue for all blobs ...")
        print("    unzip nuScenes-map-expansion-v1.3.zip")
        print()
        print("  ⚠️  Extract ALL into the SAME folder. They merge, don't overwrite.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
