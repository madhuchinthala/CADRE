"""
Prepare Domain Splits
======================
Splits the raw BDD100K and nuScenes datasets into domain-specific
subsets and prints statistics.

Usage:
    python scripts/prepare_domains.py
    python scripts/prepare_domains.py --config configs/base_config.yaml
"""

import argparse
import sys
sys.path.insert(0, ".")

from src.data.domain_splitter import DomainSplitter
from src.data.transforms import DrivingTransforms


def main():
    parser = argparse.ArgumentParser(description="Prepare domain splits")
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    transforms = DrivingTransforms()

    splitter = DomainSplitter(
        config_path=args.config,
        transform=transforms.train_transform(),
    )

    # Print domain statistics
    splitter.print_domain_stats()


if __name__ == "__main__":
    main()
