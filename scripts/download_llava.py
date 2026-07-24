"""
Download LLaVA-v1.5-7B Pretrained Model
=========================================
Downloads the LLaVA-v1.5-7B model from HuggingFace.
Approximately 14 GB download.

Usage:
    python scripts/download_llava.py
    python scripts/download_llava.py --model_id llava-hf/llava-1.5-13b-hf
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download LLaVA model")
    parser.add_argument(
        "--model_id",
        default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/llava-v1.5-7b",
        help="Where to save the model",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Downloading LLaVA-v1.5-7B")
    print("=" * 60)
    print(f"  Model:  {args.model_id}")
    print(f"  Output: {output_dir.absolute()}")
    print(f"  Size:   ~14 GB")
    print("=" * 60)
    print()

    from huggingface_hub import snapshot_download

    # Try modern argument set first, fall back to legacy if it fails
    try:
        snapshot_download(
            repo_id=args.model_id,
            local_dir=str(output_dir),
        )
    except TypeError:
        # Fallback for older huggingface_hub versions
        snapshot_download(
            repo_id=args.model_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    print()
    print("=" * 60)
    print("  ✅ LLaVA-v1.5-7B downloaded successfully!")
    print("=" * 60)

    # Verify
    expected_files = ["config.json", "tokenizer.json"]
    for f in expected_files:
        if (output_dir / f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} — MISSING")


if __name__ == "__main__":
    main()
