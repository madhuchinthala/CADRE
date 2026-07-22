#!/bin/bash
# ============================================
# Download BDD100K from Kaggle (solesensei)
# ============================================
# Prerequisites:
#   1. pip install kaggle
#   2. Place kaggle.json in ~/.kaggle/
#   3. chmod 600 ~/.kaggle/kaggle.json
# ============================================

set -e

echo "============================================"
echo "  Downloading BDD100K from Kaggle"
echo "  Dataset: solesensei/solesensei_bdd100k"
echo "============================================"

# Create data directory
mkdir -p data/bdd100k

# Download from Kaggle
echo "[1/3] Downloading dataset..."
kaggle datasets download -d solesensei/solesensei_bdd100k -p data/bdd100k

# Extract
echo "[2/3] Extracting..."
cd data/bdd100k
unzip -q -o solesensei_bdd100k.zip
rm -f solesensei_bdd100k.zip
cd ../..

# Verify
echo "[3/3] Verifying..."
echo ""
echo "Directory structure:"
find data/bdd100k -type d | head -20
echo ""
echo "Image counts:"
echo "  train: $(find data/bdd100k -path '*/train/*.jpg' 2>/dev/null | wc -l) images"
echo "  val:   $(find data/bdd100k -path '*/val/*.jpg'   2>/dev/null | wc -l) images"
echo "  test:  $(find data/bdd100k -path '*/test/*.jpg'  2>/dev/null | wc -l) images"
echo ""
echo "============================================"
echo "  ✅ BDD100K download complete!"
echo "============================================"
