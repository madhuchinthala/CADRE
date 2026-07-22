#!/bin/bash
# ============================================
# CADRE Full Training Pipeline
# ============================================
# Run all 7 parts in sequence.
# Usage: bash scripts/run_pipeline.sh
# ============================================

set -e

echo "============================================"
echo "  CADRE — Full Training Pipeline"
echo "============================================"
echo ""

# Part 1 — Load & Freeze Backbone
echo "[PART 1/7] Loading and freezing VLA backbone..."
python -m src.models.vla_backbone \
    --model_path checkpoints/llava-v1.5-7b \
    --verify

# Part 2 + 3 + 4 — Train domains with EWC + Replay
for DOMAIN in domain_us domain_sg domain_eu domain_rainy; do
    DATASET="bdd100k"
    if [ "$DOMAIN" = "domain_sg" ] || [ "$DOMAIN" = "domain_eu" ]; then
        DATASET="nuscenes"
    fi

    echo ""
    echo "[PART 2-4] Training domain: $DOMAIN (dataset: $DATASET)..."
    python -m src.continual.continual_trainer \
        --config configs/base_config.yaml \
        --domain $DOMAIN \
        --dataset $DATASET \
        --ewc_lambda 5000 \
        --replay_ratio 0.3 \
        --replay_size 2000 \
        --epochs 10
done

# Part 5 — Train Domain Router
echo ""
echo "[PART 5/7] Training domain router..."
python -m src.router.domain_router \
    --config configs/router_config.yaml \
    --domains domain_us,domain_sg,domain_eu,domain_rainy \
    --epochs 20

# Part 6 — Train Output Heads
echo ""
echo "[PART 6/7] Training output heads..."
python -m src.heads.integration_layer \
    --config configs/heads_config.yaml \
    --heads waypoint,hazard,regulation,weather \
    --epochs 15

# Part 7 — Run CADRE-Bench
echo ""
echo "[PART 7/7] Running CADRE-Bench evaluation..."
python -m src.benchmark.cadre_bench \
    --config configs/benchmark_config.yaml \
    --domains domain_us,domain_sg,domain_eu,domain_rainy \
    --output_dir outputs/cadre_bench

echo ""
echo "============================================"
echo "  ✅ Pipeline complete!"
echo "  Results in: outputs/cadre_bench/"
echo "============================================"
