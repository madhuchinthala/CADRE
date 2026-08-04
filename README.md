# 🚗 CADRE — Continual Adaptation for Driving with Robust Evolution

> **Problem Statement B5:** Design a continual adaptation method for VLA autonomous driving models that incorporates new regional regulations, road layouts, and weather patterns while retaining prior driving competence across previously learned environments.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Key Results](#-key-results)
3. [Architecture](#-architecture)
4. [Project Structure](#-project-structure)
5. [Setup & Installation](#-setup--installation)
6. [Training Pipeline](#-training-pipeline)
7. [Benchmark Results (CADRE-Bench)](#-benchmark-results-cadre-bench)
8. [How to Share the Model](#-how-to-share-the-model)
9. [Configuration](#-configuration)
10. [Deliverables Checklist](#-deliverables-checklist)
11. [References](#-references)

---

## 🔍 Overview

**CADRE** is a continual learning framework for **Vision-Language-Action (VLA)** autonomous driving models. It enables a single model to learn new driving domains (regions, regulations, weather) **sequentially** without forgetting previously learned domains — solving the problem of **catastrophic forgetting** in neural networks.

### How It Works

```
Training Sequence:  domain_us → domain_sg → domain_eu → domain_rainy
                    (BDD100K)   (nuScenes)  (nuScenes)   (BDD100K)
```

The system uses three complementary strategies:

| Strategy | What It Does | Module |
|----------|-------------|--------|
| **LoRA Adapters** | Adds small trainable parameters (~0.35%) to the frozen 7B backbone — avoids retraining the whole model | `src/adapters/lora_adapter.py` |
| **EWC (Elastic Weight Consolidation)** | Penalizes changes to parameters important for previous domains | `src/continual/ewc.py` |
| **Experience Replay** | Mixes ~30% old-domain samples into new-domain training batches | `src/continual/replay_buffer.py` |

### Datasets Used

| Dataset | Domains | Description |
|---------|---------|-------------|
| **BDD100K** | `domain_us`, `domain_rainy` | 100K driving videos from US roads (Berkeley) |
| **nuScenes** | `domain_sg`, `domain_eu` | 1000 driving scenes from Singapore and Boston |

---

## 📊 Key Results

| Metric | Score | Meaning |
|--------|-------|---------|
| **BWT** (Backward Transfer) | **-1.17%** | ✅ Near-zero forgetting of old domains |
| **FWT** (Forward Transfer) | **+23.33%** | ✅ Prior training helps learn new domains |
| **Plasticity** | **98.46%** | ✅ Excellent ability to learn new domains |
| **Efficiency** | **99.65%** | ✅ Only 0.35% parameter overhead per domain |
| **CDAR** (Composite Score) | **0.9735** | ✅ Outstanding overall performance |

### Performance Matrix

After training all 4 domains sequentially, the model retains >92% accuracy on every domain:

```
                 Evaluate on →
                 US      SG      EU      Rainy
After US:      [ 0.94    0.32    0.28    0.25  ]
After SG:      [ 0.93    0.95    0.41    0.38  ]
After EU:      [ 0.92    0.94    0.96    0.52  ]
After Rainy:   [ 0.925   0.935   0.955   0.97  ]   ← Final model
```

**Key takeaway:** The diagonal shows strong learning (94-97%), and the final row shows minimal forgetting — US only dropped from 94% → 92.5% after learning 3 more domains.

---

## 🏗 Architecture

```
┌─────────────┐    ┌────────────────────┐    ┌──────────────┐
│  Driving     │───▶│  LLaVA-v1.5-7B     │───▶│ Domain Router │
│  Image       │    │  (Frozen Backbone) │    │ (Classifier)  │
└─────────────┘    │  7.06B params       │    └──────┬───────┘
                   └────────────────────┘           │
                                                     ▼
                   ┌──────┬──────┬──────┬──────────────┐
                   │LoRA  │LoRA  │LoRA  │LoRA          │
                   │US    │SG    │EU    │Rainy         │
                   │24.7M │24.7M │24.7M │24.7M params  │
                   └──┬───┴──┬───┴──┬───┴──┬───────────┘
                      │      │      │      │
                      ▼      ▼      ▼      ▼
              ┌──────────────────────────────────────┐
              │  Multi-Head Output                    │
              │  ┌──────────┬──────────┬───────────┐ │
              │  │Waypoint  │ Hazard   │Regulation │ │
              │  │Prediction│Detection │ Parsing   │ │
              │  └──────────┴──────────┴───────────┘ │
              │  ┌──────────┬─────────────────────┐  │
              │  │ Weather  │ Integration Layer    │  │
              │  │ Classify │ (Attention Fusion)   │  │
              │  └──────────┴─────────────────────┘  │
              └──────────────────────────────────────┘
```

### Continual Learning Components

```
During Training:
  ┌────────────────────────────────────────────────┐
  │  New Domain Data (70%) ──┐                     │
  │                          ├──▶ Mixed DataLoader  │
  │  Replay Buffer (30%) ───┘    │                 │
  │                              ▼                 │
  │  Loss = Task_Loss + λ × EWC_Penalty            │
  │                              │                 │
  │  Only LoRA params updated ◀──┘                 │
  └────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
CADRE/
├── configs/                          # All configuration files
│   ├── base_config.yaml              #   Main training config (paths, domains, hyperparams)
│   ├── ewc_config.yaml               #   EWC regularization settings
│   ├── replay_config.yaml            #   Experience replay settings
│   ├── lora_config.yaml              #   LoRA adapter configuration
│   ├── router_config.yaml            #   Domain router config
│   ├── heads_config.yaml             #   Output heads config
│   └── benchmark_config.yaml         #   CADRE-Bench evaluation config
│
├── src/                              # Source code
│   ├── models/
│   │   └── vla_backbone.py           #   LLaVA-v1.5-7B backbone (load & freeze)
│   ├── adapters/
│   │   └── lora_adapter.py           #   LoRA adapter injection & management
│   ├── continual/
│   │   ├── continual_trainer.py      #   Main training loop (EWC + Replay)
│   │   ├── ewc.py                    #   Elastic Weight Consolidation
│   │   └── replay_buffer.py          #   Experience replay buffer
│   ├── router/
│   │   └── domain_router.py          #   Domain routing classifier
│   ├── heads/
│   │   └── integration_layer.py      #   4 output heads + attention fusion
│   ├── data/
│   │   ├── dataloader.py             #   DataLoader factory
│   │   ├── bdd100k_dataset.py        #   BDD100K dataset class
│   │   └── nuscenes_dataset.py       #   nuScenes dataset class
│   └── benchmark/
│       └── cadre_bench.py            #   CADRE-Bench evaluation protocol
│
├── scripts/                          # Runnable scripts
│   ├── run_pipeline.py               #   Resumable full training pipeline
│   ├── run_pipeline.bat              #   Windows batch launcher
│   ├── generate_visualizations.py    #   Generate result plots
│   ├── download_bdd100k.sh           #   BDD100K download script
│   ├── download_nuscenes.py          #   nuScenes download script
│   └── download_llava.py             #   LLaVA model download script
│
├── checkpoints/                      # Trained model artifacts
│   ├── llava-v1.5-7b/               #   Frozen backbone weights
│   ├── lora_adapters/                #   Per-domain LoRA adapters
│   │   ├── domain_us/                #     US adapter (~40 MB)
│   │   ├── domain_sg/                #     Singapore adapter
│   │   ├── domain_eu/                #     EU adapter
│   │   └── domain_rainy/             #     Rainy adapter
│   ├── fisher_matrices/              #   EWC Fisher information
│   ├── router/                       #   Domain router weights
│   ├── heads/                        #   Multi-head model weights
│   └── pipeline_state.json           #   Pipeline progress tracker
│
├── outputs/                          # Results
│   ├── cadre_bench/
│   │   └── cadre_bench_report.json   #   Benchmark metrics report
│   └── visualizations/               #   Generated plots
│       ├── performance_matrix_heatmap.png
│       ├── metrics_summary.png
│       ├── training_timeline.png
│       ├── parameter_efficiency.png
│       ├── domain_performance.png
│       └── architecture_overview.png
│
├── data/                             # Datasets (not in git)
│   ├── bdd100k/                      #   BDD100K images + annotations
│   └── nuscenes/                     #   nuScenes data
│
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
└── README.md                         # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites

- **Python** 3.9+
- **GPU**: NVIDIA GPU with ≥12 GB VRAM (tested on RTX 4000 Ada)
- **CUDA**: 11.8 or 12.x
- **OS**: Windows 10/11 (also works on Linux)

### Step 1: Clone and Create Virtual Environment

```bash
git clone <repository-url>
cd CADRE

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 2: Install PyTorch with CUDA

```bash
# For CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Project Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 4: Download Datasets

```bash
# BDD100K (via Kaggle)
# 1. Set up Kaggle credentials (~/.kaggle/kaggle.json)
# 2. Run:
bash scripts/download_bdd100k.sh

# nuScenes
python scripts/download_nuscenes.py
```

### Step 5: Download LLaVA Model

```bash
python scripts/download_llava.py
```

### Step 6: Verify Installation

```bash
python -m src.models.vla_backbone --model_path checkpoints/llava-v1.5-7b --verify
```

---

## 🚀 Training Pipeline

### Run Full Pipeline (Recommended)

```bash
python scripts/run_pipeline.py
```

This runs all 8 stages in order:

| # | Stage | Description | ~Time |
|---|-------|-------------|-------|
| 1 | `verify_backbone` | Load & freeze LLaVA backbone | ~10 sec |
| 2 | `domain_us` | Train on US driving data (BDD100K) | ~30-60 min |
| 3 | `domain_sg` | Train on Singapore data (nuScenes) | ~30-60 min |
| 4 | `domain_eu` | Train on EU/Boston data (nuScenes) | ~30-60 min |
| 5 | `domain_rainy` | Train on rainy/adverse weather (BDD100K) | ~30-60 min |
| 6 | `train_router` | Train domain routing classifier | ~15 sec |
| 7 | `train_heads` | Train 4 output heads + integration layer | ~10-15 min |
| 8 | `benchmark` | Run CADRE-Bench evaluation | ~30 sec |

### Resumable Training

The pipeline **automatically saves progress**. If interrupted (Ctrl+C, crash, power loss), just rerun the same command — it skips completed stages:

```bash
# Resume after interruption
python scripts/run_pipeline.py

# Check status without running
python scripts/run_pipeline.py --status

# Redo a specific stage
python scripts/run_pipeline.py --redo domain_us

# Run only one stage
python scripts/run_pipeline.py --only benchmark

# Reset all progress
python scripts/run_pipeline.py --reset
```

### Generate Visualizations

After training completes:

```bash
python scripts/generate_visualizations.py
```

This creates 6 publication-quality plots in `outputs/visualizations/`.

---

## 📈 Benchmark Results (CADRE-Bench)

CADRE-Bench is our custom benchmark protocol for evaluating continual adaptation in autonomous driving. It measures:

### Metrics Explained

| Metric | Formula | Score | Interpretation |
|--------|---------|-------|----------------|
| **BWT** | `(1/(T-1)) × Σ (R[T,i] - R[i,i])` | **-1.17%** | How much new learning hurts old domains. Closer to 0 = less forgetting |
| **FWT** | `(1/(T-1)) × Σ (R[i-1,i] - R_0[i])` | **+23.33%** | How much old training helps new domains. Higher = better transfer |
| **Plasticity** | `(1/T) × Σ R[i,i] / R*[i]` | **98.46%** | How well the model learns each new domain vs single-task upper bound |
| **Efficiency** | `1 - (adapter_params / backbone_params)` | **99.65%** | Parameter overhead. Higher = more efficient |
| **CDAR** | `0.3×Stability + 0.3×Plasticity + 0.2×Transfer + 0.2×Efficiency` | **0.9735** | Composite score (0-1). Higher = better |

### Parameter Statistics

| Component | Parameters | Storage |
|-----------|-----------|---------|
| Frozen backbone (LLaVA-v1.5-7B) | 7,063,000,000 | ~14 GB |
| LoRA adapter (per domain) | 24,720,000 | ~40 MB |
| All 4 adapters combined | 98,880,000 | ~160 MB |
| **Overhead per domain** | **0.35%** | **~40 MB** |

---

## 📦 How to Share the Model

### Share LoRA Adapters Only (Recommended — ~160 MB)

The trained LoRA adapters are small and self-contained. Anyone with the same LLaVA backbone can use them:

```bash
# Zip just the adapters
python -c "import shutil; shutil.make_archive('cadre_adapters', 'zip', 'checkpoints/lora_adapters')"
# Creates cadre_adapters.zip (~160 MB for all 4 domains)
```

### Share Complete Model (All Artifacts)

```bash
# Zip everything needed to reproduce
python -c "
import shutil, os
files = ['checkpoints/lora_adapters', 'checkpoints/router', 'checkpoints/heads',
         'checkpoints/fisher_matrices', 'outputs/cadre_bench']
for f in files:
    print(f'Including: {f}')
shutil.make_archive('cadre_full_model', 'zip', '.', 'checkpoints')
"
```

### Load a Shared Model

```python
from peft import PeftModel
from transformers import LlavaForConditionalGeneration

# Load base model
base_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Load domain-specific adapter
model = PeftModel.from_pretrained(base_model, "checkpoints/lora_adapters/domain_us")
```

---

## ⚙️ Configuration

All configurations are in `configs/`. Key settings:

### `base_config.yaml` — Main Configuration

```yaml
model:
  backbone: "llava-hf/llava-1.5-7b-hf"
  dtype: "float16"
  freeze_backbone: true

training:
  batch_size: 4
  learning_rate: 2.0e-4
  max_epochs: 10
  gradient_accumulation_steps: 4

domains:
  sequence:
    - name: "domain_us"      # US roads (BDD100K, clear weather)
    - name: "domain_sg"      # Singapore (nuScenes)
    - name: "domain_eu"      # Boston/EU (nuScenes)
    - name: "domain_rainy"   # Adverse weather (BDD100K, rain/fog/snow)
```

### `ewc_config.yaml` — EWC Settings

```yaml
ewc:
  lambda_ewc: 5000.0         # Regularization strength
  fisher_n_samples: 200      # Samples for Fisher computation
  variant: "online_ewc"      # online_ewc or standard
  gamma: 0.95                # Decay factor for online EWC
```

### `lora_config.yaml` — LoRA Settings

```yaml
lora:
  r: 16                      # LoRA rank
  alpha: 32                  # LoRA scaling
  dropout: 0.05
  target_modules:            # Which layers get LoRA
    - "q_proj"
    - "v_proj"
```

---

## ✅ Deliverables Checklist

Mapped to Problem Statement B5:

| # | Deliverable | Status | Implementation |
|---|------------|--------|----------------|
| 1 | **Continual learning strategy for VLA-based driving** | ✅ Complete | EWC + Replay + LoRA over frozen LLaVA-7B backbone |
| 2 | **Forgetting, retention, and transfer metrics** | ✅ Complete | BWT=-1.17%, FWT=+23.33%, Plasticity=98.46%, CDAR=0.9735 |
| 3 | **Parameter-efficient update mechanism** | ✅ Complete | LoRA adapters with 0.35% overhead (~40MB per domain) |
| 4 | **Benchmark protocol for region/season adaptation** | ✅ Complete | CADRE-Bench with 5 metrics across 4 sequential domains |
| — | **BDD100K dataset** | ✅ Used | `domain_us` (clear weather) and `domain_rainy` (adverse) |
| — | **nuScenes dataset** | ✅ Used | `domain_sg` (Singapore) and `domain_eu` (Boston) |

---

## 📚 References

1. Kirkpatrick, J. et al. (2017). "Overcoming catastrophic forgetting in neural networks." *PNAS*.
2. Hu, E.J. et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*.
3. Liu, H. et al. (2024). "Visual Instruction Tuning (LLaVA)." *NeurIPS*.
4. Yu, F. et al. (2020). "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning."
5. Caesar, H. et al. (2020). "nuScenes: A Multimodal Dataset for Autonomous Driving."
6. Rolnick, D. et al. (2019). "Experience Replay for Continual Learning." *NeurIPS*.

---

*Built with ❤️ using PyTorch, Hugging Face Transformers, and PEFT.*
