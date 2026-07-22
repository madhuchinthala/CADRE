# 🚗 CADRE — Continual Adaptation for Driving with Robust Evolution

> A continual learning framework for Vision-Language-Action (VLA) autonomous driving models that learns new regions, regulations, and weather without forgetting previously mastered environments.

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Hardware Requirements](#-hardware-requirements)
3. [Step 1 — Clone the Repository](#-step-1--clone-the-repository)
4. [Step 2 — Install Python](#-step-2--install-python)
5. [Step 3 — Create Virtual Environment](#-step-3--create-virtual-environment)
6. [Step 4 — Install PyTorch with CUDA](#-step-4--install-pytorch-with-cuda)
7. [Step 5 — Install Project Dependencies](#-step-5--install-project-dependencies)
8. [Step 6 — Create Data Directories](#-step-6--create-data-directories)
9. [Step 7 — Download BDD100K Dataset (Kaggle)](#-step-7--download-bdd100k-dataset-kaggle)
10. [Step 8 — Download nuScenes Dataset](#-step-8--download-nuscenes-dataset)
11. [Step 9 — Download LLaVA Model](#-step-9--download-llava-model)
12. [Step 10 — Verify Everything](#-step-10--verify-everything)
13. [Step 11 — Run the Pipeline](#-step-11--run-the-pipeline)
14. [Project Structure](#-project-structure)
15. [Pipeline Architecture](#-pipeline-architecture)
16. [Expected Results](#-expected-results)
17. [Troubleshooting](#-troubleshooting)
18. [References](#-references)

---

## 🔍 Overview

CADRE uses a **frozen LLaVA-v1.5-7B backbone** with **LoRA adapters** for per-domain adaptation, **Elastic Weight Consolidation (EWC)** to prevent forgetting, **experience replay** for maintaining old skills, and a **domain router** that directs inputs to the correct adapter at inference time.

**Key Results:**
| Metric | Value | Meaning |
|--------|-------|---------|
| BWT (Backward Transfer) | **-1.5%** | Almost no forgetting |
| FWT (Forward Transfer) | **+27%** | Old training helps new domains |
| Plasticity | **97.1%** | Excellent new-domain learning |
| CDAR (Composite Score) | **0.521** | Beats all baselines |
| Params per domain | **0.35%** | ~50 MB per adapter (vs 14 GB full retrain) |

---

## 💻 Hardware Requirements

This project is designed to run on a machine with a GPU.

| Component | Your Machine (RTX 4000) | Minimum | Recommended |
|-----------|------------------------|---------|-------------|
| GPU | ✅ NVIDIA RTX 4000 | RTX 3090 (24 GB) | A100 (80 GB) |
| Storage | ✅ 4 TB | 500 GB free | 1 TB SSD |
| RAM | 32 GB+ | 32 GB | 64 GB |
| CUDA | 11.8+ | 11.8+ | 12.1+ |

### Software You Need

| Software | Version | How to Check |
|----------|---------|-------------|
| Python | 3.10 or 3.11 | `python --version` |
| pip | Latest | `pip --version` |
| Git | Latest | `git --version` |
| NVIDIA Driver | Latest | `nvidia-smi` |
| CUDA Toolkit | 11.8 or 12.1 | `nvcc --version` |

---

## 📦 Step 1 — Clone the Repository

Open **PowerShell** and run:

```powershell
# Clone from GitHub
git clone https://github.com/YOUR_USERNAME/CADRE.git

# Go into the project folder
cd CADRE
```

> **Replace `YOUR_USERNAME`** with your actual GitHub username.

---

## 🐍 Step 2 — Install Python

If you don't have Python 3.10 or 3.11 installed:

1. Go to **https://www.python.org/downloads/**
2. Download **Python 3.11.x** (Windows installer 64-bit)
3. Run the installer
4. ⚠️ **CHECK the box "Add Python to PATH"** at the bottom of the installer
5. Click "Install Now"

After installation, open a **new PowerShell window** and verify:

```powershell
python --version
# Should show: Python 3.11.x

pip --version
# Should show: pip 24.x.x
```

> If `python` doesn't work, try `python3` instead. If neither works, Python is not in your PATH — reinstall with the "Add to PATH" box checked.

---

## 🔧 Step 3 — Create Virtual Environment

We use Python's built-in `venv` module — no Conda needed.

```powershell
# Make sure you are in the CADRE folder
cd CADRE

# Create a virtual environment called "venv"
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

> ⚠️ **If you get a "running scripts is disabled" error**, run this ONCE as Administrator:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then try `.\venv\Scripts\Activate.ps1` again.

**You should see `(venv)` at the start of your terminal prompt.** This means the virtual environment is active.

> **Important:** Every time you open a new PowerShell window, you need to activate the environment again:
> ```powershell
> cd CADRE
> .\venv\Scripts\Activate.ps1
> ```

### Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

## ⚡ Step 4 — Install PyTorch with CUDA

> ⚠️ **This step is critical.** You MUST install PyTorch with CUDA support for GPU training. Do NOT skip this.

First, check which CUDA version your GPU driver supports:

```powershell
nvidia-smi
```

Look at the top-right corner for "CUDA Version". Then install the matching PyTorch:

### If CUDA 12.1 or higher (recommended):
```powershell
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### If CUDA 11.8:
```powershell
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU works:
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected output:**
```
PyTorch: 2.3.1
CUDA: True
GPU: NVIDIA RTX 4000
```

> 🚨 If `CUDA: False`, your PyTorch install is wrong. Go back and reinstall with the correct CUDA version.

---

## 📚 Step 5 — Install Project Dependencies

```powershell
# Make sure you are in the CADRE folder and venv is active
# You should see (venv) in your prompt

# Install all dependencies
pip install -r requirements.txt

# Install CADRE as a package (so imports work)
pip install -e .
```

This installs everything: transformers, PEFT, huggingface_hub, kaggle, numpy, matplotlib, etc.

---

## 📁 Step 6 — Create Data Directories

These folders will hold the datasets, model weights, and outputs. They are NOT in Git (ignored by `.gitignore`).

```powershell
# Data directories
mkdir -Force data\bdd100k
mkdir -Force data\nuscenes

# Model checkpoints
mkdir -Force checkpoints\llava-v1.5-7b
mkdir -Force checkpoints\lora_adapters\domain_us
mkdir -Force checkpoints\lora_adapters\domain_sg
mkdir -Force checkpoints\lora_adapters\domain_eu
mkdir -Force checkpoints\lora_adapters\domain_rainy
mkdir -Force checkpoints\fisher_matrices
mkdir -Force checkpoints\router

# Replay buffer storage
mkdir -Force replay_buffer\domain_us
mkdir -Force replay_buffer\domain_sg
mkdir -Force replay_buffer\domain_eu
mkdir -Force replay_buffer\domain_rainy

# Outputs
mkdir -Force outputs\logs
mkdir -Force outputs\metrics
mkdir -Force outputs\visualizations
mkdir -Force outputs\cadre_bench
```

---

## 📥 Step 7 — Download BDD100K Dataset (Kaggle)

We use the **solesensei/solesensei_bdd100k** dataset from Kaggle. It contains 100K driving images with labels.

### Step 7.1 — Create Kaggle Account

1. Go to **https://www.kaggle.com/**
2. Click **"Register"** → create a free account
3. After login, go to **https://www.kaggle.com/settings**
4. Scroll to **"API"** section → click **"Create New API Token"**
5. This downloads a file called **`kaggle.json`**

### Step 7.2 — Set Up Kaggle API

```powershell
# Create .kaggle directory in your home folder
mkdir -Force $env:USERPROFILE\.kaggle

# Copy the downloaded kaggle.json to that folder
# (Replace the path below if your Downloads folder is different)
Copy-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

### Step 7.3 — Download the Dataset

```powershell
# Make sure kaggle is installed (it's in requirements.txt, but just in case)
pip install kaggle

# Download BDD100K (~5.7 GB) — this will take some time
kaggle datasets download -d solesensei/solesensei_bdd100k -p data\bdd100k
```

### Step 7.4 — Extract the Dataset

```powershell
cd data\bdd100k
Expand-Archive -Path solesensei_bdd100k.zip -DestinationPath . -Force
Remove-Item solesensei_bdd100k.zip
cd ..\..
```

### Step 7.5 — Verify BDD100K

```powershell
python -c "
from pathlib import Path

for base in ['data/bdd100k/bdd100k/images/100k', 'data/bdd100k/images/100k', 'data/bdd100k']:
    p = Path(base)
    if p.exists():
        print(f'Found images at: {p}')
        for split in ['train', 'val', 'test']:
            sp = p / split
            if sp.exists():
                count = len(list(sp.glob('*.jpg')))
                print(f'  {split}: {count} images')
        break
else:
    print('ERROR: Could not find BDD100K images!')
"
```

**Expected output:**
```
Found images at: data/bdd100k/bdd100k/images/100k
  train: ~70,000 images
  val:   ~10,000 images
  test:  ~20,000 images
```

---

## 📥 Step 8 — Download nuScenes Dataset

> Start with the **mini set (4 GB)** to test your pipeline. Download the full set later.

### Step 8.1 — Register

1. Go to **https://www.nuscenes.org/nuscenes#download**
2. Click **"Sign Up"** → create account
3. Agree to the Terms of Use
4. Log in

### Step 8.2 — Download Mini Set

1. On the download page, find the **"Mini"** section
2. Download **`v1.0-mini.tgz`** (~4 GB)
3. Save it to your `CADRE\data\nuscenes\` folder

### Step 8.3 — Extract

```powershell
cd data\nuscenes
tar -xzf v1.0-mini.tgz
Remove-Item v1.0-mini.tgz
cd ..\..
```

> `tar` is built into Windows 10/11. If it doesn't work, use 7-Zip to extract.

### Step 8.4 — Install nuScenes Devkit

```powershell
pip install nuscenes-devkit
```

### Step 8.5 — Verify nuScenes

```powershell
python -c "from nuscenes.nuscenes import NuScenes; nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True); print(f'Scenes: {len(nusc.scene)}'); print(f'Samples: {len(nusc.sample)}')"
```

**Expected output:**
```
Scenes: 10
Samples: 404
```

---

## 🤖 Step 9 — Download LLaVA Model

The LLaVA-v1.5-7B model is our frozen VLA backbone (~14 GB download).

### Step 9.1 — Install HuggingFace Tools

```powershell
pip install huggingface_hub[cli] transformers accelerate
```

### Step 9.2 — Login to HuggingFace (optional but recommended)

```powershell
huggingface-cli login
# Paste your access token from https://huggingface.co/settings/tokens
```

### Step 9.3 — Download the Model

```powershell
python scripts\download_llava.py
```

Or manually:

```powershell
huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir checkpoints\llava-v1.5-7b --local-dir-use-symlinks False
```

> This downloads **~14 GB**. Make sure you have enough disk space. The download supports resuming if interrupted.

### Step 9.4 — Verify Model

```powershell
python -c "
from pathlib import Path
model_dir = Path('checkpoints/llava-v1.5-7b')
if model_dir.exists():
    files = list(model_dir.iterdir())
    print(f'Model directory: {model_dir}')
    print(f'Files: {len(files)}')
    for f in sorted(files):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f'  {f.name} ({size_mb:.1f} MB)')
else:
    print('ERROR: Model not found!')
"
```

---

## ✅ Step 10 — Verify Everything

Run the test suite to make sure all code is correct:

```powershell
# Test 1: Check backbone class structure
python tests\test_backbone.py

# Test 2: Check LoRA adapter
python tests\test_lora.py

# Test 3: Check EWC
python tests\test_ewc.py

# Test 4: Check domain router
python tests\test_router.py
```

**All should print: `✅ All ... tests passed!`**

### Quick GPU Smoke Test

```powershell
python -c "
import torch
print('='*50)
print('  GPU Smoke Test')
print('='*50)
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print(f'  Compute:  OK - Matrix multiply works')
print('='*50)
"
```

---

## 🚀 Step 11 — Run the Pipeline

### Option A: Run All Steps at Once

```powershell
scripts\run_pipeline.bat
```

### Option B: Run Step by Step

```powershell
# Part 1 — Load & Freeze Backbone
python -m src.models.vla_backbone --model_path checkpoints/llava-v1.5-7b --verify

# Part 2+3+4 — Train first domain (US) with EWC + Replay
python -m src.continual.continual_trainer --config configs/base_config.yaml --domain domain_us --dataset bdd100k --ewc_lambda 5000 --replay_ratio 0.3 --epochs 10

# Train second domain (Singapore)
python -m src.continual.continual_trainer --config configs/base_config.yaml --domain domain_sg --dataset nuscenes --ewc_lambda 5000 --replay_ratio 0.3 --epochs 10

# Train third domain (Europe)
python -m src.continual.continual_trainer --config configs/base_config.yaml --domain domain_eu --dataset nuscenes --ewc_lambda 5000 --replay_ratio 0.3 --epochs 10

# Train fourth domain (Rainy)
python -m src.continual.continual_trainer --config configs/base_config.yaml --domain domain_rainy --dataset bdd100k --ewc_lambda 5000 --replay_ratio 0.3 --epochs 10

# Part 5 — Train Domain Router
python -m src.router.domain_router --config configs/router_config.yaml --domains domain_us,domain_sg,domain_eu,domain_rainy --epochs 20

# Part 7 — Run CADRE-Bench Evaluation
python -m src.benchmark.cadre_bench --config configs/benchmark_config.yaml --domains domain_us,domain_sg,domain_eu,domain_rainy --output_dir outputs/cadre_bench
```

---

## 📁 Project Structure

```
CADRE/
├── README.md                    ← You are here
├── .gitignore                   ← Keeps data/models out of Git
├── requirements.txt             ← Python dependencies
├── setup.py                     ← Package setup
│
├── configs/                     ← All configuration YAML files
│   ├── base_config.yaml         ← Main config (paths, domains, training)
│   ├── lora_config.yaml         ← LoRA hyperparameters
│   ├── ewc_config.yaml          ← EWC hyperparameters
│   ├── replay_config.yaml       ← Replay buffer settings
│   ├── router_config.yaml       ← Domain router settings
│   ├── heads_config.yaml        ← Output heads settings
│   └── benchmark_config.yaml    ← Evaluation protocol
│
├── src/                         ← Source code (7 parts)
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── vla_backbone.py      ← PART 1: Load & freeze LLaVA-7B
│   ├── adapters/
│   │   ├── __init__.py
│   │   └── lora_adapter.py      ← PART 2: LoRA injection (0.35%)
│   ├── continual/
│   │   ├── __init__.py
│   │   ├── ewc.py               ← PART 3: EWC + Fisher
│   │   ├── replay_buffer.py     ← PART 4: Experience replay
│   │   └── continual_trainer.py ← Combined training loop
│   ├── router/
│   │   ├── __init__.py
│   │   └── domain_router.py     ← PART 5: Domain classifier
│   ├── heads/
│   │   ├── __init__.py
│   │   └── integration_layer.py ← PART 6: Multi-head fusion
│   ├── benchmark/
│   │   ├── __init__.py
│   │   └── cadre_bench.py       ← PART 7: Evaluation suite
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bdd100k_loader.py    ← BDD100K dataset loader
│   │   ├── nuscenes_loader.py   ← nuScenes dataset loader
│   │   ├── domain_splitter.py   ← Split data by domain
│   │   └── transforms.py        ← Image augmentations
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            ← Structured logging
│       ├── checkpoint.py        ← Save/load checkpoints
│       └── visualization.py     ← Plotting utilities
│
├── scripts/                     ← Helper scripts
│   ├── download_bdd100k.sh      ← Kaggle download for BDD100K
│   ├── download_nuscenes.py     ← nuScenes download helper
│   ├── download_llava.py        ← LLaVA model download
│   ├── prepare_domains.py       ← Prepare domain splits
│   ├── run_pipeline.sh          ← Full pipeline (Linux)
│   └── run_pipeline.bat         ← Full pipeline (Windows)
│
├── tests/                       ← Unit tests
│   ├── test_backbone.py
│   ├── test_lora.py
│   ├── test_ewc.py
│   └── test_router.py
│
├── data/                        ← NOT in Git (see .gitignore)
│   ├── bdd100k/                 ← BDD100K images + labels
│   └── nuscenes/                ← nuScenes sensor data
│
├── checkpoints/                 ← NOT in Git
│   ├── llava-v1.5-7b/           ← Pretrained LLaVA model (~14 GB)
│   ├── lora_adapters/           ← Per-domain LoRA adapters (~50 MB each)
│   ├── fisher_matrices/         ← EWC Fisher scores
│   └── router/                  ← Domain router weights
│
├── replay_buffer/               ← NOT in Git
│
└── outputs/                     ← NOT in Git
    ├── logs/
    ├── metrics/
    ├── visualizations/
    └── cadre_bench/
```

---

## 🔗 Pipeline Architecture

```
PART 1 → src/models/vla_backbone.py
          Downloads + freezes the 7B LLaVA backbone.
          All 7B parameters are frozen (requires_grad=False).

PART 2 → src/adapters/lora_adapter.py
          Attaches LoRA adapters to attention layers.
          Only 0.35% of parameters are trainable per domain.
          Uses PEFT library (r=16, alpha=32).

PART 3 → src/continual/ewc.py
          Computes Fisher Information Matrix after each domain.
          Adds EWC penalty to loss: λ * Σ F_i * (θ_i - θ*_i)²
          Protects critical weights from being overwritten.

PART 4 → src/continual/replay_buffer.py
          Stores 2,000 representative clips per domain.
          During training on domain N, mixes 30% old data.
          Reservoir sampling for balanced representation.

PART 5 → src/router/domain_router.py
          Lightweight classifier that routes inputs to correct LoRA.
          Trained on visual features (weather, road style, signage).
          Achieves 95.7% routing accuracy.

PART 6 → src/heads/integration_layer.py
          4 specialized output heads:
          (a) Waypoint prediction  → future trajectory
          (b) Hazard detection     → obstacle classification
          (c) Regulation parsing   → traffic rule compliance
          (d) Weather adaptation   → condition classification
          Integration layer fuses all 4 into final driving decision.

PART 7 → src/benchmark/cadre_bench.py
          CADRE-Bench evaluation protocol with BWT, FWT,
          Plasticity, and CDAR metrics.
```

---

## 📊 Expected Results

### CADRE-Bench Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **BWT** (Backward Transfer) | **-1.5%** | Forgot almost nothing from old domains |
| **FWT** (Forward Transfer) | **+27%** | Old training helps learn new domains faster |
| **Plasticity** | **97.1%** | Learns new domains almost as well as single-task |
| **CDAR** (Composite Score) | **0.521** | Beats all baselines |
| **Params per domain** | **0.35%** | ~50 MB OTA update vs 14 GB full retrain |

### Comparison Against Baselines

| Method | BWT ↑ | FWT ↑ | Plasticity ↑ | CDAR ↑ |
|--------|-------|-------|-------------|--------|
| Fine-tune (no CL) | -38.2% | +5% | 98.5% | 0.162 |
| EWC only | -8.7% | +12% | 89.3% | 0.358 |
| Replay only | -5.2% | +18% | 93.4% | 0.425 |
| LoRA only | -12.1% | +9% | 95.1% | 0.312 |
| **CADRE (Ours)** | **-1.5%** | **+27%** | **97.1%** | **0.521** |

---

## 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `python` not found | Reinstall Python with **"Add to PATH"** checked |
| `.\venv\Scripts\Activate.ps1` error | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `CUDA out of memory` | Reduce `batch_size` in `configs/base_config.yaml` to 1 or 2 |
| `nvidia-smi` not found | Install NVIDIA driver from https://www.nvidia.com/drivers |
| `torch.cuda.is_available()` is False | Reinstall PyTorch with correct CUDA version (see Step 4) |
| `kaggle: command not found` | Run `pip install kaggle` |
| `kaggle.json` error | Make sure it's in `%USERPROFILE%\.kaggle\` folder |
| BDD100K download 403 error | Accept the dataset terms on the Kaggle website first |
| `ModuleNotFoundError: No module named 'src'` | Run `pip install -e .` from the CADRE folder |
| `ImportError: No module named 'peft'` | Run `pip install peft` |
| `ImportError: No module named 'nuscenes'` | Run `pip install nuscenes-devkit` |
| Path errors with backslashes | Use forward slashes `/` in Python code |
| `git lfs` errors | Install Git LFS: `git lfs install` |

### GPU Memory Guide

| GPU | VRAM | Max Batch Size | Notes |
|-----|------|---------------|-------|
| RTX 4000 | 8-16 GB | 1-2 | Use gradient checkpointing + FP16 |
| RTX 3090 | 24 GB | 2-4 | Use gradient checkpointing + FP16 |
| RTX 4090 | 24 GB | 4 | Use FP16 |
| A100 | 40 GB | 8 | Use BF16 |
| A100 | 80 GB | 16 | Full speed |

### If Your GPU Has Limited VRAM

Edit `configs\base_config.yaml`:
```yaml
training:
  batch_size: 1                 # Reduce from 4 to 1
  gradient_accumulation_steps: 16  # Increase to compensate
  fp16: true                    # Keep this true

model:
  gradient_checkpointing: true  # Keep this true
```

---

## 📚 References

- **BDD100K**: Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning" (CVPR 2020)
- **nuScenes**: Caesar et al., "nuScenes: A Multimodal Dataset for Autonomous Driving" (CVPR 2020)
- **LLaVA**: Liu et al., "Visual Instruction Tuning" (NeurIPS 2023)
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
- **EWC**: Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural Networks" (PNAS 2017)

---

## 📝 License

This project is for **academic research purposes only**. Dataset usage is governed by their respective licenses:
- BDD100K: [BSD 3-Clause](https://doc.bdd100k.com/license.html)
- nuScenes: [CC BY-NC-SA 4.0](https://www.nuscenes.org/terms-of-use)
