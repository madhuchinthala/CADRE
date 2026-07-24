# 🚗 CADRE — Continual Adaptation for Driving with Robust Evolution

> A continual learning framework for Vision-Language-Action (VLA) autonomous
> driving models that is *supposed* to learn new regions, regulations, and
> weather conditions without forgetting previously mastered environments.

This README was rewritten after auditing the actual source code (not just
the original documentation) at
`https://github.com/madhuchinthala/CADRE`. It keeps the useful setup
instructions from the original README, but adds a **Known Issues** section
that lists exactly what is broken, where, and why — so that an assistant
(e.g. GitHub Copilot) can be pointed directly at each problem instead of
guessing.

---

## 📋 Table of Contents

1. [What This Project Claims To Do](#what-this-project-claims-to-do)
2. [Actual Repository Structure](#actual-repository-structure)
3. [Setup Instructions](#setup-instructions)
4. [How the Pipeline Is *Supposed* to Run](#how-the-pipeline-is-supposed-to-run)
5. [🔴 Known Issues (for Copilot to fix)](#-known-issues-for-copilot-to-fix)
6. [References](#references)

---

## What This Project Claims To Do

CADRE is meant to combine:

- A **frozen LLaVA-v1.5-7B** backbone (Vision-Language-Action model)
- **LoRA adapters** for cheap per-domain fine-tuning
- **Elastic Weight Consolidation (EWC)** to reduce catastrophic forgetting
- **Experience replay** to retain old-domain competence
- A **domain router** that picks the right adapter at inference time
- A **CADRE-Bench** evaluation suite reporting BWT, FWT, Plasticity, and a
  composite CDAR score

The original README presented these headline numbers as results:

| Metric | Value |
|---|---|
| BWT (Backward Transfer) | -1.5% |
| FWT (Forward Transfer) | +27% |
| Plasticity | 97.1% |
| CDAR | 0.521 |
| Params per domain | 0.35% |

**These numbers are not reproducible from the current code** — see Issue #3
below. They appear to be hand-written placeholder/demo values, not measured
results.

---

## Actual Repository Structure

This is what is *actually* in the repo (verified by cloning it), not what
the old README claimed:

```
CADRE/
├── README.md
├── .gitignore
├── requirements.txt
├── setup.py
├── debug_image.py                 ← imports a module that does not exist (Issue #1)
│
├── configs/
│   ├── base_config.yaml
│   ├── lora_config.yaml
│   ├── ewc_config.yaml
│   ├── replay_config.yaml
│   ├── router_config.yaml
│   ├── heads_config.yaml
│   └── benchmark_config.yaml
│
├── src/
│   ├── models/vla_backbone.py          (PART 1)
│   ├── adapters/lora_adapter.py        (PART 2)
│   ├── continual/ewc.py                (PART 3)
│   ├── continual/replay_buffer.py      (PART 4)
│   ├── continual/continual_trainer.py  (combined loop — CLI is a stub, Issue #2)
│   ├── router/domain_router.py         (PART 5 — CLI is a stub, has a bug, Issues #2, #6)
│   ├── heads/integration_layer.py      (PART 6 — has NO CLI at all, Issue #4)
│   ├── benchmark/cadre_bench.py        (PART 7 — CLI only runs fake demo data, Issue #3)
│   └── utils/ (logger.py, checkpoint.py, visualization.py)
│
├── scripts/
│   ├── download_bdd100k.sh
│   ├── download_nuscenes.py
│   ├── download_llava.py          ← uses deprecated/removed HF Hub args (Issue #10)
│   ├── prepare_domains.py         ← imports a module that does not exist (Issue #1)
│   ├── run_training.py            ← imports a module that does not exist (Issue #1)
│   ├── run_pipeline.sh            ← correct per-domain dataset selection
│   └── run_pipeline.bat           ← WRONG dataset for 2 of 4 domains (Issue #5)
│
└── tests/
    ├── test_backbone.py
    ├── test_lora.py
    ├── test_ewc.py
    └── test_router.py
```

**IMPORTANT — there is no `src/data/` folder anywhere in the repository.**
The old README's "Project Structure" section documented a `src/data/`
package (`bdd100k_loader.py`, `nuscenes_loader.py`, `domain_splitter.py`,
`transforms.py`) that was **never actually committed**. This is the single
biggest problem in the project — see Issue #1.

`data/`, `checkpoints/`, `replay_buffer/`, and `outputs/` are intentionally
excluded from Git via `.gitignore` and must be created locally.

### Local data setup (confirmed working)

A local checkout of this project correctly shows the following, which is
**expected and correct** — none of this should ever be pushed to GitHub
because of dataset size:

```
CADRE/
├── cadre.egg-info/        ← created by `pip install -e .`
├── checkpoints/           ← gitignored
├── configs/
├── data/                  ← gitignored, populated locally
│   ├── bdd100k/
│   │   ├── bdd100k_labels_release/
│   │   ├── bdd100k_seg/
│   │   └── images/
│   └── nuscenes/
│       ├── maps/
│       ├── samples/
│       ├── sweeps/
│       ├── v1.0-mini/
│       ├── .v1.0-mini.txt
│       └── LICENSE
├── kaggle/                ← local Kaggle API credentials
├── outputs/               ← gitignored
├── replay_buffer/         ← gitignored
├── scripts/
└── src/
```

This confirms the **raw dataset files are present and downloaded
correctly** — that part of the setup works exactly as the README
describes. **The problem is not the data itself; it's that no code in
`src/` exists to read this data.** There is no `src/data/` package with
a `BDD100KDataset`, `NuScenesDataset`, or `DomainSplitter` class to turn
these downloaded folders into PyTorch `Dataset`/`DataLoader` objects. See
Issue #1 below — that issue is about missing *loader code*, not a missing
or incomplete dataset download.

---

## Setup Instructions

(Condensed from the original README — these steps are fine on their own,
the problems start once you try to actually *run* training. Full detail is
in the original README if you need copy-paste PowerShell commands.)

1. Clone the repo and `cd CADRE`
2. Install Python 3.10/3.11
3. Create and activate a venv
4. Install PyTorch matching your CUDA version (must be done **before**
   step 5 — see Issue #9 for why order matters)
5. `pip install -r requirements.txt` then `pip install -e .`
6. Create the local data/checkpoint/output folders (not tracked in Git)
7. Download BDD100K (Kaggle) and nuScenes (mini set)
8. Download the LLaVA-v1.5-7B model from Hugging Face
9. Run the unit tests in `tests/`
10. Attempt to run the training pipeline — **this is where it breaks**
    (see below)

---

## How the Pipeline Is *Supposed* to Run

The original README instructs users to run, in order:

```bash
python -m src.models.vla_backbone --model_path checkpoints/llava-v1.5-7b --verify
python -m src.continual.continual_trainer --config configs/base_config.yaml --domain domain_us --dataset bdd100k --ewc_lambda 5000 --replay_ratio 0.3 --epochs 10
python -m src.router.domain_router --config configs/router_config.yaml --domains domain_us,domain_sg,domain_eu,domain_rainy --epochs 20
python -m src.benchmark.cadre_bench --config configs/benchmark_config.yaml --domains domain_us,domain_sg,domain_eu,domain_rainy --output_dir outputs/cadre_bench
```

Only the **first** command does what it says. The rest either do nothing
useful or produce fabricated output. Details below.

---

## 🔴 Known Issues (for Copilot to fix)

These are listed in rough order of severity. File paths and line-level
descriptions are given so an AI coding assistant can jump straight to the
problem.

### Issue #1 — `src/data/` package is missing entirely (blocks all real training)
**Severity: Critical.**
**Note: this is not a missing-dataset problem.** BDD100K and nuScenes are
correctly downloaded to `data/bdd100k/` and `data/nuscenes/` locally (this
is confirmed working and is properly excluded from Git via `.gitignore`
because of size). The gap is purely on the code side: nothing in `src/`
reads those folders into a PyTorch `Dataset`. Three different files
import from a `src/data/` package that does not exist anywhere in the
repository:
- `debug_image.py` → `from src.data.bdd100k_dataset import BDD100KDataset`
- `scripts/run_training.py` → `from src.data.dataloader import get_dataloader`
- `scripts/prepare_domains.py` → `from src.data.domain_splitter import DomainSplitter` and `from src.data.transforms import DrivingTransforms`

Note these three files even disagree on the module name
(`bdd100k_dataset` vs `dataloader` vs `domain_splitter`/`transforms`),
suggesting the data layer was either deleted or never pushed, and that the
three files were written independently against an assumed (and different)
interface. **Every script that actually tries to load data will raise
`ModuleNotFoundError: No module named 'src.data'`.**
Fix needed: implement `src/data/__init__.py`, a BDD100K dataset loader, a
nuScenes dataset loader, a domain splitter (that applies the
`split_criteria` blocks in `configs/base_config.yaml`), and image
transforms — then make the three consumer files agree on one shared
interface.

### Issue #2 — The "training" CLI commands don't train anything
**Severity: Critical.**
- `src/continual/continual_trainer.py`, `if __name__ == "__main__":` block:
  parses `--domain`, `--dataset`, `--ewc_lambda`, etc., and just **prints**
  them back out. It never constructs a `VLABackbone`, never injects LoRA,
  never builds a dataset/dataloader, and never calls
  `ContinualTrainer.train_domain()`.
- `src/router/domain_router.py`, `if __name__ == "__main__":` block: builds
  a `DomainRouter` and prints its parameter count. It never trains
  (`RouterTrainer.train_epoch()` is never called).

Anyone following the README's "Step 11 — Run the Pipeline" instructions
will see log output and assume training happened, but no weights are ever
updated or saved. The real training logic exists (in the classes), it's
just never wired up to the command line.

### Issue #3 — CADRE-Bench's CLI reports hand-typed fake numbers, not real evaluation
**Severity: Critical / misleading.**
In `src/benchmark/cadre_bench.py`, the `__main__` block literally contains:
```python
print("Running CADRE-Bench demo with synthetic performance data...\n")
...
demo_matrix = np.array([
    [0.94, 0.32, 0.28, 0.25],
    [0.93, 0.95, 0.41, 0.38],
    [0.92, 0.94, 0.96, 0.52],
    [0.925, 0.935, 0.955, 0.97],
])[:T, :T]
bench.perf_matrix = demo_matrix
```
Running `python -m src.benchmark.cadre_bench ...` never touches a trained
model at all — it computes BWT/FWT/Plasticity/CDAR from these hardcoded
numbers. This is exactly where the README's "Key Results" table (BWT
-1.5%, FWT +27%, CDAR 0.521, etc.) comes from. These are **not measured
results from this codebase** — they should not be presented as achieved
results until a real `run_evaluation()` pass over trained models produces
them.

### Issue #4 — `src/heads/integration_layer.py` has no CLI, but the pipeline scripts call it as one
**Severity: High.**
Both `scripts/run_pipeline.sh` and `scripts/run_pipeline.bat` include a
"Part 6" step:
```
python -m src.heads.integration_layer --config configs/heads_config.yaml --heads waypoint,hazard,regulation,weather --epochs 15
```
But `integration_layer.py` contains only `nn.Module` class definitions
(`WaypointHead`, `HazardHead`, `RegulationHead`, `WeatherHead`,
`IntegrationLayer`, `MultiHeadDrivingModel`) — there is no `argparse`, no
`if __name__ == "__main__":` block, and no training loop. Running this
command does not error, it just silently does nothing and exits with code
0, so a user watching the pipeline run would see no failure, just no
progress either. Needs a `HeadsTrainer` class and a CLI entry point.

### Issue #5 — `run_pipeline.bat` (Windows) trains the wrong dataset for 2 of 4 domains
**Severity: High.**
`configs/base_config.yaml` and `scripts/run_pipeline.sh` (the Linux
script) correctly say `domain_sg` and `domain_eu` should be trained on
`nuscenes`, while `domain_us` and `domain_rainy` use `bdd100k`. The Linux
script implements this correctly with a conditional. But
`scripts/run_pipeline.bat` hardcodes `--dataset bdd100k` for **every**
domain in its loop, including `domain_sg` and `domain_eu`. Since the whole
README is written for Windows/PowerShell users, this is the version most
people following the instructions will actually run — and it silently
trains the wrong data for half the domains.

### Issue #6 — `DomainRouter.route()` always logs sample 0's confidence, not the actual low-confidence sample
**Severity: Medium (silent, misleading logs).**
In `src/router/domain_router.py`:
```python
for pred, conf in zip(predictions, confidences):
    domain_idx = pred.item()
    if conf.item() >= self.confidence_threshold:
        domain_names.append(self.domain_labels[domain_idx])
    else:
        top2 = probs[0].topk(2)   # <-- BUG: always index 0, not the current sample
        logger.warning(...)
        domain_names.append(self.domain_labels[domain_idx])
```
For any batch with more than one sample, every "low confidence routing"
warning reports probabilities from `probs[0]` regardless of which sample
in the batch actually triggered it. Needs `enumerate()` over the batch and
use of the per-sample index instead of a hardcoded `0`.

### Issue #7 — Advertised "0.35% params per domain / ~50MB adapter" doesn't match the default LoRA config
**Severity: Medium (documentation/config mismatch).**
`configs/lora_config.yaml` only enables `q_proj` and `v_proj` by default
(the other four target modules — `k_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj` — are commented out). With `rank: 16` on just those
2 modules across a ~7B-parameter LLaMA backbone, the actual trainable
parameter count works out to roughly 8.4M (~0.12%), not the ~24.72M
(0.35%) figure written in the same config file's comments and repeated in
the README's results table. Either the default `target_modules` list
needs to be expanded to match the advertised number, or the advertised
number needs to be corrected to match the default config.

### Issue #8 — "domain_eu" is not actually European driving data
**Severity: Medium (data/labeling correctness, not just code).**
In `configs/base_config.yaml`:
```yaml
- name: "domain_eu"
  dataset: "nuscenes"
  description: "European urban driving (Boston mapped to EU-style)"
  split_criteria:
    location: ["boston-seaport"]
```
The nuScenes dataset only contains driving data captured in **Boston,
USA** and **Singapore** — it has no European city data at all. Labeling
Boston-sourced samples as "domain_eu" / "European urban driving" is
factually incorrect, not just a naming quirk, since any claims about
cross-region generalization (US → Europe) in this project are actually
US → US (different neighborhood) comparisons.

### Issue #9 — `setup.py` lists `torch` as an install dependency, risking the CUDA install from Step 4
**Severity: Low/Medium.**
The README correctly warns that PyTorch with the right CUDA build must be
installed manually before anything else. But `setup.py` still declares
`"torch>=2.1.0"` in `install_requires`. Running `pip install -e .`
(Step 5) invokes pip's dependency resolver against that constraint, which
can pull in a different (e.g. CPU-only or mismatched-CUDA) torch wheel in
some environments/pip versions, quietly undoing the manual CUDA install.
Torch should either be removed from `install_requires` (since the README
already handles it as a manual prerequisite) or pinned/commented with a
clear explanation.

### Issue #10 — `download_llava.py` uses Hugging Face Hub arguments that are deprecated/removed in newer versions
**Severity: Low/Medium (will break on a future `pip install`).**
```python
snapshot_download(
    repo_id=args.model_id,
    local_dir=str(output_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)
```
`requirements.txt` only pins a floor (`huggingface_hub>=0.23.0`) with no
upper bound. Both `local_dir_use_symlinks` and `resume_download` have been
deprecated in newer `huggingface_hub` releases and are removed entirely in
some versions, which will raise a `TypeError` on a fresh install. Needs
either a version ceiling in `requirements.txt` or updated `snapshot_download`
call arguments.

### Issue #11 — `EWC.compute_fisher()` counts batches, not samples, despite the config name
**Severity: Low (naming/behavior mismatch, not a crash).**
`configs/ewc_config.yaml` sets `fisher_n_samples: 2000`, described as
"Number of samples to use for Fisher Information computation." But in
`src/continual/ewc.py`:
```python
for batch in tqdm(dataloader, ...):
    if n_samples >= self.fisher_n_samples:
        break
    ...
    n_samples += 1   # incremented once per BATCH, not per sample
```
With `batch_size=4` (the project default), this actually processes up to
8,000 individual samples, not 2,000, before stopping. Either rename the
config key to reflect that it counts batches, or update the loop to break
based on the number of individual samples actually processed.

---

## References

- **BDD100K**: Yu et al., "BDD100K: A Diverse Driving Dataset for
  Heterogeneous Multitask Learning" (CVPR 2020)
- **nuScenes**: Caesar et al., "nuScenes: A Multimodal Dataset for
  Autonomous Driving" (CVPR 2020)
- **LLaVA**: Liu et al., "Visual Instruction Tuning" (NeurIPS 2023)
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language
  Models" (ICLR 2022)
- **EWC**: Kirkpatrick et al., "Overcoming Catastrophic Forgetting in
  Neural Networks" (PNAS 2017)

---

## License

Academic research use only. Dataset usage is governed by BDD100K's
BSD 3-Clause license and nuScenes' CC BY-NC-SA 4.0 terms.
