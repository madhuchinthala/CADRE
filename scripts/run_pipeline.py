#!/usr/bin/env python
"""
CADRE — Resumable Full Training Pipeline
==========================================
Runs all pipeline stages in order (backbone verify -> 4 domain trainings ->
router -> heads -> benchmark) and remembers, on disk, which stages already
finished successfully.

Why this exists
----------------
`continual_trainer.py` already checkpoints *inside* a single domain's
training: if you kill it mid-training on domain_us, it saves an
epoch-level checkpoint and resumes from the next epoch next time it's
invoked for that domain. But the *pipeline* itself (run_pipeline.sh/.bat)
had no memory across stages — if you stopped the whole pipeline after
domain_us finished and reran it, it started again from stage 1 and
retrained domain_us from scratch.

This script fixes that at the pipeline level:
  - Every stage that finishes successfully is recorded in a small JSON
    state file (checkpoints/pipeline_state.json).
  - Stages already marked complete are skipped on the next run.
  - If the pipeline is stopped mid-stage (Ctrl+C, crash, power loss,
    OOM, etc.), that stage is simply NOT marked complete, so the next
    run re-enters it. For domain training stages, re-entering means
    calling continual_trainer.py again, which resumes from the last
    completed epoch on its own (unchanged, existing behavior).
  - State is written atomically (write to a temp file, then rename) so
    a crash in the middle of saving state can never corrupt the file.

Usage
-----
    # Run (or resume) the full pipeline
    python scripts/run_pipeline.py

    # See what's done and what's left, without running anything
    python scripts/run_pipeline.py --status

    # Force one stage to re-run even though it's marked complete
    python scripts/run_pipeline.py --redo domain_sg

    # Wipe all pipeline progress and start over from stage 1
    python scripts/run_pipeline.py --reset

    # Only run one specific stage (still respects/updates the state file)
    python scripts/run_pipeline.py --only domain_us

Stopping and resuming
----------------------
Just stop it however you need to (Ctrl+C, closing the terminal, killing
the process, the machine losing power). Whatever stage was running is
left unmarked. Run the exact same command again later:

    python scripts/run_pipeline.py

...and it will skip every stage that already finished, and continue
from the stage that was interrupted (resuming mid-epoch for domain
training, since that part already checkpoints on its own).
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run_pipeline")

# Repo root = parent of this script's directory (scripts/..)
REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / "checkpoints" / "pipeline_state.json"

DEFAULT_DOMAINS = ["domain_us", "domain_sg", "domain_eu", "domain_rainy"]
DOMAIN_DATASETS = {
    "domain_us": "bdd100k",
    "domain_sg": "nuscenes",
    "domain_eu": "nuscenes",
    "domain_rainy": "bdd100k",
}


# ──────────────────────────────────────────────────────────
# Pipeline state (the actual "checkpoint" for the pipeline)
# ──────────────────────────────────────────────────────────

def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {"completed_stages": [], "history": []}
    try:
        with open(STATE_PATH, "r") as f:
            data = json.load(f)
        data.setdefault("completed_stages", [])
        data.setdefault("history", [])
        return data
    except Exception as e:
        logger.warning(
            f"⚠️  Could not read {STATE_PATH} ({e}). "
            f"Treating pipeline as if nothing has completed yet. "
            f"(The file was not deleted — inspect/fix it manually if needed.)"
        )
        return {"completed_stages": [], "history": []}


def _save_state(state: dict) -> None:
    """Write state atomically: write to a temp file in the same directory,
    then os.replace() it over the real path. This means a crash or kill
    signal during the write can never leave a half-written/corrupt state
    file behind — the old file stays valid until the new one is fully
    written and swapped in."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(STATE_PATH.parent), prefix=".pipeline_state_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, STATE_PATH)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def mark_complete(state: dict, stage_name: str) -> None:
    if stage_name not in state["completed_stages"]:
        state["completed_stages"].append(stage_name)
    state["history"].append(
        {"stage": stage_name, "status": "completed", "at": datetime.now().isoformat()}
    )
    _save_state(state)


def mark_started(state: dict, stage_name: str) -> None:
    state["history"].append(
        {"stage": stage_name, "status": "started", "at": datetime.now().isoformat()}
    )
    _save_state(state)


def is_complete(state: dict, stage_name: str) -> bool:
    return stage_name in state["completed_stages"]


# ──────────────────────────────────────────────────────────
# Stage definitions
# ──────────────────────────────────────────────────────────

@dataclass
class Stage:
    name: str                 # unique id, stored in state file
    label: str                # human-readable description
    command: List[str]        # argv to run
    check: Optional[Callable[[], bool]] = None
    # ^ optional extra sanity check: if the stage is marked complete in the
    #   state file but its expected output is missing (e.g. someone deleted
    #   checkpoints/ by hand), we warn and re-run it instead of trusting a
    #   stale state file blindly.


def build_stages(args) -> List[Stage]:
    py = sys.executable  # use the same interpreter this script was launched with
    stages: List[Stage] = []

    # ── Stage: backbone verify ──
    stages.append(
        Stage(
            name="verify_backbone",
            label="Load & freeze VLA backbone (verify)",
            command=[
                py, "-m", "src.models.vla_backbone",
                "--model_path", args.model_path,
                "--verify",
            ],
        )
    )

    # ── Stages: one per domain ──
    # NOTE: continual_trainer.py's actual CLI (see its __main__ block) only
    # accepts --config, --domain, --epochs, --device, --resume/--no-resume,
    # --max-samples-per-epoch, --max-val-samples. It does NOT accept
    # --dataset, --ewc_lambda, or --replay_ratio (those are handled
    # internally / via the config file), so we must not pass them here.
    for domain in args.domains:
        adapter_dir = REPO_ROOT / "checkpoints" / "lora_adapters" / domain

        def _make_check(d=adapter_dir):
            def _check() -> bool:
                return d.exists() and any(d.iterdir())
            return _check

        cmd = [
            py, "-m", "src.continual.continual_trainer",
            "--config", args.config,
            "--domain", domain,
            "--epochs", str(args.epochs),
            "--device", args.device,
            # continual_trainer.py resumes from its own epoch-level
            # checkpoint by default; --resume is the default but we pass
            # it explicitly for clarity.
            "--resume",
        ]
        if args.max_samples_per_epoch is not None:
            cmd += ["--max-samples-per-epoch", str(args.max_samples_per_epoch)]
        if args.max_val_samples is not None:
            cmd += ["--max-val-samples", str(args.max_val_samples)]

        stages.append(
            Stage(
                name=domain,
                label=f"Train domain '{domain}' with EWC + Replay",
                command=cmd,
                check=_make_check(),
            )
        )

    # ── Stage: domain router ──
    stages.append(
        Stage(
            name="train_router",
            label="Train domain router",
            command=[
                py, "-m", "src.router.domain_router",
                "--config", args.router_config,
                "--domains", ",".join(args.domains),
                "--epochs", str(args.router_epochs),
            ],
        )
    )

    # ── Stage: output heads ──
    stages.append(
        Stage(
            name="train_heads",
            label="Train output heads (waypoint, hazard, regulation, weather)",
            command=[
                py, "-m", "src.heads.integration_layer",
                "--config", args.heads_config,
                "--heads", "waypoint,hazard,regulation,weather",
                "--domains", ",".join(args.domains),
                "--epochs", str(args.heads_epochs),
                "--max-samples-per-epoch", "3000",
            ],
        )
    )

    # ── Stage: CADRE-Bench ──
    bench_out = Path(args.benchmark_output_dir)

    def _bench_check() -> bool:
        return bench_out.exists() and any(bench_out.iterdir())

    stages.append(
        Stage(
            name="benchmark",
            label="Run CADRE-Bench evaluation",
            command=[
                py, "-m", "src.benchmark.cadre_bench",
                "--config", args.benchmark_config,
                "--domains", ",".join(args.domains),
                "--output_dir", args.benchmark_output_dir,
            ],
            check=_bench_check,
        )
    )

    return stages


# ──────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────

def run_stage(stage: Stage, state: dict) -> None:
    """Run a single stage as a subprocess, streaming its output live.
    Raises SystemExit on failure or interruption, after saving state."""
    print("\n" + "=" * 60)
    print(f"  ▶ STAGE: {stage.label}")
    print(f"    ({stage.name})")
    print("=" * 60)
    print(f"$ {' '.join(stage.command)}\n")

    mark_started(state, stage.name)
    start = time.time()

    try:
        result = subprocess.run(stage.command, cwd=str(REPO_ROOT))
    except KeyboardInterrupt:
        elapsed = time.time() - start
        print(
            f"\n\n⏸  Interrupted during stage '{stage.name}' after "
            f"{elapsed/60:.1f} min. Nothing lost: this stage is NOT marked "
            f"complete, and if it was a domain-training stage, its own "
            f"epoch checkpoint (checkpoints/training_checkpoints/{stage.name}/"
            f"latest.pt) already has your progress.\n"
            f"Just rerun:\n    python scripts/run_pipeline.py\n"
            f"...and it will pick up exactly here."
        )
        sys.exit(130)

    if result.returncode != 0:
        print(
            f"\n\n❌ Stage '{stage.name}' failed (exit code {result.returncode}). "
            f"It is NOT marked complete. Fix the issue above, then rerun:\n"
            f"    python scripts/run_pipeline.py\n"
            f"...and every already-completed stage will be skipped."
        )
        sys.exit(result.returncode)

    elapsed = time.time() - start
    mark_complete(state, stage.name)
    print(f"\n✅ Stage '{stage.name}' complete ({elapsed/60:.1f} min). State saved.")


def print_status(stages: List[Stage], state: dict) -> None:
    print("\nCADRE pipeline status")
    print("-" * 60)
    for stage in stages:
        done = is_complete(state, stage.name)
        if done and stage.check is not None and not stage.check():
            marker = "⚠️  marked done, but expected output is missing"
        elif done:
            marker = "✅ done"
        else:
            marker = "⬜ pending"
        print(f"  {marker:<45} {stage.name}")
    print("-" * 60)
    print(f"State file: {STATE_PATH}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run (or resume) the full CADRE training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--router_config", default="configs/router_config.yaml")
    parser.add_argument("--heads_config", default="configs/heads_config.yaml")
    parser.add_argument("--benchmark_config", default="configs/benchmark_config.yaml")
    parser.add_argument("--model_path", default="checkpoints/llava-v1.5-7b")
    parser.add_argument(
        "--domains", default=",".join(DEFAULT_DOMAINS),
        help="Comma-separated domains to train, in order",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per domain")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max_samples_per_epoch", type=int, default=1500,
        help="Passed through to continual_trainer.py as --max-samples-per-epoch",
    )
    parser.add_argument(
        "--max_val_samples", type=int, default=None,
        help="Passed through to continual_trainer.py as --max-val-samples "
             "(omit to use continual_trainer.py's own default)",
    )
    parser.add_argument("--router_epochs", type=int, default=20)
    parser.add_argument("--heads_epochs", type=int, default=15)
    parser.add_argument("--benchmark_output_dir", default="outputs/cadre_bench")

    parser.add_argument(
        "--status", action="store_true",
        help="Print which stages are done / pending and exit without running anything.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Forget all pipeline progress (does NOT delete any trained files) and exit.",
    )
    parser.add_argument(
        "--redo", metavar="STAGE_NAME", default=None,
        help="Force one stage to be treated as not-done before running "
             "(e.g. --redo domain_sg). Runs the full pipeline afterward "
             "unless combined with --only.",
    )
    parser.add_argument(
        "--only", metavar="STAGE_NAME", default=None,
        help="Run only this one stage (still records completion in state).",
    )
    args = parser.parse_args()
    args.domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    stages = build_stages(args)
    state = _load_state()

    if args.reset:
        _save_state({"completed_stages": [], "history": []})
        print(f"🔄 Pipeline progress reset. {STATE_PATH} cleared.")
        print("   (No trained checkpoints/adapters were deleted.)")
        return

    if args.redo:
        valid_names = {s.name for s in stages}
        if args.redo not in valid_names:
            print(f"Unknown stage '{args.redo}'. Valid stages: {sorted(valid_names)}")
            sys.exit(1)
        state["completed_stages"] = [
            s for s in state["completed_stages"] if s != args.redo
        ]
        _save_state(state)
        print(f"🔁 Stage '{args.redo}' will be re-run.")

    if args.status:
        print_status(stages, state)
        return

    if args.only:
        matching = [s for s in stages if s.name == args.only]
        if not matching:
            print(f"Unknown stage '{args.only}'. Valid stages: {[s.name for s in stages]}")
            sys.exit(1)
        run_stage(matching[0], state)
        return

    print("\n" + "=" * 60)
    print("  CADRE — Resumable Full Training Pipeline")
    print("=" * 60)
    print_status(stages, state)

    for stage in stages:
        already_done = is_complete(state, stage.name)
        output_missing = stage.check is not None and not stage.check()

        if already_done and not output_missing:
            print(f"⏭  Skipping '{stage.name}' — already completed.")
            continue

        if already_done and output_missing:
            print(
                f"⚠️  '{stage.name}' was marked complete but its expected output "
                f"is missing — re-running it."
            )

        run_stage(stage, state)

    print("\n" + "=" * 60)
    print("  ✅ Pipeline complete!")
    print(f"  Results in: {args.benchmark_output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()