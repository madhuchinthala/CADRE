"""
Quick training runner for a single domain using the CADRE scaffolding.

This script wires the backbone, replay buffer, EWC, datasets and trainer
and runs training for one domain. It's a convenience helper for local
experimentation — adapt paths and options as needed.
"""
import argparse
import logging
try:
    import yaml
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])  # install into current env
    import yaml

import torch

from src.models.vla_backbone import VLABackbone
from src.continual.ewc import EWC
from src.continual.replay_buffer import DomainReplayBuffer
from src.continual.continual_trainer import ContinualTrainer
from src.data.dataloader import get_dataloader
from src.adapters.lora_adapter import LoRAAdapterManager


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run single-domain training (scaffold)")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load backbone (processor + model)
    backbone = VLABackbone(model_path=cfg["paths"]["model_checkpoint"], dtype=cfg["model"]["dtype"], gradient_checkpointing=cfg["model"]["gradient_checkpointing"])
    model = backbone.get_model()

    # EWC + Replay buffer
    ewc = EWC(lambda_ewc=5000.0)
    replay = DomainReplayBuffer(buffer_size_per_domain=2000, replay_ratio=0.3, base_dir=cfg["paths"].get("replay_buffer", "replay_buffer"))

    # Inject LoRA adapters to create trainable parameters
    lora_mgr = LoRAAdapterManager(config_path="configs/lora_config.yaml")
    try:
        model = lora_mgr.inject_lora(model)
    except Exception:
        # If PEFT injection fails, continue with base model (will likely error on optimizer)
        logging.getLogger(__name__).warning("LoRA injection failed — continuing with base model")

    # Trainer (optimizer acts on trainable LoRA params)
    trainer = ContinualTrainer(model=model, ewc=ewc, replay_buffer=replay, max_epochs=args.epochs, device=args.device)

    # DataLoader for the target domain — provide processor from backbone to avoid redundant loads
    processor = backbone.get_processor()
    train_dl = get_dataloader(args.config, args.domain, split="train", processor=processor)
    val_dl = get_dataloader(args.config, args.domain, split="val", processor=processor) if True else None

    metrics = trainer.train_domain(domain_name=args.domain, train_dataloader=train_dl, val_dataloader=val_dl)
    print("Training complete:", metrics)


if __name__ == "__main__":
    main()
