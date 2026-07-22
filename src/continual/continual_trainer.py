"""
Continual Trainer — Combined EWC + Replay Training Loop
========================================================
Orchestrates the full continual learning pipeline:
1. Load backbone + inject LoRA
2. For each domain in sequence:
   a. Create mixed dataloader (new data + replay)
   b. Train with EWC penalty
   c. Compute Fisher for EWC
   d. Populate replay buffer
   e. Save adapter checkpoint
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.continual.ewc import EWC
from src.continual.replay_buffer import DomainReplayBuffer

logger = logging.getLogger(__name__)


class ContinualTrainer:
    """
    Orchestrates continual training across sequential driving domains.

    Training flow for domain D_k:
    1. Create mixed dataloader: 70% D_k + 30% replay(D_1..D_{k-1})
    2. Standard loss + EWC penalty
    3. After training: compute Fisher, populate replay buffer
    4. Save LoRA adapter for D_k
    """

    def __init__(
        self,
        model: nn.Module,
        ewc: EWC,
        replay_buffer: DomainReplayBuffer,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 10,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.ewc = ewc
        self.replay_buffer = replay_buffer
        self.max_epochs = max_epochs
        self.grad_accum = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Optimizer only for trainable (LoRA) parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()

        logger.info(
            f"ContinualTrainer: lr={learning_rate}, epochs={max_epochs}, "
            f"grad_accum={gradient_accumulation_steps}"
        )

    def train_domain(
        self,
        domain_name: str,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Train on a single domain with EWC + replay.

        Args:
            domain_name: Name of the domain (e.g., "domain_us")
            train_dataloader: DataLoader for the new domain
            val_dataloader: Optional validation DataLoader

        Returns:
            Dict of training metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  Training domain: {domain_name}")
        logger.info(f"{'='*60}")

        # Create mixed dataloader with replay
        mixed_loader = self.replay_buffer.get_mixed_dataloader(
            new_domain_dataset=train_dataloader.dataset,
            exclude_domain=domain_name,
            batch_size=train_dataloader.batch_size,
        )

        best_val_score = 0.0
        training_history = []

        for epoch in range(self.max_epochs):
            # ── Train epoch ──
            train_loss = self._train_epoch(mixed_loader, epoch)

            # ── Validate ──
            val_score = 0.0
            if val_dataloader is not None:
                val_score = self._validate(val_dataloader)

            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs}: "
                f"loss={train_loss:.4f}, val_score={val_score:.4f}"
            )

            training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_score": val_score,
            })

            if val_score > best_val_score:
                best_val_score = val_score

        # ── Post-training: compute Fisher for EWC ──
        logger.info(f"Computing Fisher Information for {domain_name}...")
        self.ewc.compute_fisher(
            model=self.model,
            dataloader=train_dataloader,
            domain_name=domain_name,
        )
        self.ewc.save(
            path="checkpoints/fisher_matrices",
            domain_name=domain_name,
        )

        # ── Post-training: populate replay buffer ──
        logger.info(f"Populating replay buffer for {domain_name}...")
        self.replay_buffer.populate_from_dataloader(
            domain_name=domain_name,
            dataloader=train_dataloader,
        )
        self.replay_buffer.save(domain_name)

        return {
            "domain": domain_name,
            "best_val_score": best_val_score,
            "final_train_loss": training_history[-1]["train_loss"] if training_history else 0,
            "epochs_trained": len(training_history),
        }

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Run one training epoch with EWC penalty."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(pbar):
            # Forward pass
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                task_loss = outputs.loss
            else:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                task_loss = self.criterion(outputs.logits if hasattr(outputs, 'logits') else outputs, targets)

            # EWC penalty
            ewc_loss = self.ewc.penalty(self.model)

            # Combined loss
            loss = task_loss + ewc_loss
            loss = loss / self.grad_accum

            loss.backward()

            if (step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss/n_batches:.4f}",
                "ewc": f"{ewc_loss.item():.4f}",
            })

        return total_loss / max(n_batches, 1)

    def _validate(self, dataloader: DataLoader) -> float:
        """Run validation and return accuracy."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    preds = outputs.logits.argmax(dim=-1)
                    targets = batch.get("labels", batch.get("input_ids"))
                else:
                    inputs, targets = batch
                    outputs = self.model(inputs.to(self.device))
                    preds = (outputs.logits if hasattr(outputs, 'logits') else outputs).argmax(dim=-1)
                    targets = targets.to(self.device)

                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return correct / max(total, 1)


# ── CLI Entry Point ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Continual training with EWC + Replay")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--dataset", choices=["bdd100k", "nuscenes"], required=True)
    parser.add_argument("--ewc_lambda", type=float, default=5000.0)
    parser.add_argument("--replay_ratio", type=float, default=0.3)
    parser.add_argument("--replay_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    print(f"Continual trainer configured for domain: {args.domain}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  EWC lambda:   {args.ewc_lambda}")
    print(f"  Replay ratio: {args.replay_ratio}")
    print(f"  Replay size:  {args.replay_size}")
    print(f"  Epochs:       {args.epochs}")
    print("\nTo run: load backbone, inject LoRA, then call train_domain()")
