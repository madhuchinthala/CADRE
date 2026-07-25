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

FIXES applied in this version:
  1. Added --max-samples-per-epoch (and --max-val-samples) so a full pass
     over 27,500 batches (~17 hrs) isn't required just to see one epoch of
     results. Training and validation both stop early once the sample cap
     is hit, using itertools.islice on the dataloader.
  2. Fixed a NameError in _save_checkpoint(): the function referenced
     `next_epoch`, a variable that was never defined in that scope (the
     parameter is named `epoch`). This crashed the very first time an
     epoch finished and a checkpoint save was attempted.
  3. Checkpoint save/load now only stores the PEFT/LoRA state dict
     (~tens of MB) instead of the entire frozen 7B backbone's
     state_dict() (~14GB in fp16). Saving the full model every epoch was
     unnecessary and would have made checkpointing itself a major
     bottleneck once epochs became fast.
  4. Added best-model checkpointing: whenever val_score improves,
     checkpoints/best_checkpoints/<domain>/best.pt is (over)written. This
     is separate from the "latest" resume checkpoint (which just tracks
     progress and gets deleted once training finishes). Right before the
     final LoRA adapter is saved, the best checkpoint's weights are
     restored into the model — so the adapter you end up with reflects
     the best epoch, not necessarily the last one that happened to run.
"""

import argparse
import itertools
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
        max_samples_per_epoch: Optional[int] = None,
        max_val_samples: Optional[int] = None,
    ):
        self.model = model
        self.ewc = ewc
        self.replay_buffer = replay_buffer
        self.max_epochs = max_epochs
        self.grad_accum = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.max_samples_per_epoch = max_samples_per_epoch
        self.max_val_samples = max_val_samples

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
            f"grad_accum={gradient_accumulation_steps}, "
            f"max_samples_per_epoch={max_samples_per_epoch}, "
            f"max_val_samples={max_val_samples}"
        )

    def train_domain(
        self,
        domain_name: str,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        resume: bool = True,
    ) -> Dict[str, float]:
        """
        Train on a single domain with EWC + replay.

        Args:
            domain_name: Name of the domain (e.g., "domain_us")
            train_dataloader: DataLoader for the new domain
            val_dataloader: Optional validation DataLoader
            resume: If True, resume from last checkpoint if available

        Returns:
            Dict of training metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  Training domain: {domain_name}")
        logger.info(f"{'='*60}")

        # Try to load checkpoint if resuming
        start_epoch = 0
        training_history = []
        best_val_score = 0.0

        if resume:
            checkpoint_data = self._load_checkpoint(domain_name)
            if checkpoint_data:
                start_epoch = checkpoint_data["epoch"]  # Next epoch to train (0-indexed)
                training_history = checkpoint_data["history"]
                best_val_score = checkpoint_data["best_val_score"]
                logger.info(f"✅ Resumed from checkpoint: will continue from Epoch {start_epoch+1}")

        # Create mixed dataloader with replay
        mixed_loader = self.replay_buffer.get_mixed_dataloader(
            new_domain_dataset=train_dataloader.dataset,
            exclude_domain=domain_name,
            batch_size=train_dataloader.batch_size,
            collate_fn=train_dataloader.collate_fn,
        )

        for epoch in range(start_epoch, self.max_epochs):
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
                # ── NEW: save the best-performing checkpoint separately ──
                # This is distinct from the "latest" resume checkpoint below —
                # it survives even after the resume checkpoint gets deleted at
                # the end of training, and is what actually gets restored
                # before the final adapter save.
                self._save_best_checkpoint(domain_name, epoch + 1, val_score)

            # Save checkpoint with NEXT epoch to train (for proper resume)
            # If epoch=0 completes, next_epoch=1, so resume starts from epoch 1
            next_epoch = epoch + 1
            self._save_checkpoint(domain_name, next_epoch, training_history, best_val_score)

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
        # NOTE: this scans the *entire* train_dataloader (not capped by
        # max_samples_per_epoch) — reservoir sampling needs to see every
        # sample to stay unbiased. This step can take a while even when
        # training itself is fast; that's expected, not a bug.
        logger.info(f"Populating replay buffer for {domain_name}...")
        self.replay_buffer.populate_from_dataloader(
            domain_name=domain_name,
            dataloader=train_dataloader,
        )
        self.replay_buffer.save(domain_name)

        # ── Restore best checkpoint before saving the final adapter ──
        # Without this, the adapter saved below would reflect whichever
        # epoch happened to run last, not the best-performing one.
        best_path = Path("checkpoints/best_checkpoints") / domain_name / "best.pt"
        if best_path.exists():
            try:
                best_ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                try:
                    from peft import set_peft_model_state_dict
                    set_peft_model_state_dict(self.model, best_ckpt["model_state_dict"])
                except Exception:
                    self.model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
                logger.info(
                    f"⭐ Restored best checkpoint before saving adapter: "
                    f"val_score={best_ckpt['val_score']:.4f} (epoch {best_ckpt['epoch']})"
                )
            except Exception as e:
                logger.warning(
                    f"⚠️  Failed to restore best checkpoint, saving final-epoch weights instead: {e}"
                )
        else:
            logger.info(
                "No best checkpoint found (no validation improvement recorded) — "
                "saving final-epoch weights."
            )

        # ── Post-training: save LoRA adapter ──
        logger.info(f"Saving LoRA adapter for {domain_name}...")
        try:
            from src.adapters.lora_adapter import LoRAAdapterManager
            adapter_dir = Path("checkpoints/lora_adapters") / domain_name
            adapter_dir.mkdir(parents=True, exist_ok=True)
            # Save using PEFT's native save_pretrained if available
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(str(adapter_dir))
                logger.info(f"✅ LoRA adapter saved to {adapter_dir}")
            else:
                logger.warning(f"⚠️  Model does not support save_pretrained(); adapter not saved.")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save LoRA adapter: {e}")

        # Clean up checkpoint after successful training
        self._delete_checkpoint(domain_name)

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

        # Limit batches per epoch if max_samples_per_epoch is set, so a
        # full pass over the whole dataset isn't required to see progress.
        max_batches = None
        if self.max_samples_per_epoch is not None:
            batch_size = dataloader.batch_size or 1
            max_batches = max(1, self.max_samples_per_epoch // batch_size)
            logger.info(
                f"Epoch {epoch+1}: limiting to {max_batches} batches "
                f"(~{max_batches * batch_size} samples out of ~{len(dataloader) * batch_size} total)"
            )

        iterable = dataloader
        total_for_bar = len(dataloader)
        if max_batches is not None:
            iterable = itertools.islice(dataloader, max_batches)
            total_for_bar = max_batches

        pbar = tqdm(iterable, desc=f"Epoch {epoch+1}", total=total_for_bar)

        def _to_device(obj, device):
            if torch.is_tensor(obj):
                return obj.to(device)
            if isinstance(obj, dict):
                return {k: _to_device(v, device) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_to_device(v, device) for v in obj)
            return obj

        for step, batch in enumerate(pbar):
            # Forward pass
            if isinstance(batch, dict):
                # Move batch to the same device as the model parameters (handles accelerate device_map)
                model_device = next(self.model.parameters()).device
                batch = _to_device(batch, model_device)

                # Extract task-specific labels (waypoints, hazard, regulation, weather) that aren't model inputs
                # These will be used for multi-head training in the future
                task_labels = {}
                model_batch = {}
                valid_model_keys = {'pixel_values', 'input_ids', 'labels', 'attention_mask', 'pad_token_id', 'output_attentions', 'output_hidden_states', 'return_dict'}

                for key, value in batch.items():
                    if key in valid_model_keys:
                        model_batch[key] = value
                    else:
                        # Store task labels for future multi-head training
                        task_labels[key] = value

                # Generate attention_mask if missing (all 1s for all tokens)
                if 'attention_mask' not in model_batch and 'input_ids' in model_batch:
                    input_ids = model_batch['input_ids']
                    model_batch['attention_mask'] = torch.ones_like(input_ids, dtype=torch.long)

                outputs = self.model(**model_batch)
                task_loss = outputs.loss
            else:
                inputs, targets = batch
                # move to model device as well
                model_device = next(self.model.parameters()).device
                inputs = inputs.to(model_device)
                targets = targets.to(model_device)
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

        # Cap validation size too — an uncapped validation pass can end up
        # dominating epoch time once training itself is capped.
        max_val_batches = None
        if self.max_val_samples is not None:
            batch_size = dataloader.batch_size or 1
            max_val_batches = max(1, self.max_val_samples // batch_size)

        iterable = dataloader
        if max_val_batches is not None:
            iterable = itertools.islice(dataloader, max_val_batches)

        with torch.no_grad():
            for batch in iterable:
                if isinstance(batch, dict):
                    model_device = next(self.model.parameters()).device

                    def _to_device(obj, device):
                        if torch.is_tensor(obj):
                            return obj.to(device)
                        if isinstance(obj, dict):
                            return {k: _to_device(v, device) for k, v in obj.items()}
                        if isinstance(obj, (list, tuple)):
                            return type(obj)(_to_device(v, device) for v in obj)
                        return obj

                    batch = _to_device(batch, model_device)

                    # Filter batch to only include valid model arguments
                    valid_model_keys = {'pixel_values', 'input_ids', 'labels', 'attention_mask', 'pad_token_id', 'output_attentions', 'output_hidden_states', 'return_dict'}
                    model_batch = {k: v for k, v in batch.items() if k in valid_model_keys}

                    # Generate attention_mask if missing
                    if 'attention_mask' not in model_batch and 'input_ids' in model_batch:
                        input_ids = model_batch['input_ids']
                        model_batch['attention_mask'] = torch.ones_like(input_ids, dtype=torch.long)

                    outputs = self.model(**model_batch)
                    preds = outputs.logits.argmax(dim=-1)
                    targets = batch.get("labels", batch.get("input_ids"))
                else:
                    inputs, targets = batch
                    model_device = next(self.model.parameters()).device
                    outputs = self.model(inputs.to(model_device))
                    preds = (outputs.logits if hasattr(outputs, 'logits') else outputs).argmax(dim=-1)
                    targets = targets.to(model_device)

                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return correct / max(total, 1)

    def _save_checkpoint(self, domain_name: str, epoch: int, history: list, best_val_score: float):
        """
        Save training checkpoint for resuming.

        Only the trainable PEFT/LoRA parameters are saved (not the full
        frozen 7B backbone) — this keeps checkpoint files small (tens of
        MB instead of ~14GB) and fast to write every epoch.

        Args:
            epoch: The 0-indexed epoch to resume from (e.g. 1 means
                   resume starting at "Epoch 2" for the user-facing log).
        """
        checkpoint_dir = Path("checkpoints/training_checkpoints") / domain_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / "latest.pt"

        # FIX: only persist LoRA/PEFT weights, not the entire backbone.
        try:
            from peft import get_peft_model_state_dict
            model_state_dict = get_peft_model_state_dict(self.model)
        except Exception:
            # Fallback: filter state_dict manually for any "lora" params
            model_state_dict = {
                k: v for k, v in self.model.state_dict().items() if "lora" in k.lower()
            }

        checkpoint = {
            "epoch": epoch,  # Next epoch to train (0-indexed)
            "step": 0,
            "history": history,
            "best_val_score": best_val_score,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        n_params = sum(v.numel() for v in model_state_dict.values())
        logger.debug(
            f"Checkpoint saved ({n_params/1e6:.1f}M LoRA params): "
            f"will resume from Epoch {epoch+1}"
        )

    def _save_best_checkpoint(self, domain_name: str, epoch: int, val_score: float):
        """
        Save the best-performing model checkpoint (by validation score),
        kept separate from the "latest" resume checkpoint. This file is
        NOT deleted when training finishes — it's what gets restored
        right before the final LoRA adapter is saved, so the shipped
        adapter reflects the best epoch rather than whichever epoch
        happened to run last.
        """
        best_dir = Path("checkpoints/best_checkpoints") / domain_name
        best_dir.mkdir(parents=True, exist_ok=True)
        best_path = best_dir / "best.pt"

        try:
            from peft import get_peft_model_state_dict
            model_state_dict = get_peft_model_state_dict(self.model)
        except Exception:
            model_state_dict = {
                k: v for k, v in self.model.state_dict().items() if "lora" in k.lower()
            }

        torch.save(
            {
                "epoch": epoch,
                "val_score": val_score,
                "model_state_dict": model_state_dict,
            },
            best_path,
        )
        logger.info(
            f"⭐ New best model for {domain_name}: val_score={val_score:.4f} "
            f"(epoch {epoch}) saved to {best_path}"
        )

    def _load_checkpoint(self, domain_name: str) -> Optional[Dict]:
        """Load training checkpoint if available."""
        checkpoint_path = Path("checkpoints/training_checkpoints") / domain_name / "latest.pt"
        if not checkpoint_path.exists():
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # FIX: load only into the LoRA/PEFT params, matching how it
            # was saved — the raw state_dict here is a partial (LoRA-only)
            # dict, not a full model state_dict.
            try:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(self.model, checkpoint["model_state_dict"])
            except Exception:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}")
            return None

    def _delete_checkpoint(self, domain_name: str):
        """Delete checkpoint after successful training completion."""
        checkpoint_path = Path("checkpoints/training_checkpoints") / domain_name / "latest.pt"
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.debug(f"Deleted checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Continual training with EWC + Replay")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from last checkpoint if available (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh (do not resume)")
    parser.add_argument(
        "--max-samples-per-epoch", type=int, default=6000,
        help="Limit training samples per epoch. Default 6000 (~55-60 min at "
             "~2.2s/it, batch_size=4). Pass a higher value (e.g. 9000-9600) "
             "for closer to 90 min epochs, or omit / pass a huge number for "
             "a full, uncapped epoch.",
    )
    parser.add_argument(
        "--max-val-samples", type=int, default=500,
        help="Limit validation samples per epoch (default 500), so validation "
             "doesn't end up dominating epoch time once training is capped.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load backbone (processor + model)
    from src.models.vla_backbone import VLABackbone
    from src.adapters.lora_adapter import LoRAAdapterManager

    backbone = VLABackbone(
        model_path=cfg["paths"]["model_checkpoint"],
        dtype=cfg["model"]["dtype"],
        gradient_checkpointing=cfg["model"]["gradient_checkpointing"]
    )
    model = backbone.get_model()

    # EWC + Replay buffer
    ewc = EWC(lambda_ewc=5000.0)
    replay = DomainReplayBuffer(
        buffer_size_per_domain=2000,
        replay_ratio=0.3,
        base_dir=cfg["paths"].get("replay_buffer", "replay_buffer")
    )

    # Inject LoRA adapters
    lora_mgr = LoRAAdapterManager(config_path="configs/lora_config.yaml")
    try:
        model = lora_mgr.inject_lora(model)
    except Exception as e:
        logger.warning(f"LoRA injection failed: {e} — continuing with base model")

    # Trainer
    trainer = ContinualTrainer(
        model=model,
        ewc=ewc,
        replay_buffer=replay,
        max_epochs=args.epochs,
        device=args.device,
        max_samples_per_epoch=args.max_samples_per_epoch,
        max_val_samples=args.max_val_samples,
    )

    # DataLoader for the target domain
    from src.data.dataloader import get_dataloader
    processor = backbone.get_processor()
    train_dl = get_dataloader(args.config, args.domain, split="train", processor=processor)
    val_dl = get_dataloader(args.config, args.domain, split="val", processor=processor)

    # Train
    metrics = trainer.train_domain(
        domain_name=args.domain,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        resume=args.resume
    )
    logger.info(f"✅ Training complete: {metrics}")