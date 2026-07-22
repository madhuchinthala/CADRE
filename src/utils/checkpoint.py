"""
Checkpoint Utilities
=====================
Save and load model checkpoints, LoRA adapters, and training state.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model,
    optimizer=None,
    epoch: int = 0,
    metrics: Dict = None,
    save_path: str = "checkpoints",
    filename: str = "checkpoint.pt",
):
    """
    Save a training checkpoint.

    Args:
        model: The model (or PeftModel)
        optimizer: Optional optimizer state
        epoch: Current epoch number
        metrics: Optional dict of metrics
        save_path: Directory to save checkpoint
        filename: Checkpoint filename
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "metrics": metrics or {},
    }

    # Save model state dict
    if hasattr(model, "save_pretrained"):
        # PEFT model — save adapter only
        model.save_pretrained(str(save_dir / "adapter"))
        checkpoint["model_type"] = "peft"
    else:
        checkpoint["model_state_dict"] = model.state_dict()
        checkpoint["model_type"] = "standard"

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, save_dir / filename)
    logger.info(f"Checkpoint saved: {save_dir / filename}")


def load_checkpoint(
    model,
    load_path: str = "checkpoints",
    filename: str = "checkpoint.pt",
    optimizer=None,
    device: str = "cpu",
) -> Dict:
    """
    Load a training checkpoint.

    Args:
        model: The model to load weights into
        load_path: Directory containing checkpoint
        filename: Checkpoint filename
        optimizer: Optional optimizer to load state into
        device: Device to map tensors to

    Returns:
        Dict with epoch, metrics, etc.
    """
    load_file = Path(load_path) / filename

    if not load_file.exists():
        logger.warning(f"No checkpoint found at {load_file}")
        return {"epoch": 0, "metrics": {}}

    checkpoint = torch.load(load_file, map_location=device, weights_only=False)

    if checkpoint.get("model_type") == "peft":
        # PEFT adapter — load separately
        adapter_dir = Path(load_path) / "adapter"
        if adapter_dir.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            logger.info(f"PEFT adapter loaded from {adapter_dir}")
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(
        f"Checkpoint loaded: epoch={checkpoint.get('epoch', 0)}, "
        f"metrics={checkpoint.get('metrics', {})}"
    )

    return checkpoint
