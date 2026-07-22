"""
PART 2: LoRA Adapter — Parameter-Efficient Domain Adaptation
=============================================================
Injects Low-Rank Adaptation (LoRA) layers into the frozen LLaVA backbone.
Only 0.35% of parameters become trainable per domain (~50 MB saved adapter).
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

logger = logging.getLogger(__name__)


class LoRAAdapterManager:
    """
    Manages LoRA adapter injection, saving, and loading for per-domain adaptation.

    Each domain gets its own LoRA adapter (~50 MB), stored in:
        checkpoints/lora_adapters/domain_<name>/

    At inference, the domain router selects which adapter to activate.
    """

    def __init__(
        self,
        config_path: str = "configs/lora_config.yaml",
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["lora"]

        self.lora_config = LoraConfig(
            r=self.config["rank"],
            lora_alpha=self.config["alpha"],
            lora_dropout=self.config["dropout"],
            target_modules=self.config["target_modules"],
            bias=self.config["bias"],
            task_type=TaskType.CAUSAL_LM,
        )

        logger.info(
            f"LoRA config: rank={self.config['rank']}, "
            f"alpha={self.config['alpha']}, "
            f"targets={self.config['target_modules']}"
        )

    def inject_lora(self, model) -> PeftModel:
        """
        Inject LoRA adapters into the frozen backbone model.

        Args:
            model: The frozen HuggingFace LLaVA model

        Returns:
            PeftModel with LoRA adapters (trainable params ~0.35%)
        """
        peft_model = get_peft_model(model, self.lora_config)

        # Report parameter stats
        trainable, total = peft_model.get_nb_trainable_parameters()
        ratio = trainable / total * 100

        logger.info(f"LoRA injected successfully:")
        logger.info(f"  Total params:     {total:,}")
        logger.info(f"  Trainable params: {trainable:,}")
        logger.info(f"  Trainable ratio:  {ratio:.3f}%")

        return peft_model

    def save_adapter(self, peft_model: PeftModel, domain_name: str, base_dir: str = "checkpoints/lora_adapters"):
        """
        Save the LoRA adapter weights for a specific domain.

        Args:
            peft_model: The PEFT-wrapped model
            domain_name: e.g. "domain_us", "domain_sg"
            base_dir: Base directory for adapter storage
        """
        save_path = Path(base_dir) / domain_name
        save_path.mkdir(parents=True, exist_ok=True)

        peft_model.save_pretrained(str(save_path))

        # Report saved size
        total_size = sum(f.stat().st_size for f in save_path.rglob("*") if f.is_file())
        logger.info(
            f"Adapter saved to {save_path} "
            f"({total_size / 1024 / 1024:.1f} MB)"
        )

    def load_adapter(self, base_model, domain_name: str, base_dir: str = "checkpoints/lora_adapters") -> PeftModel:
        """
        Load a previously saved LoRA adapter for a specific domain.

        Args:
            base_model: The frozen backbone model
            domain_name: e.g. "domain_us"
            base_dir: Base directory for adapter storage

        Returns:
            PeftModel with loaded adapter weights
        """
        adapter_path = Path(base_dir) / domain_name

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"No adapter found at {adapter_path}. "
                f"Train domain '{domain_name}' first."
            )

        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        logger.info(f"Loaded adapter from {adapter_path}")
        return peft_model

    def list_available_adapters(self, base_dir: str = "checkpoints/lora_adapters") -> List[str]:
        """List all saved domain adapters."""
        base = Path(base_dir)
        if not base.exists():
            return []
        return [d.name for d in base.iterdir() if d.is_dir() and (d / "adapter_config.json").exists()]


# ── CLI Entry Point ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Inject LoRA adapters")
    parser.add_argument("--config", default="configs/lora_config.yaml")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., domain_us)")
    parser.add_argument("--rank", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--alpha", type=int, default=None, help="Override LoRA alpha")
    args = parser.parse_args()

    print(f"LoRA adapter manager initialized for domain: {args.domain}")
    print("To use: first load the backbone, then call inject_lora(model)")
