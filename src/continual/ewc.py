"""
PART 3: Elastic Weight Consolidation (EWC)
==========================================
Computes the Fisher Information Matrix after each domain training.
Adds a quadratic penalty to the loss to protect important weights
from being overwritten when learning new domains.
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EWC:
    """
    Elastic Weight Consolidation for continual learning.

    After training on domain D_k:
    1. Compute Fisher Information F_k for each trainable parameter
    2. Store optimal parameter values θ*_k

    When training on domain D_{k+1}:
    3. Add penalty: L_ewc = λ/2 * Σ_i F_i * (θ_i - θ*_i)²

    This penalizes changes to weights that were important for D_k,
    preventing catastrophic forgetting while still allowing plasticity.
    """

    def __init__(
        self,
        lambda_ewc: float = 5000.0,
        fisher_n_samples: int = 2000,
        normalize_fisher: bool = True,
        variant: str = "online_ewc",
        gamma: float = 0.95,
        lora_only: bool = True,
    ):
        self.lambda_ewc = lambda_ewc
        self.fisher_n_samples = fisher_n_samples
        self.normalize_fisher = normalize_fisher
        self.variant = variant
        self.gamma = gamma
        self.lora_only = lora_only

        # Storage for Fisher matrices and optimal params from previous domains
        self.fisher_matrices: Dict[str, Dict[str, torch.Tensor]] = {}
        self.optimal_params: Dict[str, Dict[str, torch.Tensor]] = {}

        # For online EWC: running Fisher estimate
        self.running_fisher: Optional[Dict[str, torch.Tensor]] = None
        self.running_optimal: Optional[Dict[str, torch.Tensor]] = None

        logger.info(
            f"EWC initialized: λ={lambda_ewc}, variant={variant}, "
            f"γ={gamma}, samples={fisher_n_samples}"
        )

    def _get_trainable_params(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        """Get trainable parameters (LoRA params if lora_only=True)."""
        params = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self.lora_only and "lora" not in name.lower():
                continue
            params[name] = param
        return params

    def compute_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        domain_name: str,
        criterion: nn.Module = None,
    ):
        """
        Compute diagonal Fisher Information Matrix for the current domain.

        The Fisher Information measures how sensitive the loss is to each
        parameter — high Fisher = important parameter = protect it.

        Args:
            model: The trained model (with LoRA)
            dataloader: Data from the current domain
            criterion: Loss function (defaults to cross-entropy)
            domain_name: Name of current domain for storage
        """
        logger.info(f"Computing Fisher Information for domain: {domain_name}")

        model.eval()
        trainable_params = self._get_trainable_params(model)

        # Initialize Fisher accumulators
        fisher = {
            name: torch.zeros_like(param)
            for name, param in trainable_params.items()
        }

        n_samples = 0

        for batch in tqdm(dataloader, desc="Computing Fisher", total=min(len(dataloader), self.fisher_n_samples)):
            if n_samples >= self.fisher_n_samples:
                break

            model.zero_grad()

            # Forward pass — adapt this to your data format
            if isinstance(batch, dict):
                model_device = next(model.parameters()).device
                valid_model_keys = {'pixel_values', 'input_ids', 'labels', 'attention_mask', 'pad_token_id', 'output_attentions', 'output_hidden_states', 'return_dict'}
                model_batch = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k in valid_model_keys}
                if 'attention_mask' not in model_batch and 'input_ids' in model_batch:
                    input_ids = model_batch['input_ids']
                    model_batch['attention_mask'] = torch.ones_like(input_ids, dtype=torch.long)
                outputs = model(**model_batch)
                loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else criterion(outputs, batch)
                batch_size = len(batch.get("labels", [1]))
            else:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets) if criterion else outputs.loss
                batch_size = targets.size(0)

            loss.backward()

            # Accumulate squared gradients (diagonal Fisher approximation)
            for name, param in trainable_params.items():
                if param.grad is not None:
                    fisher[name] += param.grad.data.clone() ** 2

            n_samples += batch_size

        # Normalize
        for name in fisher:
            fisher[name] /= max(n_samples, 1)

        if self.normalize_fisher:
            max_fisher = max(f.max().item() for f in fisher.values())
            if max_fisher > 0:
                for name in fisher:
                    fisher[name] /= max_fisher

        # Store Fisher and optimal parameters
        if self.variant == "online_ewc":
            self._update_online_fisher(fisher, trainable_params)
        else:
            self.fisher_matrices[domain_name] = fisher
            self.optimal_params[domain_name] = {
                name: param.data.clone()
                for name, param in trainable_params.items()
            }

        logger.info(
            f"Fisher computed: {n_samples} samples, "
            f"{len(fisher)} parameters tracked"
        )

    def _update_online_fisher(
        self,
        new_fisher: Dict[str, torch.Tensor],
        trainable_params: Dict[str, nn.Parameter],
    ):
        """Update running Fisher estimate for Online EWC."""
        if self.running_fisher is None:
            self.running_fisher = new_fisher
            self.running_optimal = {
                name: param.data.clone()
                for name, param in trainable_params.items()
            }
        else:
            for name in new_fisher:
                if name in self.running_fisher:
                    # Ensure device match (running_fisher may have been loaded on CPU)
                    self.running_fisher[name] = (
                        self.gamma * self.running_fisher[name].to(new_fisher[name].device)
                        + new_fisher[name]
                    )
                else:
                    self.running_fisher[name] = new_fisher[name]

            self.running_optimal = {
                name: param.data.clone()
                for name, param in trainable_params.items()
            }

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC penalty term to add to the training loss.

        Returns:
            EWC loss: λ/2 * Σ_i F_i * (θ_i - θ*_i)²
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        trainable_params = self._get_trainable_params(model)

        if self.variant == "online_ewc":
            if self.running_fisher is None:
                return loss
            for name, param in trainable_params.items():
                if name in self.running_fisher and name in self.running_optimal:
                    fisher = self.running_fisher[name].to(param.device)
                    optimal = self.running_optimal[name].to(param.device)
                    loss += (fisher * (param - optimal) ** 2).sum()
        else:
            for domain_name in self.fisher_matrices:
                fisher = self.fisher_matrices[domain_name]
                optimal = self.optimal_params[domain_name]
                for name, param in trainable_params.items():
                    if name in fisher:
                        f = fisher[name].to(param.device)
                        o = optimal[name].to(param.device)
                        loss += (f * (param - o) ** 2).sum()

        return (self.lambda_ewc / 2.0) * loss

    def save(self, path: str, domain_name: str):
        """Save Fisher matrices and optimal params to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_data = {
            "variant": self.variant,
            "lambda_ewc": self.lambda_ewc,
        }

        if self.variant == "online_ewc":
            save_data["running_fisher"] = self.running_fisher
            save_data["running_optimal"] = self.running_optimal
        else:
            save_data["fisher_matrices"] = self.fisher_matrices
            save_data["optimal_params"] = self.optimal_params

        torch.save(save_data, save_dir / f"ewc_{domain_name}.pt")
        logger.info(f"EWC state saved to {save_dir / f'ewc_{domain_name}.pt'}")

    def load(self, path: str, domain_name: str):
        """Load Fisher matrices and optimal params from disk."""
        load_path = Path(path) / f"ewc_{domain_name}.pt"
        if not load_path.exists():
            logger.warning(f"No EWC state found at {load_path}")
            return

        data = torch.load(load_path, map_location="cpu", weights_only=False)

        if data["variant"] == "online_ewc":
            self.running_fisher = data.get("running_fisher")
            self.running_optimal = data.get("running_optimal")
        else:
            self.fisher_matrices = data.get("fisher_matrices", {})
            self.optimal_params = data.get("optimal_params", {})

        logger.info(f"EWC state loaded from {load_path}")
