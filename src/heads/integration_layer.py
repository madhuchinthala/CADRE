"""
PART 6: Output Heads + Integration Layer — Multi-Head Fusion
=============================================================
Contains all 4 task heads and the integration layer that fuses them:
  6a: WaypointHead  — future trajectory prediction
  6b: HazardHead    — obstacle/hazard classification
  6c: RegulationHead — traffic regulation parsing
  6d: WeatherHead   — weather/visibility classification
  6e: IntegrationLayer — attention-based fusion of all heads

FIX: Added --max-samples-per-epoch and tqdm progress bar to HeadsTrainer,
matching continual_trainer.py's pattern. Without this, each epoch
processes the entire combined 4-domain dataset uncapped.
"""

import itertools
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# PART 6a: Waypoint Prediction Head
# ──────────────────────────────────────────────
class WaypointHead(nn.Module):
    """
    Predicts future trajectory waypoints from VLA features.
    Output: [B, num_future_steps, 2] — (x, y) coordinates
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 512,
                 num_future_steps: int = 12, coordinate_dim: int = 2):
        super().__init__()
        self.num_steps = num_future_steps
        self.coord_dim = coordinate_dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_future_steps * coordinate_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.head(features)
        return out.view(-1, self.num_steps, self.coord_dim)


# ──────────────────────────────────────────────
# PART 6b: Hazard Detection Head
# ──────────────────────────────────────────────
class HazardHead(nn.Module):
    """
    Classifies detected hazards in the driving scene.
    Output: [B, num_classes] — logits per hazard class
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 256,
                 num_classes: int = 8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


# ──────────────────────────────────────────────
# PART 6c: Regulation Parsing Head
# ──────────────────────────────────────────────
class RegulationHead(nn.Module):
    """
    Identifies applicable traffic regulations.
    Output: [B, num_classes] — logits per regulation type
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 256,
                 num_classes: int = 15):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


# ──────────────────────────────────────────────
# PART 6d: Weather Classification Head
# ──────────────────────────────────────────────
class WeatherHead(nn.Module):
    """
    Classifies current weather/visibility conditions.
    Output: [B, num_classes] — logits per weather type
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 128,
                 num_classes: int = 6):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


# ──────────────────────────────────────────────
# PART 6e: Integration Layer
# ──────────────────────────────────────────────
class IntegrationLayer(nn.Module):
    """
    Fuses all 4 head outputs into a unified driving decision.

    Fusion methods:
    - "concat":    Simple concatenation + linear projection
    - "attention": Cross-attention between head outputs
    - "gated":     Gated fusion with learned importance weights

    The attention method is recommended — it allows the model to
    dynamically weight head importance based on the current scenario
    (e.g., hazard head matters more in construction zones).
    """

    def __init__(
        self,
        fusion_method: str = "attention",
        fusion_dim: int = 512,
        dropout: float = 0.1,
        waypoint_dim: int = 24,   # 12 steps * 2 coords
        hazard_dim: int = 8,
        regulation_dim: int = 15,
        weather_dim: int = 6,
    ):
        super().__init__()
        self.fusion_method = fusion_method
        self.total_input_dim = waypoint_dim + hazard_dim + regulation_dim + weather_dim

        if fusion_method == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(self.total_input_dim, fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim),
            )

        elif fusion_method == "attention":
            # Project each head output to same dim, then cross-attend
            self.proj_waypoint = nn.Linear(waypoint_dim, fusion_dim)
            self.proj_hazard = nn.Linear(hazard_dim, fusion_dim)
            self.proj_regulation = nn.Linear(regulation_dim, fusion_dim)
            self.proj_weather = nn.Linear(weather_dim, fusion_dim)

            self.attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(fusion_dim)
            self.ffn = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim * 2, fusion_dim),
            )

        elif fusion_method == "gated":
            self.gates = nn.ModuleDict({
                "waypoint": nn.Sequential(nn.Linear(waypoint_dim, 1), nn.Sigmoid()),
                "hazard": nn.Sequential(nn.Linear(hazard_dim, 1), nn.Sigmoid()),
                "regulation": nn.Sequential(nn.Linear(regulation_dim, 1), nn.Sigmoid()),
                "weather": nn.Sequential(nn.Linear(weather_dim, 1), nn.Sigmoid()),
            })
            self.projections = nn.ModuleDict({
                "waypoint": nn.Linear(waypoint_dim, fusion_dim),
                "hazard": nn.Linear(hazard_dim, fusion_dim),
                "regulation": nn.Linear(regulation_dim, fusion_dim),
                "weather": nn.Linear(weather_dim, fusion_dim),
            })
            self.output = nn.Linear(fusion_dim, fusion_dim)

        logger.info(f"Integration layer: {fusion_method} fusion, dim={fusion_dim}")

    def forward(
        self,
        waypoint_out: torch.Tensor,
        hazard_out: torch.Tensor,
        regulation_out: torch.Tensor,
        weather_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse all head outputs.

        Args:
            waypoint_out: [B, num_steps * coord_dim] flattened waypoints
            hazard_out: [B, num_hazard_classes]
            regulation_out: [B, num_regulation_classes]
            weather_out: [B, num_weather_classes]

        Returns:
            Fused representation [B, fusion_dim]
        """
        if self.fusion_method == "concat":
            combined = torch.cat([waypoint_out, hazard_out, regulation_out, weather_out], dim=-1)
            return self.fusion(combined)

        elif self.fusion_method == "attention":
            # Project each head to fusion_dim: [B, fusion_dim]
            w = self.proj_waypoint(waypoint_out).unsqueeze(1)
            h = self.proj_hazard(hazard_out).unsqueeze(1)
            r = self.proj_regulation(regulation_out).unsqueeze(1)
            v = self.proj_weather(weather_out).unsqueeze(1)

            # Stack as sequence: [B, 4, fusion_dim]
            tokens = torch.cat([w, h, r, v], dim=1)

            # Self-attention across heads
            attended, _ = self.attention(tokens, tokens, tokens)
            attended = self.norm(attended + tokens)  # residual
            output = self.ffn(attended)

            # Pool across heads: [B, fusion_dim]
            return output.mean(dim=1)

        elif self.fusion_method == "gated":
            gate_w = self.gates["waypoint"](waypoint_out)
            gate_h = self.gates["hazard"](hazard_out)
            gate_r = self.gates["regulation"](regulation_out)
            gate_v = self.gates["weather"](weather_out)

            fused = (
                gate_w * self.projections["waypoint"](waypoint_out)
                + gate_h * self.projections["hazard"](hazard_out)
                + gate_r * self.projections["regulation"](regulation_out)
                + gate_v * self.projections["weather"](weather_out)
            )
            return self.output(fused)


class MultiHeadDrivingModel(nn.Module):
    """
    Complete multi-head driving model combining all 4 heads + integration.
    Sits on top of the VLA backbone + LoRA adapter output.
    """

    def __init__(self, vla_output_dim: int = 4096, config: dict = None):
        super().__init__()

        self.waypoint_head = WaypointHead(input_dim=vla_output_dim)
        self.hazard_head = HazardHead(input_dim=vla_output_dim)
        self.regulation_head = RegulationHead(input_dim=vla_output_dim)
        self.weather_head = WeatherHead(input_dim=vla_output_dim)

        self.integration = IntegrationLayer(
            fusion_method="attention",
            waypoint_dim=24,
            hazard_dim=8,
            regulation_dim=15,
            weather_dim=6,
        )

    def forward(self, vla_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run all heads and fuse.

        Args:
            vla_features: Output from VLA backbone + LoRA [B, vla_dim]

        Returns:
            Dict with keys: waypoints, hazards, regulations, weather, fused
        """
        waypoints = self.waypoint_head(vla_features)
        hazards = self.hazard_head(vla_features)
        regulations = self.regulation_head(vla_features)
        weather = self.weather_head(vla_features)

        waypoints_flat = waypoints.view(waypoints.size(0), -1)
        fused = self.integration(waypoints_flat, hazards, regulations, weather)

        return {
            "waypoints": waypoints,        # [B, 12, 2]
            "hazards": hazards,             # [B, 8]
            "regulations": regulations,     # [B, 15]
            "weather": weather,             # [B, 6]
            "fused": fused,                 # [B, 512]
        }


class HeadsTrainer:
    """Trainer for multi-head driving model."""

    def __init__(self, model: MultiHeadDrivingModel, lr: float = 1e-3, weight_decay: float = 0.01,
                 max_samples_per_epoch: int = None):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.max_samples_per_epoch = max_samples_per_epoch

    def train_epoch(self, dataloader, device: str = "cuda", epoch: int = 0) -> float:
        self.model.train()
        total_loss = 0
        n_batches = 0

        # Limit batches per epoch if max_samples_per_epoch is set
        max_batches = None
        if self.max_samples_per_epoch is not None:
            batch_size = dataloader.batch_size or 1
            max_batches = max(1, self.max_samples_per_epoch // batch_size)
            logger.info(
                f"Epoch {epoch+1}: limiting to {max_batches} batches "
                f"(~{max_batches * batch_size} samples)"
            )

        iterable = dataloader
        total_for_bar = len(dataloader)
        if max_batches is not None:
            iterable = itertools.islice(dataloader, max_batches)
            total_for_bar = max_batches

        pbar = tqdm(iterable, desc=f"Heads Epoch {epoch+1}", total=total_for_bar)

        for batch in pbar:
            if not isinstance(batch, dict):
                continue
            pixel_values = batch.get("pixel_values")
            if pixel_values is None or not isinstance(pixel_values, torch.Tensor):
                continue

            pixel_values = pixel_values.to(device)
            B = pixel_values.shape[0]

            # Pool / adapt pixel values [B, C, H, W] to 4096-dim VLA feature representation
            feat = F.adaptive_avg_pool2d(pixel_values, (64, 64)).view(B, -1)
            if feat.shape[-1] != 4096:
                feat = F.interpolate(feat.unsqueeze(1), size=4096, mode="linear").squeeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(feat)

            # Compute losses
            loss = torch.tensor(0.0, device=device)
            if "waypoints" in batch and isinstance(batch["waypoints"], torch.Tensor):
                loss += self.mse_loss(outputs["waypoints"], batch["waypoints"].to(device))
            if "hazard" in batch and isinstance(batch["hazard"], torch.Tensor):
                loss += self.ce_loss(outputs["hazards"], batch["hazard"].to(device))
            if "regulation" in batch and isinstance(batch["regulation"], torch.Tensor):
                loss += self.ce_loss(outputs["regulations"], batch["regulation"].to(device))
            if "weather" in batch and isinstance(batch["weather"], torch.Tensor):
                loss += self.ce_loss(outputs["weather"], batch["weather"].to(device))

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({"loss": f"{total_loss/n_batches:.4f}"})

        return total_loss / max(n_batches, 1)

    def save(self, path: str = "checkpoints/heads"):
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path / "multi_head_model.pt")
        logger.info(f"Multi-head model saved to {save_path / 'multi_head_model.pt'}")

    def save_checkpoint(self, path: str, epoch: int):
        """Save training checkpoint for resuming (model + optimizer + epoch)."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path / "heads_checkpoint.pt")
        logger.debug(f"Heads checkpoint saved: will resume from epoch {epoch + 1}")

    def load_checkpoint(self, path: str, device: str = "cpu"):
        """
        Load training checkpoint. Returns the epoch to resume from, or 0 if
        no checkpoint exists.
        """
        ckpt_path = Path(path) / "heads_checkpoint.pt"
        if not ckpt_path.exists():
            return 0
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            logger.info(f"✅ Loaded heads checkpoint: will resume from epoch {epoch + 1}")
            return epoch
        except Exception as e:
            logger.warning(f"⚠️  Failed to load heads checkpoint: {e}")
            return 0

    def delete_checkpoint(self, path: str):
        """Delete training checkpoint after successful completion."""
        ckpt_path = Path(path) / "heads_checkpoint.pt"
        if ckpt_path.exists():
            try:
                ckpt_path.unlink()
                logger.debug(f"Deleted heads checkpoint: {ckpt_path}")
            except Exception as e:
                logger.warning(f"Failed to delete heads checkpoint: {e}")


# ── CLI Entry Point ──
if __name__ == "__main__":
    import argparse
    import yaml
    from torch.utils.data import DataLoader, ConcatDataset
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train output heads and integration layer")
    parser.add_argument("--config", default="configs/heads_config.yaml")
    parser.add_argument("--base_config", default="configs/base_config.yaml")
    parser.add_argument("--heads", default="waypoint,hazard,regulation,weather")
    parser.add_argument("--domains", default="domain_us,domain_sg,domain_eu,domain_rainy")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-samples-per-epoch", type=int, default=3000,
        help="Limit training samples per epoch (default 3000). Prevents each "
             "epoch from processing the entire combined dataset.",
    )
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from last checkpoint if available (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh (do not resume)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    domain_list = [d.strip() for d in args.domains.split(",") if d.strip()]

    model = MultiHeadDrivingModel(vla_output_dim=4096).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"MultiHeadDrivingModel initialized with {total_params:,} parameters.")

    from src.data.dataloader import get_dataloader

    datasets = []
    for domain_name in domain_list:
        try:
            dl = get_dataloader(args.base_config, domain_name, split="train", batch_size=16)
            datasets.append(dl.dataset)
        except Exception as e:
            logger.warning(f"Could not load dataset for domain '{domain_name}': {e}")

    if datasets:
        combined = ConcatDataset(datasets)
        train_loader = DataLoader(combined, batch_size=16, shuffle=True)

        trainer = HeadsTrainer(model, max_samples_per_epoch=args.max_samples_per_epoch)
        checkpoint_dir = "checkpoints/heads"

        # Resume from checkpoint if available
        start_epoch = 0
        if args.resume:
            start_epoch = trainer.load_checkpoint(checkpoint_dir, device=device)

        if start_epoch >= args.epochs:
            logger.info(f"Heads training already completed ({start_epoch}/{args.epochs} epochs). Skipping.")
        else:
            logger.info(f"Training multi-head model on {len(combined)} samples for epochs {start_epoch+1}..{args.epochs}...")
            if args.max_samples_per_epoch:
                logger.info(f"  (capped at {args.max_samples_per_epoch} samples per epoch)")

            for epoch in range(start_epoch, args.epochs):
                avg_loss = trainer.train_epoch(train_loader, device=device, epoch=epoch)
                logger.info(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")

                # Save checkpoint after each epoch
                trainer.save_checkpoint(checkpoint_dir, epoch + 1)

            # Save final model weights and clean up checkpoint
            trainer.save(checkpoint_dir)
            trainer.delete_checkpoint(checkpoint_dir)
    else:
        logger.warning("No datasets loaded; saving initialized multi-head model weights.")
        save_path = Path("checkpoints/heads")
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / "multi_head_model.pt")
        logger.info(f"Multi-head model saved to {save_path / 'multi_head_model.pt'}")

