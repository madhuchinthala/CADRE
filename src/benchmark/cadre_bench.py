"""
PART 7: CADRE-Bench — Benchmark Protocol
=========================================
Comprehensive evaluation suite for continual adaptation in driving.
Measures forgetting (BWT), transfer (FWT), plasticity, and the
composite CDAR score across sequential driving domains.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


class ContinualDrivingMetrics:
    """
    Computes all continual learning metrics for driving adaptation.

    Metrics:
    - BWT (Backward Transfer): How much learning new domains hurts old ones
      BWT = (1/T-1) * Σ_{i=1}^{T-1} (R_{T,i} - R_{i,i})
      Negative = forgetting, zero = perfect retention

    - FWT (Forward Transfer): How much old training helps new domains
      FWT = (1/T-1) * Σ_{i=2}^{T} (R_{i-1,i} - R_{0,i})
      Positive = beneficial transfer

    - Plasticity: How well the model learns each new domain
      Plasticity = (1/T) * Σ_{i=1}^{T} R_{i,i} / R*_{i}
      where R*_{i} is the single-task upper bound

    - CDAR (Continual Driving Adaptation Rating): Composite score
      CDAR = w_s * Stability + w_p * Plasticity + w_t * Transfer + w_e * Efficiency
    """

    def __init__(
        self,
        cdar_weights: Dict[str, float] = None,
    ):
        if cdar_weights is None:
            cdar_weights = {
                "stability": 0.3,
                "plasticity": 0.3,
                "transfer": 0.2,
                "efficiency": 0.2,
            }
        self.cdar_weights = cdar_weights

        # Performance matrix R[i][j] = performance on domain j after training on domain i
        # Rows: training stage, Columns: evaluation domain
        self.perf_matrix: Optional[np.ndarray] = None

        # Single-task upper bounds for each domain
        self.single_task_bounds: Optional[np.ndarray] = None

        # Zero-shot baselines (before any domain-specific training)
        self.zero_shot_perf: Optional[np.ndarray] = None

        # Parameter efficiency
        self.total_backbone_params: int = 0
        self.adapter_params_per_domain: int = 0

    def set_performance_matrix(self, matrix: np.ndarray):
        """
        Set the T×T performance matrix.

        R[i][j] = performance on domain j after training through domain i.
        """
        assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], \
            "Performance matrix must be square (T×T)"
        self.perf_matrix = matrix
        logger.info(f"Performance matrix set: {matrix.shape}")

    def set_baselines(
        self,
        single_task: np.ndarray,
        zero_shot: np.ndarray,
    ):
        """Set single-task upper bounds and zero-shot baselines."""
        self.single_task_bounds = single_task
        self.zero_shot_perf = zero_shot

    def set_param_counts(self, backbone: int, adapter_per_domain: int):
        """Set parameter counts for efficiency computation."""
        self.total_backbone_params = backbone
        self.adapter_params_per_domain = adapter_per_domain

    def compute_bwt(self) -> float:
        """
        Backward Transfer (BWT).

        Measures average performance change on old domains after learning new ones.
        BWT = (1/(T-1)) * Σ_{i=1}^{T-1} (R[T-1][i] - R[i][i])
        """
        T = self.perf_matrix.shape[0]
        if T < 2:
            return 0.0

        bwt = 0.0
        for i in range(T - 1):
            bwt += self.perf_matrix[T - 1][i] - self.perf_matrix[i][i]

        return bwt / (T - 1)

    def compute_fwt(self) -> float:
        """
        Forward Transfer (FWT).

        Measures how much prior training helps on unseen domains.
        FWT = (1/(T-1)) * Σ_{i=1}^{T-1} (R[i-1][i] - R_zero_shot[i])
        """
        T = self.perf_matrix.shape[0]
        if T < 2 or self.zero_shot_perf is None:
            return 0.0

        fwt = 0.0
        for i in range(1, T):
            fwt += self.perf_matrix[i - 1][i] - self.zero_shot_perf[i]

        return fwt / (T - 1)

    def compute_plasticity(self) -> float:
        """
        Plasticity.

        Measures how well the model learns each new domain relative to
        the single-task upper bound.
        Plasticity = (1/T) * Σ_{i=0}^{T-1} R[i][i] / R*[i]
        """
        T = self.perf_matrix.shape[0]
        if self.single_task_bounds is None:
            # Without upper bounds, use raw diagonal average
            return np.mean(np.diag(self.perf_matrix))

        plasticity = 0.0
        for i in range(T):
            if self.single_task_bounds[i] > 0:
                plasticity += self.perf_matrix[i][i] / self.single_task_bounds[i]

        return plasticity / T

    def compute_efficiency(self) -> float:
        """
        Parameter Efficiency.

        Ratio of adapter parameters to total backbone parameters.
        Lower is better — normalized to [0, 1] where 1 = maximally efficient.

        efficiency = 1 - (adapter_params / backbone_params)
        """
        if self.total_backbone_params == 0:
            return 0.0

        ratio = self.adapter_params_per_domain / self.total_backbone_params
        return 1.0 - min(ratio, 1.0)

    def compute_cdar(self) -> float:
        """
        Continual Driving Adaptation Rating (CDAR).

        Composite score combining stability, plasticity, transfer, and efficiency.
        CDAR = w_s * Stability + w_p * Plasticity + w_t * Transfer + w_e * Efficiency

        Where:
        - Stability = 1 + BWT (maps BWT from [-1,0] to [0,1], approximately)
        - Plasticity = plasticity score
        - Transfer = sigmoid(FWT) to map to [0,1]
        - Efficiency = 1 - param_ratio
        """
        bwt = self.compute_bwt()
        fwt = self.compute_fwt()
        plasticity = self.compute_plasticity()
        efficiency = self.compute_efficiency()

        # Normalize to [0, 1] range
        stability = max(0, 1 + bwt)  # BWT in [-1, 0] → stability in [0, 1]
        transfer = 1 / (1 + np.exp(-fwt * 10))  # sigmoid scaling

        cdar = (
            self.cdar_weights["stability"] * stability
            + self.cdar_weights["plasticity"] * plasticity
            + self.cdar_weights["transfer"] * transfer
            + self.cdar_weights["efficiency"] * efficiency
        )

        return cdar

    def full_report(self) -> Dict:
        """Generate complete metrics report."""
        bwt = self.compute_bwt()
        fwt = self.compute_fwt()
        plasticity = self.compute_plasticity()
        efficiency = self.compute_efficiency()
        cdar = self.compute_cdar()

        report = {
            "metrics": {
                "BWT": round(bwt * 100, 2),
                "FWT": round(fwt * 100, 2),
                "Plasticity": round(plasticity * 100, 2),
                "Efficiency": round(efficiency * 100, 2),
                "CDAR": round(cdar, 4),
            },
            "performance_matrix": self.perf_matrix.tolist() if self.perf_matrix is not None else None,
            "single_task_bounds": self.single_task_bounds.tolist() if self.single_task_bounds is not None else None,
            "zero_shot_performance": self.zero_shot_perf.tolist() if self.zero_shot_perf is not None else None,
            "param_stats": {
                "backbone_params": self.total_backbone_params,
                "adapter_params_per_domain": self.adapter_params_per_domain,
                "overhead_percentage": round(
                    self.adapter_params_per_domain / max(self.total_backbone_params, 1) * 100, 3
                ),
            },
        }

        return report


class CADREBench:
    """
    CADRE-Bench: Full benchmark protocol for continual driving adaptation.

    Protocol:
    1. Evaluate zero-shot performance on all domains
    2. For each domain in sequence:
       a. Train on domain with EWC + Replay
       b. Evaluate on ALL domains (builds the performance matrix)
    3. Compute BWT, FWT, Plasticity, CDAR
    4. Generate report with visualizations
    """

    def __init__(
        self,
        domains: List[str],
        config_path: str = "configs/benchmark_config.yaml",
        output_dir: str = "outputs/cadre_bench",
    ):
        self.domains = domains
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["benchmark"]

        self.metrics = ContinualDrivingMetrics(
            cdar_weights=self.config.get("cdar_weights"),
        )

        self.T = len(domains)
        self.perf_matrix = np.zeros((self.T, self.T))

        logger.info(f"CADRE-Bench initialized: {self.T} domains, output → {self.output_dir}")

    def record_performance(self, train_stage: int, eval_domain: int, score: float):
        """
        Record a performance score in the matrix.

        Args:
            train_stage: Which training stage (0-indexed)
            eval_domain: Which domain was evaluated (0-indexed)
            score: Performance score (0-1)
        """
        self.perf_matrix[train_stage][eval_domain] = score

    def run_evaluation(
        self,
        model,
        eval_dataloaders: Dict[str, torch.utils.data.DataLoader],
        train_stage: int,
        task_metric: str = "accuracy",
    ):
        """
        Evaluate model on all domains after a training stage.

        Args:
            model: The current model state
            eval_dataloaders: Dict mapping domain names to DataLoaders
            train_stage: Current training stage index
            task_metric: Which metric to compute
        """
        model.eval()

        for domain_idx, domain_name in enumerate(self.domains):
            if domain_name not in eval_dataloaders:
                continue

            dataloader = eval_dataloaders[domain_name]
            score = self._evaluate_domain(model, dataloader, task_metric)
            self.record_performance(train_stage, domain_idx, score)

            logger.info(
                f"Stage {train_stage} → {domain_name}: {score:.4f}"
            )

    def _evaluate_domain(self, model, dataloader, metric: str) -> float:
        """Evaluate model on a single domain."""
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    outputs = model(**{k: v.cuda() for k, v in batch.items()})
                else:
                    inputs, targets = batch
                    outputs = model(inputs.cuda())

                # Simplified — adapt based on your actual output format
                if hasattr(outputs, "logits"):
                    preds = outputs.logits.argmax(dim=-1)
                    if isinstance(batch, dict) and "labels" in batch:
                        targets = batch["labels"].cuda()
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

        return correct / max(total, 1)

    def finalize(
        self,
        single_task_bounds: np.ndarray = None,
        zero_shot_perf: np.ndarray = None,
        backbone_params: int = 7_063_000_000,
        adapter_params: int = 24_720_000,
    ) -> Dict:
        """
        Finalize benchmark: compute all metrics and save report.

        Args:
            single_task_bounds: Upper-bound performance for each domain
            zero_shot_perf: Zero-shot performance before any training
            backbone_params: Total backbone parameters
            adapter_params: LoRA adapter parameters per domain

        Returns:
            Full metrics report dict
        """
        self.metrics.set_performance_matrix(self.perf_matrix)
        self.metrics.set_param_counts(backbone_params, adapter_params)

        if single_task_bounds is not None:
            self.metrics.set_baselines(
    single_task_bounds,
    zero_shot_perf if zero_shot_perf is not None else np.zeros(self.T)
)

        report = self.metrics.full_report()

        # Save report
        report_path = self.output_dir / "cadre_bench_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("  CADRE-Bench Results")
        print("=" * 60)
        for metric, value in report["metrics"].items():
            unit = "%" if metric != "CDAR" else ""
            print(f"  {metric:.<20} {value}{unit}")
        print("=" * 60)
        print(f"\n  Report saved to: {report_path}")
        print("=" * 60 + "\n")

        return report


# ── CLI Entry Point ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run CADRE-Bench")
    parser.add_argument("--config", default="configs/benchmark_config.yaml")
    parser.add_argument("--domains", type=str, required=True)
    parser.add_argument("--output_dir", default="outputs/cadre_bench")
    args = parser.parse_args()

    domains = args.domains.split(",")

    # Demo with synthetic data
    print("Running CADRE-Bench demo with synthetic performance data...\n")

    bench = CADREBench(
        domains=domains,
        config_path=args.config,
        output_dir=args.output_dir,
    )

    # Simulated performance matrix (from paper results)
    T = len(domains)
    demo_matrix = np.array([
        [0.94, 0.32, 0.28, 0.25],  # After training domain_us
        [0.93, 0.95, 0.41, 0.38],  # After training domain_sg
        [0.92, 0.94, 0.96, 0.52],  # After training domain_eu
        [0.925, 0.935, 0.955, 0.97], # After training domain_rainy
    ])[:T, :T]

    bench.perf_matrix = demo_matrix

    report = bench.finalize(
        single_task_bounds=np.array([0.97, 0.96, 0.98, 0.97])[:T],
        zero_shot_perf=np.array([0.20, 0.18, 0.22, 0.15])[:T],
        backbone_params=7_063_000_000,
        adapter_params=24_720_000,
    )
