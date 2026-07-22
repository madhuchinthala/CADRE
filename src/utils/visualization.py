"""
Visualization Utilities
========================
Plotting functions for training metrics, performance matrices,
and benchmark results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def plot_metrics(
    metrics_history: List[Dict],
    save_path: str = "outputs/visualizations",
    filename: str = "training_metrics.png",
):
    """
    Plot training metrics over epochs.

    Args:
        metrics_history: List of dicts with 'epoch', 'train_loss', 'val_score'
        save_path: Directory to save plot
        filename: Output filename
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping plot.")
        return

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = [m["epoch"] for m in metrics_history]
    losses = [m.get("train_loss", 0) for m in metrics_history]
    scores = [m.get("val_score", 0) for m in metrics_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, losses, "b-o", linewidth=2, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, scores, "g-o", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Score")
    ax2.set_title("Validation Score")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Metrics plot saved: {save_dir / filename}")


def plot_performance_matrix(
    perf_matrix: np.ndarray,
    domain_names: List[str],
    save_path: str = "outputs/visualizations",
    filename: str = "performance_matrix.png",
):
    """
    Plot the T×T performance matrix as a heatmap.

    Args:
        perf_matrix: T×T numpy array of performance scores
        domain_names: List of domain names for axis labels
        save_path: Directory to save plot
        filename: Output filename
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed. Skipping plot.")
        return

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        perf_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        xticklabels=domain_names,
        yticklabels=[f"After {d}" for d in domain_names],
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Evaluation Domain")
    ax.set_ylabel("Training Stage")
    ax.set_title("CADRE Performance Matrix")

    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Performance matrix plot saved: {save_dir / filename}")


def plot_benchmark_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str = "outputs/visualizations",
    filename: str = "benchmark_comparison.png",
):
    """
    Plot bar chart comparing methods on BWT, FWT, Plasticity, CDAR.

    Args:
        results: Dict of method_name -> {metric_name: value}
        save_path: Directory to save plot
        filename: Output filename
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping plot.")
        return

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    methods = list(results.keys())
    metrics = ["BWT", "FWT", "Plasticity", "CDAR"]

    x = np.arange(len(metrics))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        values = [results[method].get(m, 0) for m in metrics]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("CADRE-Bench Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Benchmark comparison plot saved: {save_dir / filename}")
