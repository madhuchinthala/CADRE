#!/usr/bin/env python
"""
Generate CADRE Visualizations
==============================
Creates publication-quality visualizations from the CADRE-Bench results:
  1. Performance Matrix Heatmap
  2. CADRE-Bench Metrics Summary
  3. Training Timeline
  4. Parameter Efficiency
  5. Domain-wise Performance Comparison
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "outputs" / "cadre_bench" / "cadre_bench_report.json"
PIPELINE_STATE_PATH = REPO_ROOT / "checkpoints" / "pipeline_state.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d",
    "grid.color": "#21262d",
    "savefig.facecolor": "#0d1117",
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

COLORS = {
    "primary": "#58a6ff",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
    "purple": "#bc8cff",
    "accent": "#79c0ff",
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "muted": "#8b949e",
}

DOMAIN_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff"]
DOMAIN_LABELS = ["US (BDD100K)", "Singapore (nuScenes)", "EU (nuScenes)", "Rainy (BDD100K)"]
DOMAIN_SHORT = ["domain_us", "domain_sg", "domain_eu", "domain_rainy"]


def load_report():
    with open(REPORT_PATH, "r") as f:
        return json.load(f)


def load_pipeline_state():
    with open(PIPELINE_STATE_PATH, "r") as f:
        return json.load(f)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Performance Matrix Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_performance_matrix(report):
    matrix = np.array(report["performance_matrix"])
    T = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors_cmap = ["#1a1e2e", "#1f3a5f", "#2d6a8f", "#3fb950", "#56d364"]
    cmap = LinearSegmentedColormap.from_list("cadre", colors_cmap, N=256)

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    # Annotate cells
    for i in range(T):
        for j in range(T):
            val = matrix[i][j]
            color = "#ffffff" if val > 0.6 else "#8b949e"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=13, fontweight=weight)

    # Diagonal highlight
    for i in range(T):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, linewidth=2,
                              edgecolor="#58a6ff", facecolor="none", linestyle="--")
        ax.add_patch(rect)

    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(DOMAIN_LABELS, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(DOMAIN_LABELS, fontsize=10)
    ax.set_xlabel("Evaluation Domain", fontsize=12, labelpad=10)
    ax.set_ylabel("After Training Stage", fontsize=12, labelpad=10)
    ax.set_title("Performance Matrix R[i,j]\n(Diagonal = learned domain, Off-diagonal = retention/transfer)",
                 pad=15, fontsize=13)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Accuracy", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#e6edf3")

    fig.tight_layout()
    path = OUTPUT_DIR / "performance_matrix_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path.name}")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. CADRE-Bench Metrics Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_metrics_summary(report):
    metrics = report["metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})

    # ── Left: Bar chart ──
    ax = axes[0]
    names = ["BWT\n(Forgetting)", "FWT\n(Transfer)", "Plasticity", "Efficiency"]
    values = [metrics["BWT"], metrics["FWT"], metrics["Plasticity"], metrics["Efficiency"]]
    bar_colors = [
        COLORS["success"] if metrics["BWT"] > -5 else COLORS["danger"],
        COLORS["primary"],
        COLORS["success"],
        COLORS["success"],
    ]

    bars = ax.bar(names, values, color=bar_colors, width=0.6, edgecolor="#30363d", linewidth=1)

    for bar, val in zip(bars, values):
        y_pos = bar.get_height() + 1.5 if val >= 0 else bar.get_height() - 4
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=COLORS["text"])

    ax.axhline(y=0, color=COLORS["border"], linewidth=0.8)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Continual Learning Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim(min(-10, min(values) - 10), max(values) + 10)
    ax.grid(axis="y", alpha=0.3)

    # ── Right: CDAR gauge ──
    ax2 = axes[1]
    cdar = metrics["CDAR"]

    # Create a gauge-like visualization
    theta = np.linspace(np.pi, 0, 100)
    r = 1
    ax2.plot(r * np.cos(theta), r * np.sin(theta), color=COLORS["border"], linewidth=8, solid_capstyle="round")

    # Fill based on CDAR score
    fill_idx = int(cdar * 100)
    theta_fill = np.linspace(np.pi, np.pi - (cdar * np.pi), fill_idx)
    gradient_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(theta_fill)))
    for i in range(len(theta_fill) - 1):
        ax2.plot([r * np.cos(theta_fill[i]), r * np.cos(theta_fill[i + 1])],
                 [r * np.sin(theta_fill[i]), r * np.sin(theta_fill[i + 1])],
                 color=gradient_colors[i], linewidth=8, solid_capstyle="round")

    # Score text
    ax2.text(0, 0.3, f"{cdar:.4f}", ha="center", va="center",
             fontsize=32, fontweight="bold", color=COLORS["success"])
    ax2.text(0, 0.05, "CDAR Score", ha="center", va="center",
             fontsize=13, color=COLORS["muted"])
    ax2.text(0, -0.15, "(Continual Driving\nAdaptation Rating)", ha="center", va="center",
             fontsize=9, color=COLORS["muted"])

    # Scale labels
    ax2.text(-1.05, -0.05, "0", ha="center", fontsize=9, color=COLORS["muted"])
    ax2.text(1.05, -0.05, "1", ha="center", fontsize=9, color=COLORS["muted"])

    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-0.4, 1.2)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Composite Score", fontsize=14, fontweight="bold")

    fig.suptitle("CADRE-Bench Results", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "metrics_summary.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path.name}")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Training Timeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_training_timeline(pipeline_state):
    from datetime import datetime

    stages_order = [
        ("verify_backbone", "Verify Backbone"),
        ("domain_us", "Train: US"),
        ("domain_sg", "Train: Singapore"),
        ("domain_eu", "Train: EU"),
        ("domain_rainy", "Train: Rainy"),
        ("train_router", "Domain Router"),
        ("train_heads", "Output Heads"),
        ("benchmark", "CADRE-Bench"),
    ]

    stage_colors = [
        COLORS["muted"], DOMAIN_COLORS[0], DOMAIN_COLORS[1],
        DOMAIN_COLORS[2], DOMAIN_COLORS[3],
        COLORS["primary"], COLORS["purple"], COLORS["success"],
    ]

    # Extract start/end times from history
    stage_times = {}
    for entry in pipeline_state["history"]:
        name = entry["stage"]
        status = entry["status"]
        t = datetime.fromisoformat(entry["at"])
        if name not in stage_times:
            stage_times[name] = {"starts": [], "end": None}
        if status == "started":
            stage_times[name]["starts"].append(t)
        elif status == "completed":
            stage_times[name]["end"] = t

    fig, ax = plt.subplots(figsize=(12, 5))

    y_positions = []
    labels = []

    for i, (stage_key, stage_label) in enumerate(stages_order):
        y = len(stages_order) - 1 - i
        y_positions.append(y)
        labels.append(stage_label)

        if stage_key in stage_times and stage_times[stage_key]["end"]:
            info = stage_times[stage_key]
            first_start = min(info["starts"])
            end = info["end"]
            duration_min = (end - first_start).total_seconds() / 60

            # Draw bar
            ax.barh(y, duration_min, left=0, height=0.5,
                    color=stage_colors[i], edgecolor="#30363d",
                    alpha=0.85, linewidth=1)

            # Duration label
            if duration_min >= 60:
                dur_text = f"{duration_min / 60:.1f}h"
            elif duration_min >= 1:
                dur_text = f"{duration_min:.0f}m"
            else:
                dur_text = f"{duration_min * 60:.0f}s"

            ax.text(duration_min + 2, y, dur_text,
                    va="center", fontsize=10, color=COLORS["text"], fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Duration (minutes)", fontsize=12)
    ax.set_title("Training Pipeline Timeline", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, None)

    # Add total time annotation
    all_starts = []
    all_ends = []
    for info in stage_times.values():
        all_starts.extend(info["starts"])
        if info["end"]:
            all_ends.append(info["end"])
    if all_starts and all_ends:
        total_min = (max(all_ends) - min(all_starts)).total_seconds() / 60
        total_text = f"Total wall-clock: {total_min / 60:.1f} hours" if total_min > 60 else f"Total: {total_min:.0f} min"
        ax.text(0.98, 0.02, total_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, color=COLORS["muted"],
                style="italic")

    fig.tight_layout()
    path = OUTPUT_DIR / "training_timeline.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path.name}")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Parameter Efficiency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_parameter_efficiency(report):
    param_stats = report["param_stats"]
    backbone = param_stats["backbone_params"]
    adapter = param_stats["adapter_params_per_domain"]
    n_domains = 4
    total_adapters = adapter * n_domains
    overhead_pct = param_stats["overhead_percentage"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: Stacked comparison ──
    ax = axes[0]
    categories = ["Full Retrain\n(per domain)", "CADRE\n(per domain)", "CADRE\n(all 4 domains)"]
    backbone_vals = [backbone / 1e9, backbone / 1e9, backbone / 1e9]
    adapter_vals = [0, adapter / 1e9, total_adapters / 1e9]
    trainable_vals = [backbone / 1e9, adapter / 1e9, total_adapters / 1e9]

    bars1 = ax.bar(categories, backbone_vals, color=COLORS["border"], width=0.5,
                   label="Frozen Backbone", edgecolor="#444")
    bars2 = ax.bar(categories, adapter_vals, bottom=backbone_vals, color=COLORS["success"],
                   width=0.5, label="Trainable (LoRA)", edgecolor="#444")

    # Highlight trainable portion
    ax.text(0, backbone_vals[0] / 2, f"{backbone / 1e9:.1f}B\n(ALL trainable)", ha="center",
            va="center", fontsize=9, color=COLORS["danger"], fontweight="bold")
    ax.text(1, backbone_vals[1] + adapter_vals[1] / 2, f"{adapter / 1e6:.0f}M\n({overhead_pct}%)",
            ha="center", va="center", fontsize=9, color="#ffffff", fontweight="bold")
    ax.text(2, backbone_vals[2] + adapter_vals[2] / 2, f"{total_adapters / 1e6:.0f}M\n({overhead_pct * 4:.1f}%)",
            ha="center", va="center", fontsize=9, color="#ffffff", fontweight="bold")

    ax.set_ylabel("Parameters (Billions)", fontsize=11)
    ax.set_title("Parameter Efficiency: Full Retrain vs CADRE", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    # ── Right: Storage comparison ──
    ax2 = axes[1]
    storage_full = backbone / 1e9 * 2  # ~14 GB in fp16
    storage_adapter = adapter / 1e9 * 2  # ~50 MB in fp16
    storage_4adapters = total_adapters / 1e9 * 2

    items = ["Full Model\n(1 domain)", "1 LoRA\nAdapter", "4 LoRA\nAdapters"]
    sizes_gb = [storage_full, storage_adapter, storage_4adapters]
    bar_colors = [COLORS["danger"], COLORS["success"], COLORS["primary"]]

    bars = ax2.bar(items, sizes_gb, color=bar_colors, width=0.5, edgecolor="#444")

    for bar, val in zip(bars, sizes_gb):
        label = f"{val:.1f} GB" if val >= 1 else f"{val * 1000:.0f} MB"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 label, ha="center", fontsize=11, fontweight="bold", color=COLORS["text"])

    ax2.set_ylabel("Storage Size (GB)", fontsize=11)
    ax2.set_title("Storage: What You Need to Share", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle("Parameter-Efficient Adaptation", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "parameter_efficiency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path.name}")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Domain-wise Performance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_domain_performance(report):
    matrix = np.array(report["performance_matrix"])
    single_task = np.array(report["single_task_bounds"])
    zero_shot = np.array(report["zero_shot_performance"])
    T = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(T)
    width = 0.2

    # Zero-shot baseline
    ax.bar(x - 1.5 * width, zero_shot, width, label="Zero-Shot (before training)",
           color=COLORS["muted"], edgecolor="#444", alpha=0.7)

    # Diagonal (after training that domain)
    diagonal = np.diag(matrix)
    ax.bar(x - 0.5 * width, diagonal, width, label="After Own Training",
           color=COLORS["success"], edgecolor="#444")

    # Final performance (last row of matrix)
    final = matrix[-1]
    ax.bar(x + 0.5 * width, final, width, label="Final (after all training)",
           color=COLORS["primary"], edgecolor="#444")

    # Single-task upper bound
    ax.bar(x + 1.5 * width, single_task, width, label="Single-Task Upper Bound",
           color=COLORS["purple"], edgecolor="#444", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_LABELS, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Domain-wise Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Annotate retention (final vs own-training)
    for i in range(T):
        retention = final[i] - diagonal[i]
        color = COLORS["success"] if retention >= -0.02 else COLORS["warning"]
        symbol = "↑" if retention > 0 else "↓" if retention < -0.01 else "≈"
        ax.text(i + 0.5 * width, final[i] + 0.02, f"{symbol}{abs(retention)*100:.1f}%",
                ha="center", fontsize=8, color=color, fontweight="bold")

    fig.tight_layout()
    path = OUTPUT_DIR / "domain_performance.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path.name}")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Architecture Overview Diagram
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, fontsize=10, alpha=0.85):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="#ffffff", alpha=alpha, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="#ffffff", wrap=True)

    def draw_arrow(x1, y1, x2, y2, color="#58a6ff"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

    # Title
    ax.text(7, 6.6, "CADRE Architecture", ha="center", fontsize=18,
            fontweight="bold", color=COLORS["text"])

    # Input
    draw_box(0.3, 4.5, 2.2, 1.2, "Driving\nImage Input", "#1f3a5f", fontsize=11)

    # Backbone
    draw_box(3.2, 4.5, 2.5, 1.2, "LLaVA-v1.5-7B\n(Frozen Backbone)", "#2d333b", fontsize=10)

    # Domain Router
    draw_box(3.2, 2.5, 2.5, 1.2, "Domain Router\n(Classifier)", "#58a6ff", fontsize=10)

    # LoRA Adapters
    lora_y = 0.5
    for i, (label, color) in enumerate(zip(["LoRA\nUS", "LoRA\nSG", "LoRA\nEU", "LoRA\nRainy"], DOMAIN_COLORS)):
        draw_box(6.5 + i * 1.7, lora_y, 1.4, 1.0, label, color, fontsize=9)

    # EWC + Replay
    draw_box(6.5, 2.5, 1.8, 1.0, "EWC\nRegularizer", "#d29922", fontsize=9)
    draw_box(8.8, 2.5, 1.8, 1.0, "Replay\nBuffer", "#f85149", fontsize=9)

    # Output Heads
    draw_box(6.5, 4.5, 4.5, 1.2, "Multi-Head Output\nWaypoint | Hazard | Regulation | Weather", "#3fb950", fontsize=10)

    # Integration
    draw_box(11.5, 4.5, 2.0, 1.2, "Integration\nLayer\n(Attention)", "#bc8cff", fontsize=9)

    # Arrows
    draw_arrow(2.5, 5.1, 3.2, 5.1)   # input → backbone
    draw_arrow(5.7, 5.1, 6.5, 5.1)   # backbone → heads
    draw_arrow(11.0, 5.1, 11.5, 5.1) # heads → integration
    draw_arrow(4.45, 4.5, 4.45, 3.7)  # backbone → router
    draw_arrow(5.7, 3.1, 6.5, 3.0)   # router → EWC
    draw_arrow(5.7, 3.0, 6.5, 1.0, "#d29922")  # router → adapters

    fig.tight_layout()
    path = OUTPUT_DIR / "architecture_overview.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ {path.name}")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("\n🎨 Generating CADRE visualizations...\n")

    report = load_report()
    pipeline_state = load_pipeline_state()

    plot_performance_matrix(report)
    plot_metrics_summary(report)
    plot_training_timeline(pipeline_state)
    plot_parameter_efficiency(report)
    plot_domain_performance(report)
    plot_architecture_diagram()

    print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
