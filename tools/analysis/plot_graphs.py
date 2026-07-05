#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Training Convergence Visualization

Parses JSON metric logs across multiple initialization seeds to generate 
publication-ready vector figures of training and validation convergence.
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GraphGenerator")

def generate_bmvc_plots() -> None:
    current_dir = Path.cwd()
    project_root = current_dir.parent if current_dir.name == "scripts" else current_dir

    training_path = project_root / "checkpoints" 
    output_dir = project_root / "outputs" / "figures"
    
    logger.info("Initializing Graph Generator Engine...")
    logger.info(f"Targeting checkpoint directory: {training_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    # Extract data across standardized seeds
    for seed in [1, 2, 3, 41, 42, 43]:
        metric_file = training_path / f"seed_{seed}" / "metrics.json"
        if metric_file.exists():
            logger.info(f"Located metric tensor for Seed {seed}")
            with open(metric_file, "r", encoding="utf-8") as f:
                all_metrics.append(json.load(f))
                
    if not all_metrics:
        logger.error(f"Failed to locate valid metrics.json files within {training_path}")
        return

    epochs = [ep["epoch"] for ep in all_metrics[0]["epochs"]]
    last_epoch = epochs[-1]
    
    # Calculate dimensional means
    train_loss = np.mean([[ep["train_loss"] for ep in run["epochs"]] for run in all_metrics], axis=0)
    val_loss = np.mean([[ep["val_loss"] for ep in run["epochs"]] for run in all_metrics], axis=0)

    # ---------------------------------------------------------
    # Plot 1: Standard Convergence Curve
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o', linewidth=2.5, color='#1f77b4')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='s', linewidth=2.5, color='#d62728')
    plt.title('Training vs Validation Convergence\n(Averaged across seeds)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Tri-Task Focal Loss', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    loss_path = output_dir / 'meia_loss_curve.png'
    plt.savefig(loss_path, dpi=300)
    plt.close()
    logger.info(f"Rendered loss convergence curve to {loss_path.name}")
    
    # ---------------------------------------------------------
    # Plot 2: Tri-Task Validation & Test Performance
    # ---------------------------------------------------------
    val_emo = np.mean([[ep["emotion_accuracy"] for ep in run["epochs"]] for run in all_metrics], axis=0)
    val_int = np.mean([[ep["intention_macro_f1"] for ep in run["epochs"]] for run in all_metrics], axis=0)
    val_act = np.mean([[ep["action_macro_f1"] for ep in run["epochs"]] for run in all_metrics], axis=0)

    test_emo = np.mean([run.get("test_emotion_accuracy", 0) for run in all_metrics])
    test_int = np.mean([run.get("test_intention_f1", 0) for run in all_metrics])
    test_act = np.mean([run.get("test_action_f1", 0) for run in all_metrics])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_emo, label='Val Emotion (Acc)', marker='o', color='#2ca02c', linewidth=2.5, alpha=0.7)
    plt.plot(epochs, val_int, label='Val Intention (F1)', marker='s', color='#ff7f0e', linewidth=2.5, alpha=0.7)
    plt.plot(epochs, val_act, label='Val Action (F1)', marker='^', color='#9467bd', linewidth=2.5, alpha=0.7)
    
    plt.plot(last_epoch, test_emo, marker='*', markersize=18, color='darkgreen', label='Test Emotion (Acc)', linestyle='None', zorder=5)
    plt.plot(last_epoch, test_int, marker='*', markersize=18, color='darkorange', label='Test Intention (F1)', linestyle='None', zorder=5)
    plt.plot(last_epoch, test_act, marker='*', markersize=18, color='indigo', label='Test Action (F1)', linestyle='None', zorder=5)

    plt.title('Tri-Task Validation & Test Performance Map\n(Averaged across seeds)', fontsize=15, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Metric Score', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    metric_path = output_dir / 'meia_performance_curve.png'
    plt.savefig(metric_path, dpi=300)
    plt.close()
    logger.info(f"Rendered tri-task performance curve to {metric_path.name}")
    logger.info("Visualization routine complete.")

if __name__ == "__main__":
    generate_bmvc_plots()
