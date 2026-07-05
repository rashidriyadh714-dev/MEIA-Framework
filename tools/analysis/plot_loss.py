#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Training Convergence Plotter


Generates high-resolution publication-ready vector figures mapping training 
and validation loss convergence curves for the manuscript.
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("FigureGenerator")

def generate_loss_plot() -> None:
    current_dir = Path.cwd()
    project_root = current_dir.parent if current_dir.name == "scripts" else current_dir

    output_dir = project_root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    epochs = [1, 2, 3, 4, 5, 6]
    train_loss = [6.26, 4.74, 3.69, 3.11, 2.70, 2.41]
    val_loss = [5.24, 3.67, 2.85, 2.34, 2.03, 1.89]

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_loss, marker='o', markersize=8, linewidth=2.5, label='Training Loss', color='#2c3e50')
    plt.plot(epochs, val_loss, marker='s', markersize=8, linewidth=2.5, label='Validation Loss', color='#e74c3c')

    plt.xlabel('Epoch', fontweight='bold', fontsize=14)
    plt.ylabel('Loss', fontweight='bold', fontsize=14)
    plt.title('Training vs. Validation Loss (MEIA Framework)', fontweight='bold', fontsize=16)
    plt.xticks(epochs)
    plt.legend(frameon=True, shadow=True, fontsize=12)

    plt.tight_layout()

    png_path = output_dir / 'meia_loss_curve.png'
    pdf_path = output_dir / 'meia_loss_curve.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

    logger.info("======================================================================")
    logger.info(" PUBLICATION FIGURE GENERATION COMPLETE")
    logger.info("======================================================================")
    logger.info(f" Saved PNG Render : {png_path.name}")
    logger.info(f" Saved PDF Render : {pdf_path.name}")
    logger.info(f" Output Directory : {output_dir}")
    logger.info("======================================================================\n")

if __name__ == "__main__":
    generate_loss_plot()
