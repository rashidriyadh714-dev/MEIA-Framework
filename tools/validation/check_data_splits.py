#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Dataset Split & Class Distribution Analysis


This script analyzes the class distribution across all data subsets. It is 
used to quantify the long-tailed imbalance for the Emotion (Single-Label), 
Intention (Multi-Label), and Action (Multi-Label) taxonomies.
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.cloud_datasets import get_cloud_dataloaders

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DistributionAudit")


def count_split_distribution(dataloader, split_name: str) -> tuple:
    logger.info(f"Scanning target tensors within {split_name} Split...")
    
    emo_counts = {}
    int_counts = None
    act_counts = None

    for batch in dataloader:
        # Single-Label Evaluation: Emotion
        emos = batch["emotion_labels"].cpu().numpy()
        for e in emos:
            emo_counts[e] = emo_counts.get(e, 0) + 1

        # Multi-Label Evaluation: Intention
        ints = batch["intention_labels"].cpu().numpy()
        if int_counts is None:
            int_counts = np.zeros(ints.shape[1])
        int_counts += ints.sum(axis=0)

        # Multi-Label Evaluation: Action
        acts = batch["action_labels"].cpu().numpy()
        if act_counts is None:
            act_counts = np.zeros(acts.shape[1])
        act_counts += acts.sum(axis=0)

    return emo_counts, int_counts, act_counts


def run_distribution_check() -> None:
    logger.info("Initializing dataloaders for class distribution analysis...")
    train_loader, val_loader, test_loader = get_cloud_dataloaders(
        batch_size=32, eval_batch_size=32, num_workers=4, distributed=False, sources=["mine_curated", "fane"]
    )

    splits = {
        "TRAINING": train_loader,
        "VALIDATION": val_loader,
        "TESTING": test_loader
    }

    results = {}

    for name, loader in splits.items():
        if loader is not None:
            e_count, i_count, a_count = count_split_distribution(loader, name)
            results[name] = {"Emotion": e_count, "Intention": i_count, "Action": a_count}

    logger.info("======================================================================")
    logger.info(" CLASS DISTRIBUTION & IMBALANCE REPORT")
    logger.info("======================================================================")

    for split_name, data in results.items():
        logger.info(f"--- {split_name} SET ---")
        
        logger.info("Taxonomy: Emotion (Single-Label)")
        for class_id in sorted(data["Emotion"].keys()):
            logger.info(f"  Class {class_id:02d}: {data['Emotion'][class_id]:>6} samples")

        logger.info("Taxonomy: Intention (Multi-Label)")
        for class_id, count in enumerate(data["Intention"]):
            logger.info(f"  Class {class_id:02d}: {int(count):>6} instances")

        logger.info("Taxonomy: Action (Multi-Label)")
        for class_id, count in enumerate(data["Action"]):
            logger.info(f"  Class {class_id:02d}: {int(count):>6} instances")
            
        logger.info("----------------------------------------------------------------------")


if __name__ == "__main__":
    run_distribution_check()
