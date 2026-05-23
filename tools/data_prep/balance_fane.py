#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: FANE Dataset Partitioning Utility
Author: Imad Gohar and Rashid Riyadh, et al.
Institution: Sunway University, Malaysia

This script ensures reproducible, mathematically deterministic 80/10/10 
data splits (train/validation/test) for the FANE dataset annotations to 
prevent data leakage during multi-seed training loops.
"""

import json
import logging
import random
import sys
from pathlib import Path

# Configure academic-standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("BalanceFANE")

def balance_splits(seed: int = 42) -> None:
    """
    Loads the distilled FANE annotations, shuffles them using a fixed seed,
    and strictly partitions them into train (80%), validation (10%), and test (10%).
    """
    # Resolve project root assuming script is located at tools/data_prep/balance_fane.py
    project_root = Path(__file__).resolve().parent.parent.parent
    target_json = project_root / "data" / "fane" / "distilled_annotations.json"

    if not target_json.exists():
        logger.error(f"Missing annotation file at: {target_json}")
        sys.exit(1)

    logger.info(f"Loading FANE dataset annotations from {target_json.name}...")
    
    try:
        with open(target_json, "r", encoding="utf-8") as f:
            fane_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON: {e}")
        sys.exit(1)

    # 1. Deterministic shuffling to guarantee reproducible splits across training runs
    random.seed(seed)
    random.shuffle(fane_data)

    total_samples = len(fane_data)
    logger.info(f"Total FANE samples discovered: {total_samples}")

    # 2. Strict 80/10/10 mathematical partitioning
    train_cutoff = int(total_samples * 0.80)
    val_cutoff = train_cutoff + int(total_samples * 0.10)

    train_count, val_count, test_count = 0, 0, 0
    
    # 3. Apply split tags directly to metadata dictionaries
    for i, sample in enumerate(fane_data):
        if i < train_cutoff:
            sample["split"] = "train"
            train_count += 1
        elif i < val_cutoff:
            sample["split"] = "validation"
            val_count += 1
        else:
            sample["split"] = "test"
            test_count += 1

    # 4. Overwrite original JSON to ensure DataLoader synchronization
    try:
        with open(target_json, "w", encoding="utf-8") as f:
            json.dump(fane_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        sys.exit(1)

    logger.info("FANE data partitioning complete.")
    logger.info(f"Distribution Target -> Train: {train_count} | Val: {val_count} | Test: {test_count}")

if __name__ == "__main__":
    balance_splits()
